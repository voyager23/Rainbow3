/*	
	* Performance testing version
	* 04Apr2012
	*
*/

#ifndef __CUDA__
	#define __NO_CUDA__
#endif

//===========================Include code======================================
#include "rainbow.h"
#include "table_utils.h"
#include "freduce.h"
#include "fname_gen.h"

#include "md5.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <endian.h>
#include <time.h>
#include <pthread.h>



//==============Declarations==================================================
void table_setup(TableHeader*,TableEntry*);
int sort_table(TableHeader*,TableEntry*);
void initHash(uint32_t *h);
void sha256_transform(uint32_t *w, uint32_t *H);
void *subchain_hash_thread(void *p);
//=============================================================================

typedef struct pthread_data {
	pthread_t pthread_id;
	void *exit_status;
	//-------------------
	TableHeader *header;
	TableEntry  *entry;
	int entry_idx;
	//-------------------
} PthreadData;

void table_setup(TableHeader *header, TableEntry *entry) {
	int i,di;
	unsigned int t_size=sizeof(TableHeader)+(sizeof(TableEntry)*DIMGRIDX*THREADS);

	srand(time(NULL));
	printf("Threads: %d Table_Size: %d\n",THREADS,t_size);
	for(i=0; i<THREADS*DIMGRIDX; i++) {
		// Random password type 'UUnnllU'
		(entry+i)->initial_password[0]= (rand() % 26) + 'A';
		(entry+i)->initial_password[1]= (rand() % 26) + 'A';
		(entry+i)->initial_password[2]= (rand() % 10) + '0';
		(entry+i)->initial_password[3]= (rand() % 10) + '0';
		(entry+i)->initial_password[4]= (rand() % 26) + 'a';
		(entry+i)->initial_password[5]= (rand() % 26) + 'a';
		(entry+i)->initial_password[6]= (rand() % 26) + 'A';
		(entry+i)->initial_password[7]= '\0';
		// DEBUG
		(entry+i)->final_hash[0] = 0x776f6272;
	}
	header->hdr_size = sizeof(TableHeader);
	header->entries = THREADS*DIMGRIDX;
	header->links = LINKS;
	header->f1 =  rand()%1000000000;	// Table Index
	header->f2 = 0x3e3e3e3e;			// '>>>>'
	// Calculate the md5sum of the table entries
	md5_state_t state;
	md5_byte_t digest[16];
	
	md5_init(&state);
	for(i=0; i<THREADS*DIMGRIDX; i++)
		md5_append(&state, (const md5_byte_t *)&(entry[i]), sizeof(TableEntry));
	md5_finish(&state, digest);

	// print md5sum for test purposes
	for (di = 0; di < 16; ++di)
		printf("%02x", digest[di]);
	printf("\n");

	// Save the md5sum in check_sum slot
	for (di = 0; di < 16; ++di)
	    sprintf(header->check_sum + di * 2, "%02x", digest[di]);
	*(header->check_sum + di * 2) = '\0';
}

//=============================================================================
int sort_table(TableHeader *header,TableEntry *entry) {
	// Revised code to read directly from memory
	// Write sorted table to fout
	TableEntry  *target, *found;
	int i;	//loop variable

	printf("Sorting %u Table Entries:-\n", header->entries);


	qsort(entry, header->entries, sizeof(TableEntry), hash_compare_32bit);

	// select a hash at random to act as test target
	srand(time(NULL));
	target = (entry + rand()%header->entries);
	printf("\nRandom target: %s Hash: ", target->initial_password);
	for(i=0;i<8;i++) printf("%08x ", target->final_hash[i]);

	found = (TableEntry*)bsearch(target, entry, header->entries, sizeof(TableEntry), hash_compare_32bit);
	if(found != NULL) {
		printf("\nLocated target %s Hash: ", found->initial_password);
		for(i=0;i<8;i++) printf("%08x ", found->final_hash[i]);
		printf("\n");
	} else {
		printf("\nTarget hash not found?\n");
	}
	// end test target

	return(0);
}

//========================Hash_Calculate kernel=================================

void *subchain_hash_thread(void *pthread_arg) {
/*
	* subchain_hash_thread()
	* Given a hash value and its position (index) in the chain, determine
	* the corresponding final hash. This is then used to find a candidate
	* chain in the main table
*/
	PthreadData *mydata;
	mydata = (PthreadData*)pthread_arg;
	// TableHeader *header = mydata->header;
	TableEntry  *entry  = mydata->entry;
	uint thread_idx = mydata->entry_idx;	
	
	uint8_t  M[64];	// Initial string - zero padded and length in bits appended
	uint32_t W[64];	// Expanded Key Schedule
	uint32_t H[8];	// Hash
	int i = 0;		// working index
	uint64_t l = 0; // length of message
	uint8_t  B[64];	// store initial and working passwords here to protect original data
	uint8_t *in,*out;	
	int reduction_idx,count;	

	if(thread_idx<LINKS) {
	
		// set up a pointer to input_hash & final_hash
		TableEntry *data = entry + thread_idx;
		// move target hash to H
		for(i=0;i<8;i++) H[i] = data->input_hash[i];

		reduction_idx = thread_idx;
		count = LINKS - thread_idx - 1;

		while(count > 0) {
			// Reduce hash to zero terminated password in B
			reduce_hash(H,B,reduction_idx);

			// copy zero terminated string from B to M and note length
			in = B;
			out = M;
			i=0; l=0;
			while(in[i] != 0x00) {
				out[i] = in[i];
				i++;
				l++;
			}
			out[i++] = 0x80;
			// zero fill
			while(i < 56) out[i++]=0x00;
			/*
				 * The hash algorithm uses 32 bit (4 byte words).
				 * On little endian machines (Intel) the constants
				 * are stored lsb->msb internally. To match this the WORDS
				 * of the input message are subject to endian swap.
			*/
			uint8_t *x = M;
			int y;
			for(y=0; y<14; y++) {
				// long swap
				*(x+3) ^= *x;
				*x     ^= *(x+3);
				*(x+3) ^= *x;
				// short swap
				*(x+2) ^= *(x+1);
				*(x+1) ^= *(x+2);
				*(x+2) ^= *(x+1);
				// move pointer up
				x += 4;
			}
			// need a 32 bit pointer to store length as 2 words
			l*=8;	//length in bits
			uint32_t *p = (uint32_t*)&l;
			uint32_t *q = (uint32_t*)&out[i];
			*q = *(p+1);
			*(q+1) = *p;

			// The 64 bytes in the message block can now be used
			// to initialise the 64 4-byte words in the message schedule W[64]
			// REUSE i
			uint8_t *r = (uint8_t*)M;
			uint8_t *s = (uint8_t*)W;
			for(i=0;i<64;i++) s[i] = r[i];
			for(i=16;i<64;i++) W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];

			// set initial hash values
			initHash(H);

			// Now calc the hash
			sha256_transform(W,H);

			// update the counters
			reduction_idx += 1;
			count -= 1;

		} // while(count>0)

		// copy comp_hash to final hash
		for(i=0;i<8;i++) data->final_hash[i] = H[i];
		data->sublinks = thread_idx;

	} // if(thread_idx<LINKS)
	void *void_ptr=NULL;
	return(void_ptr);
} // hash_calculate

//=================================Main Code==================================

int main(int argc, char **argv) {
/*
	* Version 7
	* exec: csearch
	* Selects and confirms a target
	* Searches for known target
*/

#define NUM_PTHREADS 8
	// in this case assert (LINKS % NUM_PTHREADS)==0
	if((LINKS%NUM_PTHREADS)!=0) {
		printf("Sanity test in csearch6 failed.\n");
		exit(1);
	}
	PthreadData pthread_data_array[NUM_PTHREADS];
	
	const char *tables_path = "./rbt/RbowTab_tables_0.rbt";
	char rbt_file[128];
	FILE *fp_rbow, *fp_tables;
	TableHeader *header;
	TableEntry *entry, *target, *check, *compare;
	TableHeader *subchain_header;
	TableEntry *subchain_entry;
	int t,n,i,di,dx;
	int link_index, solutions, collisions, found;
	
	printf("Search a merged Rainbow Table for a selected password.\n");
	
	// get test data
	target = (TableEntry*)malloc(sizeof(TableEntry));
	srand(time(NULL));
	fp_rbow = fopen("./rbt/RbowTab_merge.rbt","r");
	get_rnd_table_entry(target, fp_rbow);
	fclose(fp_rbow);
	
	printf("\nConfirming selected target data.\nPassword: %s\nHash: ", target->initial_password);
	for(dx=0;dx<8;dx++) printf("%08x ", target->final_hash[dx]);
	printf("\n");
	
	// ----------now set up the Rainbow table----------
	fp_tables = fopen(tables_path,"r");
	if(fp_tables==NULL) {
		printf("Error - unable to open %s\n",tables_path);
		exit(1);
	}
	// ----------allocate space for subchain tables----------
	subchain_entry  = (TableEntry*)malloc(sizeof(TableEntry) * LINKS);
	
	subchain_header = (TableHeader*)malloc(sizeof(TableHeader));
	
	if((subchain_header==NULL)||(subchain_entry==NULL)) {
		printf("Error - subchain memory allocation failed.\n");
		exit(1);
	}

	// -----Given the target pass/hash pair, calculate a set of candidate final_hashes-----		
	// first set up the subchain table header
	subchain_header->hdr_size = sizeof(TableHeader);
	subchain_header->f1 = 0x00000000U;
	for(i=0;i<LINKS;i++) {
		(subchain_entry+i)->sublinks=0;
		// now set up the subchain table entries by copying the same hash
		// to every input hash.
		for(di=0;di<8;di++) {
			(subchain_entry+i)->input_hash[di] = (target)->final_hash[di];
			(subchain_entry+i)->final_hash[di] = 0xffffffff;
		}
	}

	// calculate a set of candidate final_hashes
	printf("\n: Generating by threads a table of candidate final_hashes...\n");
	
	for(link_index=0; link_index<LINKS; link_index++) {
		for(t=0; t < NUM_PTHREADS; t++) {
			// Set up the entry in thread_data_array
			pthread_data_array[t].header = subchain_header;
			pthread_data_array[t].entry = subchain_entry;
			pthread_data_array[t].entry_idx = (link_index + t);
			// launch thread
			pthread_create(&pthread_data_array[t].pthread_id, NULL, 
				subchain_hash_thread, (void *) &pthread_data_array[t]);
		}
		// join threads
		for(t=0; t<NUM_PTHREADS; t++)
			pthread_join(pthread_data_array[t].pthread_id, NULL);
	}
	
	printf("Now looking for a valid solution\n");
	// look for a valid solution
	solutions=0; found=0;
	while((n = fscanf(fp_tables,"%s",rbt_file)) != EOF) {
		fp_rbow = fopen(rbt_file,"r");
		if(fp_rbow==NULL) {
			printf("Error - unable to open %s\n",rbt_file);
			exit(1);
		} else {
			printf("\nUsing table %s\n", rbt_file);
		}
		header = (TableHeader*)malloc(sizeof(TableHeader));
		fread(header,sizeof(TableHeader),1,fp_rbow);
		entry = (TableEntry*)malloc(sizeof(TableEntry)*header->entries);
		fread(entry,sizeof(TableEntry),header->entries,fp_rbow);
		fclose(fp_rbow);	
		
		show_table_header(header);
		
		// try to match a subchain final_hash against final_hash in main table
		// if match found - report chain_index and link_index.
		printf("Looking for a matching chain...\n");
		collisions=0;
		
		check = (TableEntry*)malloc(sizeof(TableEntry)*(LINKS));
		
		for(i=0;i<LINKS;i++) {				
			// left points to candidate
			// left = (subchain_entry+i)->final_hash;
			// right points to merged table ordered by ascending final_hash
			// right = (entry+di)->final_hash;
			/*
			 * if compare == 1,  candidate > merged, continue
			 * if compare == -1, candidate < merged, break
			 */
			
			compare = (TableEntry*)bsearch((subchain_entry+i),
								entry,
								header->entries,
								sizeof(TableEntry),
								hash_compare_32bit );
			
			if(compare!=NULL) {
				printf("?");
				// Forward calculate the chain (entry+di) to (possibly) recover 
				// the password/hash pair.
				// check = (TableEntry*)malloc(sizeof(TableEntry)*(i+1));
				strcpy(check->initial_password,(compare)->initial_password);
				
				compute_chain(check,i+1);

				show_table_entries(check,0,i);
					
				if(hash_compare_uint32_t((target)->final_hash,(check+i)->final_hash)==0) {
					printf("\033[31m");
					printf("\n=====SOLUTION FOUND===== \n%s\n",(check+i)->initial_password);
					for(dx=0;dx<8;dx++) printf("%08x ", (target)->final_hash[dx]);
					// debug
					printf("\nSolution chain starts %s\n",compare->initial_password);
					printf("Final Hash is ");
					for(dx=0;dx<8;dx++) printf("%08x ", (compare)->final_hash[dx]);
					printf("\n");
					// end debug
					printf("\033[0m");
					solutions++;
					free(check);
					free(entry);
					free(header);
					goto found;
				} else { 
					printf("- ");
					collisions++; 
				}
				//free(check);				 
			} else { printf(". "); }				
		} // for[i=0]
		
		free(check);		
		
		free(entry);
		free(header);
	}
	found:
	// next two free() moved outside loop
		free(subchain_header);
		free(subchain_entry);
	// end move
	printf("\nThis run found %d solution and had %d collisions.\n",solutions,collisions);
	// free memory and file handles 
	fclose(fp_tables);
	free(target);
	return(0);
}
