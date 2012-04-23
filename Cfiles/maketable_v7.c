//      Filename maketable_v7.c
//		derived from pthread_01.c
//		Compiles to cmaketab
//      
//      Copyright 2012 Michael <mike@mike-n110>
//      
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//      
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//      
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.
//		Testbed code for a threaded version of maketable_vN

/* 
 * Output from cmaketab_v6
 * Random target: LE28ejM 
 * Hash: c374af28 cbf22be1 12b71cfe f4c8c118 3587646c a03124a7 0afd01a4 b037748c 
 * Located target LE28ejM 
 * Hash: c374af28 cbf22be1 12b71cfe f4c8c118 3587646c a03124a7 0afd01a4 b037748c
 * 
 * Output from pthread_01
 * Table index: 0 Init_pass LE28ejM
 * c374af28 cbf22be1 12b71cfe f4c8c118 3587646c a03124a7 0afd01a4 b037748c 
 * Table index: 1 Init_pass LE28ejM
 * c374af28 cbf22be1 12b71cfe f4c8c118 3587646c a03124a7 0afd01a4 b037748c
*/


#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "md5.h"
#include "rainbow.h"
#include "table_utils.h"
#include "freduce.h"
#include "fname_gen.h"

// Declarations
void initHash(uint32_t *h);
void sha256_transform(uint32_t *w, uint32_t *H);
void table_setup(TableHeader *header, TableEntry *entry);
int sort_table(TableHeader *header,TableEntry *entry);
void *chain_calc_thread(void*p);
void *show_progress(void *n);

// Definitions
typedef struct pthread_data {
	pthread_t pthread_id;
	void *exit_status;
	//-------------------
	TableHeader *header;
	TableEntry  *entry;
	int entry_idx;
	//-------------------
} PthreadData;

//============================================================================
void table_setup(TableHeader *header, TableEntry *entry) {
	int i,di;
	//unsigned int t_size=sizeof(TableHeader)+(sizeof(TableEntry)*DIMGRIDX*THREADS);

	printf("Table Setup\nLinks: %d Table_Size: %d entries\n",LINKS,T_ENTRIES);

	srand(time(NULL));
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
	printf("Empty md5sum: ");
	for (di = 0; di < 16; ++di)
		printf("%02x", digest[di]);
	printf("\n");

	// Save the md5sum in check_sum slot
	for (di = 0; di < 16; ++di)
	    sprintf(header->check_sum + di * 2, "%02x", digest[di]);
	*(header->check_sum + di * 2) = '\0';
}
//=======================================================================
int sort_table(TableHeader *header,TableEntry *entry) {
	// Revised code to read directly from memory

	TableEntry  *target, *found;
	int i;	//loop variable

	printf("Sorting %u Table Entries:-\n", header->entries);


	qsort(entry, header->entries, sizeof(TableEntry), hash_compare_32bit);

	// select a hash at random to act as test target
	srand(time(NULL));
	target = (entry + rand()%header->entries);
	printf("Random target: %s Hash: ", target->initial_password);
	for(i=0;i<8;i++) printf("%08x ", target->final_hash[i]);

	found = (TableEntry*)bsearch(target, entry, header->entries, 
		sizeof(TableEntry), hash_compare_32bit);
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
//=============================================================================== 
void *chain_calc_thread(void *pthread_arg) {
	
	PthreadData *mydata;
	mydata = (PthreadData*)pthread_arg;
	//TableHeader *header = mydata->header;
	TableEntry  *entry  = mydata->entry;
	int entry_idx = mydata->entry_idx;	
	
	uint8_t  M[64];	// Initial string - zero padded and length in bits appended
	uint32_t W[64];	// Expanded Key Schedule
	uint32_t H[8];	// Hash
	int i = 0;		// working index
	uint64_t l = 0; // length of message
	uint8_t  B[64];	// store initial and working passwords here to protect original data
	uint chain_idx, link_idx;

	// set up a pointer to initial_password & final_hash
	TableEntry *data = entry + entry_idx;

	// set up to read in the trial string into the B buffer
	uint8_t *in = (uint8_t*)data->initial_password;
	uint8_t *out = B;
	// copy zero terminated string
	i=0;
	while(in[i] != 0x00) {
		out[i] = in[i];
		i++;
	}
	out[i] = 0x00;

	// ---> main loop buffer B contains the zero term string
	for(chain_idx=0; chain_idx<LINKS; chain_idx++) {
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
		/*The hash algorithm uses 32 bit (4 byte words).
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

		// For testing use 0 as table index
		link_idx = chain_idx + 0;
		
		// call reduce function
		reduce_hash(H,B,link_idx);
		// clear internal index
		i=0;
	} // --> end main loop

	// copy comp_hash to final hash
	for(i=0;i<8;i++) data->final_hash[i] = H[i];	

	// Ensure padding is zero
	data->pad=0x0000;	
	
	//=======================================================================================
	
	pthread_exit(NULL);
}

void *show_progress(void *n) {
	int *num = (int *)n;
	if((*num % (T_ENTRIES/DIMGRIDX) == 0)) 
	printf("%d/%d\t",*num,T_ENTRIES);
	pthread_exit(NULL);
}
// ============================Main Code================================
int main(int argc, char** argv)
{
#define NUM_PTHREADS 8

#define SANITY (((DIMGRIDX*THREADS)>0)&&((DIMGRIDX*THREADS)%NUM_PTHREADS)==0)

	PthreadData pthread_data_array[NUM_PTHREADS];
	//pthread_t prog;
	//void *exit_status;
	
	FILE *sort;
	char sort_file[81];
	int t,i,di,r1,r2,n;
	
	printf("Compute and merge a Rainbow Table.\nTest with ./cperf\n");

	if(!SANITY) {
		printf("Sanity test failed.\n");
		exit(1);
	}
	
	// allocate memory for a table [T_ENTRIES]
	TableHeader *header = (TableHeader*)malloc(sizeof(TableHeader));
	TableEntry  *entry  = (TableEntry*)malloc(sizeof(TableEntry)*T_ENTRIES);
	
	// table setup
	table_setup(header,entry);
	printf("Computing the table data using %d threads.\n",NUM_PTHREADS);
	for(n=0; n<T_ENTRIES; n+=NUM_PTHREADS) {		
		for(t=0; t<NUM_PTHREADS; t++) {			
			// table_setup has loaded a set of random passwords
			// into the table
			// Set up the entry in thread_data_array
			pthread_data_array[t].header = header;
			pthread_data_array[t].entry = entry;
			pthread_data_array[t].entry_idx = (n + t);
			// launch thread
			pthread_create(&pthread_data_array[t].pthread_id, NULL, 
				chain_calc_thread, (void *) &pthread_data_array[t]);
		}	
		// join threads
		for(t=0; t<NUM_PTHREADS; t++)
			pthread_join(pthread_data_array[t].pthread_id, NULL);
	}
	printf("--------------------\n");
	
#if(0)
	// display result
	
	for(t=0; t<T_ENTRIES; t++) {
		printf("Table index: %d Init_pass %s\n",t, (entry+t)->initial_password);
		for(i=0;i<8;i++) printf("%08x ", (entry+t)->final_hash[i]);
		printf("\n");
	}
#endif

	// process and save table
	// sort on hash value	
	sort_table(header,entry);

	// Calculate the md5sum of the table entries
	md5_state_t state;
	md5_byte_t digest[16];

	md5_init(&state);
	for(i=0; i<THREADS*DIMGRIDX; i++)
		md5_append(&state, (const md5_byte_t *)&(entry[i]), sizeof(TableEntry));
	md5_finish(&state, digest);

	// print md5sum for test purposes
	printf("Checksum for TableEntries: ");
	for (di = 0; di < 16; ++di)
		printf("%02x", digest[di]);
	printf("\n\n");

	// Save the md5sum in check_sum slot
	for (di = 0; di < 16; ++di)
	sprintf(header->check_sum + di * 2, "%02x", digest[di]);
	
	fname_gen(sort_file,"sort",DIMGRIDX*THREADS);
	sort=fopen(sort_file,"w");
	if(sort==NULL) {
		printf("Error - maketable_v3: Unable to open sort file.\n");
		return(1);
	}

	// save sorted table to file
	r1=fwrite(header,sizeof(TableHeader),1,sort);
	r2=fwrite(entry,sizeof(TableEntry),header->entries,sort);
	fflush(sort);	//TODO: Is this call required since fclose will flush buffers?
	fclose(sort);
	
	// hand the sorted file to the tmerge function
	// for inclusion in the main merged file
	tmerge(sort_file,"rbt/RbowTab_merged.rbt.new");

	// cleanup
	free(entry);
	free(header);
	pthread_exit(NULL);
	return 0;
}
