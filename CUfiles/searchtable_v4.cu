/*	
	*
	* searchtable_v4.cu
	* 05Apr2012
	* Incorporating new code from the threaded searchtable_v7.c
	* nvcc -Xlinker -lm searchtable_v3.cu table_utils.c md5.c
	* Updated version 22Apr2012
	*
*/

#ifndef __CUDA__
	#define __CUDA__
#endif

//===========================Include headers===============================

#include "rainbow.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <endian.h>
#include <time.h>

//=========================Declarations=================================
__global__
void kernel(TableHeader *header, TableEntry *entry);
//=========================Definitions==================================
__host__
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//Removed hash constant k[64] from here - 23Apr2012

//=========================Include functions and utilities==============
#include "freduce.cu"
#include "initHash.cu"
#include "sha256_txfm.cu"
#include "utils.cu"
//=========================Kernel=======================================
__global__
void kernel(TableHeader *header, TableEntry *entry) {
/*
	* revised 23Apr2012
	* The parameter is the base address of a large table of TableEntry(s)
	* Derived from table_calculate - given a target hash calculate a table
	* of candidate hashes
	* Algorithm takes input_hash and calculates final_hash and sublinks value.
*/

	uint8_t  M[64];	// Initial string - zero padded and length in bits appended
	uint32_t W[64];	// Expanded Key Schedule
	uint32_t H[8];	// Hash
	int i = 0;		// working index
	uint64_t l = 0; // length of message
	uint8_t  B[64];	// store initial and working passwords here to protect original data
	uint8_t *in,*out;
	
	int reduction_idx,count;

	uint thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
	

	if(thread_idx<LINKS) {
	
		// set up a pointer to input_hash & final_hash
		TableEntry *data = entry + thread_idx;
		// move target hash to H
		for(i=0;i<8;i++) H[i] = data->input_hash[i];

		reduction_idx = thread_idx;
		count = LINKS - thread_idx - 1;

		while(count > 0) {
			// Reduce hash to zero terminated password in B
			// Use freduce.cu
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

		__syncthreads();
	} // if(thread_idx<LINKS)
} // kernel

//=================================Main Code==================================

int main(int argc, char **argv) {

	const char *tables_path = "./rbt/RbowTab_tables_0.rbt";
	char rbt_file[128];
	FILE *fp_rbow, *fp_tables;
	TableHeader *header, *dev_header;
	TableEntry *entry, *dev_entry, *target, *check, *compare;
	TableHeader *subchain_header;
	TableEntry *subchain_entry;
	int i,di,dx;
	int solutions, collisions;
	
	printf("searchtable_v4 (cuda).\n");
	printf("Search a merged Rainbow Table for a selected password.\n");
	
	// Sanity checks. In this case assert (LINKS % THREADS)==0
	if((LINKS%THREADS)!=0) {
		printf("Sanity test in csearch failed.\n");
		exit(1);
	}
	// calculate number of blocks to launch
	const int threads=THREADS;				// threads per block
	const int blocks = (LINKS+THREADS-1)/threads;		// number of thread blocks

	target = (TableEntry*)malloc(sizeof(TableEntry));

#if(1)	
	// get test data - this is a known password/hash pair	
	srand(time(NULL));
	fp_rbow = fopen("./rbt/RbowTab_merge.rbt","r");
	get_rnd_table_entry(target, fp_rbow);
	fclose(fp_rbow);
	//Confirming selected target data.
	//Password: VI21tdV
	//Hash: 1b69ba30 d4c58d76 51d114fe c9f57390 73b5f0d4 84ee9870 f483f478 bce85866 
#else
	// setup known solution in 'target' for debug
	// associated hash is loaded into "input_hash"
	target->sublinks=0;
	strcpy(target->initial_password, "ZZ90syK");
	hash2uint32("b545a2399c2cb7dabf7ce8eae859789574266163d18a7296f715c34affca2b6f",
				(uint32_t*)&target->final_hash[0]);
#endif
	
	// confirmation	of target
	printf("\nConfirming selected target data.\nPassword: %s\nHash: ", target->initial_password);
	for(dx=0;dx<8;dx++) printf("%08x ", target->final_hash[dx]);
	printf("\n");
	

	// allocate space for subchain tables
	subchain_header = (TableHeader*)malloc(sizeof(TableHeader));
	subchain_entry  = (TableEntry*)malloc(sizeof(TableEntry)*LINKS);
	if((subchain_header==NULL)||(subchain_entry==NULL)) {
		printf("Error - searchtable: Subchain host memory allocation failed.\n");
		exit(1);
	}

	// set up the subchain table
	// subchain_header->hdr_size = sizeof(TableHeader);
	// subchain_header->f1 = 0x00000000U;
	for(i=0;i<LINKS;i++) {
		(subchain_entry+i)->sublinks=0;
		for(di=0;di<8;di++) {
			(subchain_entry+i)->input_hash[di] = target->final_hash[di];
			(subchain_entry+i)->final_hash[di] = 0xffffffff;
		}
	}

	// allocate device memory
	HANDLE_ERROR(cudaMalloc((void**)&dev_header,sizeof(TableHeader)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_entry,sizeof(TableEntry)*LINKS));

	// Copy entries to device
	HANDLE_ERROR(cudaMemcpy(dev_entry, subchain_entry, sizeof(TableEntry)*LINKS, cudaMemcpyHostToDevice));

	// launch kernel
	printf("Launching %d blocks of %d threads\n",blocks,threads);
	kernel<<<blocks,threads>>>(dev_header,dev_entry);

	// copy entries to host
	HANDLE_ERROR(cudaMemcpy(subchain_entry, dev_entry, sizeof(TableEntry)*LINKS, cudaMemcpyDeviceToHost));

	// Search Rainbow Tables for solution
	
	// ----------set up the Rainbow table----------
	fp_tables = fopen(tables_path,"r");
	if(fp_tables==NULL) {
		printf("Error - unable to open %s\n",tables_path);
		exit(1);
	}
	printf("Now looking for a valid solution\n");
	// look for a valid solution
	while((fscanf(fp_tables,"%s",rbt_file)) != EOF) {
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
		
		// try to match a subchain final_hash against final_hash in main table
		// if match found - report chain_index and link_index.
		printf("Looking for a matching chain...\n");
		collisions=0;
		solutions=0;
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
			compare = (TableEntry*)bsearch((subchain_entry+i), entry,
				header->entries, sizeof(TableEntry), hash_compare_32bit );

			if(compare!=NULL) {
				// printf("?");
				// Forward calculate the chain (entry+di) to (possibly) recover 
				// the password/hash pair.
				// check = (TableEntry*)malloc(sizeof(TableEntry)*(i+1));
				strcpy(check->initial_password,compare->initial_password);							
				compute_chain(check,i+1);			
				if(hash_compare_uint32_t((target)->final_hash,(check+i)->final_hash)==0) {
					printf("\033[31m");
					printf("\n=====SOLUTION FOUND===== \n%s\n",(check+i)->initial_password);
					for(dx=0;dx<8;dx++) printf("%08x ", (target)->final_hash[dx]);
					printf("\033[0m");
					solutions++;
					free(check);
					free(entry);
					free(header);
					goto found;
				} else { 
					//printf("- ");
					collisions++; 
				}
				//free(check);				 
			} else { 
				//printf(". "); 
			} // if (compare)				
		} // for[i=0]
		free(check);		
		free(entry);
		free(header);
	} // while !EOF
	found:
	// next two free() moved outside loop
		free(subchain_header);
		free(subchain_entry);
	// end move
	printf("\nThis run had %d collisions.\n",collisions);
	// free memory and file handles 
	fclose(fp_tables);
	free(target);
	return(0);
}


