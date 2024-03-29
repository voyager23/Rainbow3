/*	
	*
	* searchtable_v5.cu
	* 05Apr2012
	* Incorporating new code from the threaded searchtable_v7.c
	* nvcc -Xlinker -lm searchtable_v3.cu table_utils.c md5.c
	* Updated version 20May2012
	*
*/

#ifndef __CUDA__
	#define __CUDA__
#endif

//===========================Include headers===============================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <endian.h>
#include <time.h>
#include <unistd.h>

#include <iniparser.h>

#include "../common/rainbow.h"

// nvcc does not support external calls so inline code here...
#include "utils_2.cu"



__global__
void kernel(TableHeader *header, TableEntry *entry);

//=========================CUDA Error Handling==================================
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
			reduce_hash(H,B,(reduction_idx+header->table_id));

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

	char rbt_file[128];
	
	FILE *fp_rbow;
	TableHeader *header, *dev_header;
	TableEntry *entry, *dev_entry, *target, *check, *compare;
	TableHeader *subchain_header;
	TableEntry *subchain_entry;
	int i,di,dx;
	int solutions, collisions;
	
	int trials=1;		// number of trials to run
	int rand_pass=0;	// use random passwords (0=no)
	int c,loops=1;
	
	TargetList tlist;
	int tlist_flag = 0;
	 
	printf("This is searchtable_v5 (cuda).\n");
	printf("Search merged Rainbow Table(s) for a selected password.\n");
	
	// Get-set options here	    
	while ((c = getopt (argc, argv, "f:n:r")) != -1)
	switch (c) {
		case 'n':
			loops = trials = atoi(optarg);
			break;
		case 'r':
			rand_pass=1;
			break;
		case 'f':
			tlist_flag=make_target_list(optarg,&tlist);
			// if tlist_flag > 0, count will be number of entries
			// and idx will be zero
			if(tlist_flag > 0) {
				loops=trials=tlist_flag;
				printf("Found %d targets to check.\n",tlist_flag);
			} else {
				loops=0;
			}
			break;
		default:
			printf("Error in getopt - stopping.\n");
			exit(1);
	}
	
	// Non-option parameter path/to/rbow/table
	// Here we are setting the table to consult
	// Replace this code block with code to parse the .ini file
	
#if(0)
	if(argv[optind]==NULL) {
		printf("Usage:search [-f] [-n [trials] ] [-r] path/to/rbow/tab.rbt\n");
		exit(1);
	} else {
		strncpy(rbt_file,argv[optind],127);
		if(fopen(rbt_file,"r")==NULL) {
			printf("Error: Unable to open %s for reading.\n",rbt_file);
			exit(1);
		} else {
			// rbt_file now has table to consult
			printf("Path to Rbow Tab: %s\n",rbt_file);
		}
	}
#endif
	//--------------New Code-----------------
	
	#define PATHSIZE 128
	#define NPATHS 128

	typedef struct mfl {
		char (*path_ptr) [PATHSIZE];
		char master_file [NPATHS][PATHSIZE];
		int count;
	}MasterFileList;
	
	char *ini_name = "rainbow.ini";
	char **key_str;	
	int n_keys;	
	dictionary *ini;
	MasterFileList mfl;
	int major_loop_count, minor_loop_count;
	// initialise masterfilelist
	mfl.path_ptr = &mfl.master_file[0];
	mfl.count = 0;
	// open .ini file
	ini = iniparser_load(ini_name);
    if (ini==NULL) {
        fprintf(stderr, "cannot parse file: %s\n", ini_name);
        return -1 ;
    }    
    // debug dump file contents
    // iniparser_dump(ini, stderr);    
    
    // get number of keys and pointer to first key
    n_keys = iniparser_getsecnkeys(ini,"masterfiles");
    key_str= iniparser_getseckeys(ini,"masterfiles");
    
    // loop through key/value pairs copying
    // value to masterfilelist and incrementing
    // the count.
    for(i=0; i<n_keys;i++) {	
		printf("%s\n",*key_str);
		printf("%s\n",iniparser_getstring(ini,*key_str,NULL));
		strcpy((char*)mfl.path_ptr, iniparser_getstring(ini,*key_str++,NULL));
		mfl.path_ptr++;	mfl.count++;
    }
    printf("\n");
    // reset the pointer in mfl and loop through values recovered
	mfl.path_ptr = &mfl.master_file[0];
	for(i=0;i<mfl.count;i++) {
		printf("->%s<-\n",mfl.path_ptr++);
	}
	
	// Cleanup
	iniparser_freedict(ini);		
	
	//--------------End New Code-------------

	// Sanity checks. In this case assert (LINKS % THREADS)==0
	if((LINKS%THREADS)!=0) {
		printf("Sanity test in csearch failed.\n");
		exit(1);
	}
	// calculate number of blocks to launch
	const int threads=THREADS;						// threads per block
	const int blocks = (LINKS+THREADS-1)/threads;	// number of thread blocks	
	
	
	srand(time(NULL));
	// ###LOOP START###
	
	// reset the pointer in mfl and loop through values
	mfl.path_ptr = &mfl.master_file[0];
	for(major_loop_count=0; major_loop_count < mfl.count; major_loop_count++) {
		strcpy(rbt_file, (char*)mfl.path_ptr++);	// load rbt_file with a table name
		minor_loop_count=loops;
		tlist.idx=0;
		solutions=0;
	
		while(minor_loop_count-- > 0) {		
			if(tlist_flag > 0) {
				target = &tlist.target_list[tlist.idx++];
			} else {
				if(rand_pass == 0) {
					// Using known data
					// get test data - this is a known password/hash pair
					// from the main table
					printf("Using Known target.\n");
					target = (TableEntry*)malloc(sizeof(TableEntry));			
					fp_rbow = fopen(rbt_file,"r");
					get_rnd_table_entry(target, fp_rbow);
					fclose(fp_rbow);			
				} else {
					printf("Using Random target.\n");
					target = (TableEntry*)malloc(sizeof(TableEntry));
					make_rnd_target(target);
				}
			}		
			
			// confirmation	of target
			printf("Confirming selected target data.\nPassword: %s\nHash: ", target->initial_password);
			for(dx=0;dx<8;dx++) printf("%08x ", target->final_hash[dx]);
			printf("\n");
			
			// allocate space for subchain tables
			subchain_header = (TableHeader*)malloc(sizeof(TableHeader));		 
			subchain_entry  = (TableEntry*)malloc(sizeof(TableEntry)*LINKS);
			if((subchain_header==NULL)||(subchain_entry==NULL)) {
				printf("Error - searchtable: Subchain host memory allocation failed.\n");
				exit(1);
			} else { printf("Space for subchains allocated.\n"); }
			
			// setup the header data
			if((fp_rbow = fopen(rbt_file,"r"))==NULL) {
				// probable config file error - stop
				printf("Failed to open %s \n",rbt_file);				
				exit(1);
			}
			fread(subchain_header,sizeof(TableHeader),1,fp_rbow);
			fclose(fp_rbow);
			
			printf("Header data read.\n");

			// set up the subchain table
			for(i=0;i<LINKS;i++) {
				(subchain_entry+i)->sublinks=0;
				for(di=0;di<8;di++) {
					(subchain_entry+i)->input_hash[di] = target->final_hash[di];
					(subchain_entry+i)->final_hash[di] = 0xffffffff;
				}
			}
			
			printf("Subchains setup.\n");
			
			// allocate device memory
			HANDLE_ERROR(cudaMalloc((void**)&dev_header,sizeof(TableHeader)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_entry,sizeof(TableEntry)*LINKS));

			// Copy entries&header to device
			HANDLE_ERROR(cudaMemcpy(dev_entry,subchain_entry,sizeof(TableEntry)*LINKS,cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dev_header,subchain_header,sizeof(TableHeader),cudaMemcpyHostToDevice));
			
			// launch kernel
			printf("Launching %d blocks of %d threads\n",blocks,threads);
			kernel<<<blocks,threads>>>(dev_header,dev_entry);

			// copy entries to host
			HANDLE_ERROR(cudaMemcpy(subchain_entry,dev_entry,sizeof(TableEntry)*LINKS,cudaMemcpyDeviceToHost));
			// Should be no change to header
			HANDLE_ERROR(cudaMemcpy(subchain_header,dev_header,sizeof(TableHeader),cudaMemcpyDeviceToHost));
			
			// Search Rainbow Tables for solution
			printf("Now looking for a valid solution\n");
			fp_rbow = fopen(rbt_file,"r");
			if(fp_rbow==NULL) {
				printf("Error - unable to open %s\n",rbt_file);
				exit(1);
			} else {
				printf("Using table %s\n", rbt_file);
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
					// Forward calculate the chain (entry+di) to (possibly) recover 
					// the password/hash pair.
					strcpy(check->initial_password,compare->initial_password);							
					compute_chain(header,check,i+1);			
					if(hash_compare_uint32_t((target)->final_hash,(check+i)->final_hash)==0) {
						printf("\033[32m");
						printf("=====SOLUTION FOUND=====\n\t%s\n",(check+i)->initial_password);
						for(dx=0;dx<8;dx++) printf("%08x ", (target)->final_hash[dx]);
						printf("\033[0m");
						solutions++;
						free(check);
						free(entry);
						free(header);
						goto found;
					} else { 
						collisions++; 
					}
					//free(check);				 
				} else { 
				} // if (compare)				
			} // for(i=0; ...)
			if(i==LINKS) {
				printf("\033[31m");
				printf("=====No solution found for this hash=====\n");
				for(dx=0;dx<8;dx++) printf("%08x ", (target)->final_hash[dx]);
				printf("\033[0m");
			}
			free(check);		
			free(entry);
			free(header);
			// end valid solution search
			found:
			// next two free() moved outside loop
			free(subchain_header);
			free(subchain_entry);
			printf("\nThis trial had %d collisions.\n",collisions);
			// free memory and file handles 
		} // ###LOOP END###
		printf("This run found %d/%d solutions.\n===========\n",solutions,trials);
	}
	if (tlist_flag==0) free(target);
	
	return(0);
}

