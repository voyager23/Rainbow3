/*	
	*
	* maketable_v4.cu
	* calculate and sort a rainbow table
	* 
	* nvcc maketable_v4.cu ./obj/fname_gen.o ./obj/md5.o -o ./bin/mktab
	* From the parameters in rainbow.h, maketable produces an unsorted
	* table (new) and a table sorted on final hash (sorted). This is
	* used for merging into the main table.
	* 
	* The kernel will time out if more than 34*1024 threads are launched.
	* Defining 1 workunit as 32*1024 threads.
	* 
	* Introduce use of a unique table_ident (header->f1). Use this
	* ident for naming/selecting files.
	* Examples: merge_03ca59e3.rbt or sort_03ca59e3.rbt
	*
*/

#ifndef __CUDA__
	#define __CUDA__
#endif

//===========================Include code======================================
// main header files
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <endian.h>
#include <time.h>
#include <unistd.h>

// local header files
#include "../common/rainbow.h"
#include "md5.h"
// nvcc does not support externel calls
//#include "utils_2.h"
#include "utils_2.cu"
//======================================================================
__host__
void table_setup(TableHeader*,TableEntry*, uint32_t);
__host__
int sort_table(TableHeader*,TableEntry*);
__host__
uint32_t get_table_id(char *tid);
//======================================================================
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

//======================================================================
__host__
void table_setup(TableHeader *header, TableEntry *entry, uint32_t table_id) {
	int i,di;
	//unsigned int t_size=sizeof(TableHeader)+(sizeof(TableEntry)*T_ENTRIES);

	srand(time(NULL));
	//printf("Threads: %d Table_Size: %d\n",THREADS,t_size);
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
		
		// DEBUG Remove either or both for operational use
		(entry+i)->final_hash[0] = 0x776f6272;
		// END DEBUG
	}
	header->hdr_size = sizeof(TableHeader);
	header->entries = T_ENTRIES;
	header->links = LINKS;
	header->table_id =  table_id;	// Table Ident
	header->f2 = 0x3e3e3e3e;	// '>>>>'
	// Calculate the md5sum of the table entries
	md5_state_t state;
	md5_byte_t digest[16];
	
	md5_init(&state);
	for(i=0; i<T_ENTRIES; i++)
		md5_append(&state, (const md5_byte_t *)&(entry[i]), sizeof(TableEntry));
	md5_finish(&state, digest);

	// print md5sum for test purposes
	//for (di = 0; di < 16; ++di)
		//printf("%02x", digest[di]);
	//printf("\n");

	// Save the md5sum in check_sum slot
	for (di = 0; di < 16; ++di)
	    sprintf(header->check_sum + di * 2, "%02x", digest[di]);
	*(header->check_sum + di * 2) = '\0';
}
//======================================================================
__host__
int sort_table(TableHeader *header,TableEntry *entry) {
	// Revised code to read directly from memory

	TableEntry  *target, *found;
	int i;	//loop variable

	printf("Sorting %u Table Entries:-", header->entries);

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
//======================================================================

//=========================Device Code==================================

//======================================================================


//----------------------------------------------------------------------
__global__
void table_calculate(TableHeader *header, TableEntry *entry) {
	// revised 29Dec2011
	// The parameter is the base address of a large table of TableEntry(s)
	
	uint8_t  M[64];	// Initial string - zero padded and length in bits appended
	uint32_t W[64];	// Expanded Key Schedule
	uint32_t H[8];	// Hash
	int i = 0;		// working index
	uint64_t l = 0; // length of message
	uint8_t  B[64];	// store initial and working passwords here to protect original data
	uint32_t chain_idx, link_idx;

	// set up a pointer to initial_password & final_hash
	TableEntry *data = entry + blockIdx.x*blockDim.x + threadIdx.x;

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
		// link_idx = chain_idx + 0;

		// Reduce the Hash using the table_ident
		
		//???????????????????????????????????????????????
		link_idx = chain_idx + header->table_id;
		//???????????????????????????????????????????????
		
		// call reduce function
		reduce_hash(H,B,link_idx);
		
		// TODO: remove???? clear internal index
		i=0;
	} // --> end main loop

	// copy comp_hash to final hash
	for(i=0;i<8;i++) data->final_hash[i] = H[i];	

	// Ensure padding is zero
	data->pad=0x0000;
	__syncthreads();
}

void Check_CUDA_Error(const char *message,FILE *table, char *fname) {
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess) {
	fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
	fprintf(stderr,"Removing invalid file in rbt.\n");
	fclose(table);
	remove(fname);
	exit(1);
	}
}


uint32_t get_table_id(char *tid) {
	// Revised version of test code
	// Expects a pointer to a hex string
	// Returns equivalent uint32_t or zero on failure
	if(tid==NULL) {return(0);}
	uint32_t table_id = strtol(tid,NULL,16);
	return(table_id);
}
//=================================Main Code============================

int main(int argc, char **argv) {

	TableHeader *header, *dev_header;
	TableEntry  *entry,  *dev_entry;
	char table_file[81];
	char sort_file[81];
	FILE *table, *sort;
	int i,di,work_unit;
	uint32_t offset,table_id;

	cudaEvent_t start;
	cudaEvent_t end;
	float ms;
	
	size_t count;
	

	printf("========= Maketable_v4 =========\n");
	// Required parameter is the table identifier
	// Supplied as hex string of the form 0x12ab34cd
	// Stored internally as uint32_t
	// String form used to generate the table name
	// of the form "sort_0x12ab34cd.rbt
	
	if(argc != 2) {
		printf("Table Identifier missing.\nUsage: mktab 0x1234abcd\n");
		exit(1);
	}
	if((table_id=get_table_id(argv[1]))==0) {
		printf("Table index zero not permitted.\n");
		exit(1);
	}	
	
	fname_gen(sort_file,"sort",table_id);		// determine the filenames
	fname_gen(table_file,"new",table_id);		// at the same time so tmerge
	table=fopen(table_file,"w");				// can delete the unrequired files.
	if(table==NULL) {
		printf("Error - maketable_v4: Unable to open 'new' file.\n");
		return(1);
	}
	
	header = (TableHeader*)malloc(sizeof(TableHeader));
	entry = (TableEntry*)malloc(sizeof(TableEntry)*T_ENTRIES);
	if((header != NULL)&&(entry != NULL)) {
		
		printf("Preparing the table header and initial passwords.\n");
		table_setup(header,entry,table_id);	// initialise header and set initial passwords		

		// cudaMalloc space for table header
		HANDLE_ERROR(cudaMalloc((void**)&dev_header,sizeof(TableHeader)));				
		// copy header to device
		HANDLE_ERROR(cudaMemcpy(dev_header, header, sizeof(TableHeader), cudaMemcpyHostToDevice));
		
		// cudaMalloc space for 1 work unit in table body
		HANDLE_ERROR(cudaMalloc((void**)&dev_entry,sizeof(TableEntry)*DIMGRIDX*THREADS));
		
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		printf("Starting the first of %d work units....\n",WORKUNITS);
		// .....workunit...loop start.....
		for(work_unit=0; work_unit<WORKUNITS; work_unit++) {			
			
			// track position in table of entries
			offset = work_unit*DIMGRIDX*THREADS;
			
			// Copy entries to device
			HANDLE_ERROR(cudaMemcpy(dev_entry, entry+offset, sizeof(TableEntry)*DIMGRIDX*THREADS, cudaMemcpyHostToDevice));

			// =====Launch Kernel=====
			cudaEventRecord(start,0);
			cudaGetLastError();	// Clear cuda error flag
			
			table_calculate<<<DIMGRIDX,THREADS>>>(dev_header,dev_entry);
			
			cudaEventRecord(end,0);
			cudaEventSynchronize(end);			
			cudaEventElapsedTime(&ms,start,end);
			printf("Work unit %d completed in %4.2f ms.\n", work_unit,ms);
			
			Check_CUDA_Error("Error thrown after kernel launch",table,table_file);

			// copy back entries to host
			HANDLE_ERROR( cudaMemcpy(entry+offset, dev_entry, sizeof(TableEntry)*DIMGRIDX*THREADS, cudaMemcpyDeviceToHost) );
			// copy back header to host
			HANDLE_ERROR( cudaMemcpy(header, dev_header, sizeof(TableHeader), cudaMemcpyDeviceToHost) );			
			
		} // .....workunit...loop end.....
		
		fwrite(header,sizeof(TableHeader),1,table);
		fwrite(entry,sizeof(TableEntry),T_ENTRIES,table);		
		fclose(table);

		// sort on hash value	
		sort_table(header,entry);

		// Calculate the md5sum of the table entries
		md5_state_t state;
		md5_byte_t digest[16];

		md5_init(&state);
		for(i=0; i<T_ENTRIES; i++)
			md5_append(&state, (const md5_byte_t *)&(entry[i]), sizeof(TableEntry));
		md5_finish(&state, digest);

		// print md5sum for test purposes
		printf("Checksum for TableEntries: ");
		for (di = 0; di < 16; ++di)
			printf("%02x", digest[di]);
		printf("\n");

		// Save the md5sum in check_sum slot
		for (di = 0; di < 16; ++di)
	    sprintf(header->check_sum + di * 2, "%02x", digest[di]);
	    
	    // Open the sort file for writing
		sort=fopen(sort_file,"w");
		if(sort==NULL) {
			printf("Error - maketable_v3: Unable to open 'sort' file.\n");
			return(1);
		}
		// save sorted table to file 'sorted table'
		count =  fwrite(header,sizeof(TableHeader),1,sort);
		count += fwrite(entry,sizeof(TableEntry),header->entries,sort);
		if((count == header->entries + 1)&&(fclose(sort)==0)) {
			// ok to remove 'new' file
			printf("Sorted file successfully writen - deleting original.\n");
			if( remove( table_file ) != 0 )
				perror( "Error deleting file\n" );
			else
				printf( "original file successfully deleted\n" );
		}
	}
	printf("table_id: %u	sort_file %s\n",table_id,sort_file);
	// Clean up memory
	free(header);
	free(entry);
	cudaFree(dev_entry);
	cudaFree(dev_header);
	return(0);
}
