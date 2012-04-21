/*	
	*
	* maketable_v3.cu
	* calculate and sort a rainbow table
	* nvcc -Xlinker -lm maketable_v3.cu table_utils.c md5.c
	* Modify to produce time stamped files
	*
*/

#ifndef __CUDA__
	#define __CUDA__
#endif

//===========================Include code======================================
#include "md5.h"
#include "rainbow.h"
#include "table_utils.h"
#include "freduce.h"
#include "fname_gen.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <endian.h>
#include <time.h>

// Hash constants
__constant__	uint32_t k[64] = {
	   0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	   0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	   0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	   0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	   0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	   0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	   0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	   0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

//=============================================================================
void table_setup(TableHeader*,TableEntry*);
int sort_table(TableHeader*,TableEntry*);

__device__ void initHash(uint32_t *h);
__device__ void sha256_transform(uint32_t *w, uint32_t *H);
__global__ void table_calculate(TableHeader*,TableEntry*);
//=============================================================================

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
//=========================Device Code=========================================
// Reduction function goes here
//=============================================================================
__device__ void initHash(uint32_t *h) {
	h[0] = 0x6a09e667;
	h[1] = 0xbb67ae85;
	h[2] = 0x3c6ef372;
	h[3] = 0xa54ff53a;
	h[4] = 0x510e527f;
	h[5] = 0x9b05688c;
	h[6] = 0x1f83d9ab;
	h[7] = 0x5be0cd19;
}
//=============================================================================
__device__ void sha256_transform(uint32_t *w, uint32_t *H) {
	//working variables 32 bit words
	int i;
	uint32_t a,b,c,d,e,f,g,h,T1,T2;

	a = H[0];
	b = H[1];
	c = H[2];
	d = H[3];
	e = H[4];
	f = H[5];
	g = H[6];
	h = H[7];
   
   for (i = 0; i < 64; ++i) {  
      T1 = h + EP1(e) + CH(e,f,g) + k[i] + w[i];
      T2 = EP0(a) + MAJ(a,b,c);
      h = g;
      g = f;
      f = e;
      e = d + T1;
      d = c;
      c = b;
      b = a;
      a = T1 + T2;
  }      
    // compute single block hash value
	H[0] += a;
	H[1] += b;
	H[2] += c;
	H[3] += d;
	H[4] += e;
	H[5] += f;
	H[6] += g;
	H[7] += h;
}

//-----------------------------------------------------------------------------
   __global__ void table_calculate(TableHeader *header, TableEntry *entry) {
	// revised 29Dec2011
	// The parameter is the base address of a large table of TableEntry(s)
	
	uint8_t  M[64];	// Initial string - zero padded and length in bits appended
	uint32_t W[64];	// Expanded Key Schedule
	uint32_t H[8];	// Hash
	int i = 0;		// working index
	uint64_t l = 0; // length of message
	uint8_t  B[64];	// store initial and working passwords here to protect original data
	uint chain_idx, link_idx;

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
		link_idx = chain_idx + 0;

		// Reduce the Hash and store in B using inline code
		// link_idx = chain_idx + header->f1;

		
		// call reduce function
		reduce_hash(H,B,link_idx);
		// clear internal index
		i=0;
	} // --> end main loop

	// copy comp_hash to final hash
	for(i=0;i<8;i++) data->final_hash[i] = H[i];	

	// Ensure padding is zero
	data->pad=0x0000;
	__syncthreads();
}

//=================================Main Code==================================
int main(int argc, char **argv) {

	TableHeader *header, *dev_header;
	TableEntry  *entry,  *dev_entry;
	char table_file[81];
	char sort_file[81];
	FILE *table, *sort;
	int i,di;

	printf("maketable_v3.\n");
	fname_gen(table_file,"new64");
	table=fopen(table_file,"w");
	if(table==NULL) {
		printf("Error - maketable_v3: Unable to open table file.\n");
		return(1);
	}
	fname_gen(sort_file,"sort64");
	sort=fopen(sort_file,"w");
	if(sort==NULL) {
		printf("Error - maketable_v3: Unable to open sort file.\n");
		return(1);
	}
	
	header = (TableHeader*)malloc(sizeof(TableHeader));
	entry = (TableEntry*)malloc(sizeof(TableEntry)*DIMGRIDX*THREADS);
	if((header != NULL)&&(entry != NULL)) {

		table_setup(header,entry);	// initialise header and set initial passwords

		// cudaMalloc spave for table header
		cudaMalloc((void**)&dev_header,sizeof(TableHeader));
		// copy header to device
		cudaMemcpy(dev_header, header, sizeof(TableHeader), cudaMemcpyHostToDevice);

		// cudaMalloc space for table body
		cudaMalloc((void**)&dev_entry,sizeof(TableEntry)*DIMGRIDX*THREADS);
		// Copy entries to device
		cudaMemcpy(dev_entry, entry, sizeof(TableEntry)*DIMGRIDX*THREADS, cudaMemcpyHostToDevice);

		// call device code

		// launch kernel
		table_calculate<<<DIMGRIDX,THREADS>>>(dev_header,dev_entry);

		// copy entries to host
		cudaMemcpy(entry, dev_entry, sizeof(TableEntry)*DIMGRIDX*THREADS, cudaMemcpyDeviceToHost);
		// copy header to host
		cudaMemcpy(dev_header, header, sizeof(TableHeader), cudaMemcpyDeviceToHost);

		// save table to file
		fwrite(header,sizeof(TableHeader),1,table);
		fwrite(entry,sizeof(TableEntry),header->entries,table);
		fclose(table);

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
		printf("\n");

		// Save the md5sum in check_sum slot
		for (di = 0; di < 16; ++di)
	    sprintf(header->check_sum + di * 2, "%02x", digest[di]);

		// save sorted table to file
		fwrite(header,sizeof(TableHeader),1,sort);
		fwrite(entry,sizeof(TableEntry),header->entries,sort);
		fclose(sort);

		// DEBUG: display entries from table
			int idx;
			for(idx=0; idx < DIMGRIDX*THREADS; idx++) {
				printf("\nEntry[%d]:%s \nFinal Hash: ",idx,(entry+idx)->initial_password);
				for(i=0;i<8;i++) printf("%08x ",(entry+idx)->final_hash[i]);
				printf("\nLinks: %d\n",LINKS);				
			}

	}
	// Clean up memory
	free(header);
	free(entry);
	cudaFree(dev_entry);
	cudaFree(dev_header);
	return(0);
}
