/*	
	*
	* searchtable_v3.cu
	* nvcc -Xlinker -lm searchtable_v3.cu table_utils.c md5.c
	*
*/

#ifndef __CUDA__
	#define __CUDA__
#endif

//===========================Include code======================================
#include "freduce.h"
#include "md5.h"
#include "rainbow.h"
#include "table_utils.h"
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


//==============Declarations==================================================
void table_setup(TableHeader*,TableEntry*);
int sort_table(TableHeader*,TableEntry*);
__device__ void initHash(uint32_t *h);
__device__ void sha256_transform(uint32_t *w, uint32_t *H);
__global__ void hash_calculate(TableSubChnHeader *header, TableSubChnEntry *entry);
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

//=========================Device Code=========================================

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

//========================Hash_Calculate kernel=================================

__global__ void hash_calculate(TableSubChnHeader *header, TableSubChnEntry *entry) {
/*
	* revised 29Dec2011
	* The parameter is the base address of a large table of TableSubChnEntry(s)
	* Derived from table_calculate - given a target hash calculate a table
	* of candidate hashes
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
		TableSubChnEntry *data = entry + thread_idx;
		// move target hash to H
		for(i=0;i<8;i++) H[i] = data->input_hash[i];

		reduction_idx = thread_idx;
		count = LINKS - thread_idx - 1;

		while(count > 0) {
			// Reduce hash to zero terminated password in B
			reduce_hash(H,B,reduction_idx);

			//DEBUG capture reduced password
			i=0; do { data->debug[i]=B[i]; i += 1; } while(B[i-1] != '\0');
			//END DEBUG

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
} // hash_calculate

//=================================Main Code==================================

int main(int argc, char **argv) {
/*
RbowTab_sort64_1326902231.rbt
LINKS=16
INITIAL: AV57ujG
Working: AV57ujG
Hash: a376d88ccefa86544188491587823810b7675c2d60d934c2aee4b4cc577fdb2d
REDUCED: EM86pgA
Working: EM86pgA
Hash: c018e2b13bc51f0eee588910807c2c3d6b50d4155888d5a8e59fc576c24c23bb
REDUCED: CK11puQ
Working: CK11puQ
Hash: 9a247edf12d5591e0cac1a88057ce9a07e6ca6963c427380820e990622ffa4eb
REDUCED: HS61iuK
Working: HS61iuK
Hash: 710cc3f23de472081d194bc3e823757acb7db1b4200ee8ffb93bb4282f1d76ee
REDUCED: LC54cnV
Working: LC54cnV
Hash: 8b7b09072dfd95224fe201d7375df5d8a7e715b3231cb4b0596761c8c7855cf6
REDUCED: BJ23hoU
Working: BJ23hoU
Hash: d51324a9e72a1cb494ef1898354af57f69b6d1b45ee2b98d4de6a9c58026c610
REDUCED: EZ38jlK
Working: EZ38jlK
Hash: 8bac5f834533050f8e5e40abf4671bebdf3d73dd00eb894b35def62569f12e51
REDUCED: RG15zuD
Working: RG15zuD
Hash: 1253dd57804fb990a65aa5eba3f69b3dc321a05a1077038a38133b8bf804b0af
REDUCED: QL17yyU
Working: QL17yyU
Hash: 9d12dc869e0dce7c45eb202e68a59d52e6aebaf919b9c94a232cf5bb8063878e
REDUCED: QO81elI
Working: QO81elI
Hash: 078b7d5190274041d8e66e4d33efcfe83e9b38c704cf58580fcaa7ad21e8d0bc
REDUCED: GH83kqL
Working: GH83kqL
Hash: 6a3cd82018b1c0471b2e4a6124357193f8fc449fa564d4eacb90c0473cfae72a
REDUCED: KA31tqR
Working: KA31tqR
Hash: d094148327d1e8ea41c38b54292aa98f17e08395d0a51e5eb97ffa510a71f9a0
REDUCED: KS73hnY
Working: KS73hnY
Hash: 52ba0b554f66da870fe2bc1d319464eb592d6245cceb8d8cd715b5ff5892bd75
REDUCED: BO56rkD
Working: BO56rkD
Hash: 3b6648cac717f670ab1daee94a11de61430335774dfb6d15600698f4e5aedf70
REDUCED: FW17svC
Working: FW17svC
Hash: f39e06a9a8aa5941f298be681a8af33853b265a4d1334e1da3145221f543a852
REDUCED: DS38iqI
Working: DS38iqI
Hash: fc67db2a80155b69ad4d440634c4d4d47e4fd6e2da312cdbe363e176d452eaf3
REDUCED: NF69jjD

*/

	TableSubChnHeader *header, *dev_header;
	TableSubChnEntry  *entry,  *dev_entry;

	// output file
	char cand_file[81];
	FILE *cand;

	// start with this hash
	char *input_hash= "a376d88ccefa86544188491587823810b7675c2d60d934c2aee4b4cc577fdb2d";
	// looking for this hash
	char *sentinel  = "fc67db2a80155b69ad4d440634c4d4d47e4fd6e2da312cdbe363e176d452eaf3";

	uint32_t hash[8];
	uint32_t sent[8];
	int i,di;

	printf("searchtable_v3.\n");

	// get the target hash into uint32 format
	hash2uint32(input_hash, hash);
	hash2uint32(sentinel, sent);

	// calculate number of blocks to launch
	const int threads=512;							// threads per block
	const int blocks = (LINKS+threads-1)/threads;	// thread blocks

	// allocate space for data tables
	header = (TableSubChnHeader*)malloc(sizeof(TableSubChnHeader));
	entry  = (TableSubChnEntry*)malloc(sizeof(TableSubChnEntry)*LINKS);
	if((header==NULL)||(entry==NULL)) {
		printf("Error - searchtable_v3: Host memory allocation failed.\n");
		exit(1);
	}

	// set up the table
	header->hdr_size = sizeof(TableSubChnHeader);
	header->f1 = 0x00000000U;
	for(i=0;i<LINKS;i++) {
		(entry+i)->sublinks=i+10;
		for(di=0;di<8;di++) {
			(entry+i)->input_hash[di] = hash[di];
			(entry+i)->final_hash[di] = 0xffffffff;
		}
	}

	// allocate device memory 
	cudaMalloc((void**)&dev_header,sizeof(TableSubChnHeader));
	cudaMalloc((void**)&dev_entry,sizeof(TableSubChnEntry)*LINKS);
	if((dev_header==NULL)||(dev_entry==NULL)) {
		printf("Error - searchtable_v3: Device memory allocation failed.\n");
		exit(1);
	}
	// copy header to device
	// cudaMemcpy(dev_header, header, sizeof(TableSubChnHeader), cudaMemcpyHostToDevice);
	// Copy entries to device
	cudaMemcpy(dev_entry, entry, sizeof(TableSubChnEntry)*LINKS, cudaMemcpyHostToDevice);

	// launch kernel
	printf("Launching %d blocks of %d threads\n",blocks,threads);
	hash_calculate<<<blocks,threads>>>(dev_header,dev_entry);


	// copy header to device
	// cudaMemcpy(header, dev_header, sizeof(TableSubChnHeader), cudaMemcpyDevicetoHost);
	// copy entries to host
	cudaMemcpy(entry, dev_entry, sizeof(TableSubChnEntry)*LINKS, cudaMemcpyDeviceToHost);

	// check table
	
	printf("\nFinal Hashes\n");
	printf("Sentinel:\n%s\n",sentinel);
	for(i=0;i<LINKS;i++) {
		printf("@ thread_idx %d:\n", (entry + i)->sublinks);
		for(di=0;di<8;di++) printf("%08x ",(entry+i)->final_hash[di]);
		printf("\n");
		uint32_t *left=(uint32_t*)&((entry+i)->final_hash);
		uint32_t *right=(uint32_t*)sent;
		for(di=0;di<8;di++) {
			if (*(left+di)==*(right+di)) 
				continue; 
		else 
			break;
		}
		if(di==8)
			printf("Matching hash at thread_idx %d\n\n",(entry+i)->sublinks);
	}

	// save results
	fname_gen(cand_file,"cand64");
	cand=fopen(cand_file,"w");
	if(cand==NULL) {
		printf("Error - searchtable_v3: Unable to open  file.\n");
		exit(1);
	}
	fwrite(header,sizeof(TableSubChnHeader),1,cand);
	fwrite(entry,sizeof(TableSubChnEntry),LINKS,cand);

	// cleanup
	fclose(cand);
	cudaFree(dev_header);
	cudaFree(dev_entry);
	free(header);
	free(entry);

	return(0);
}


