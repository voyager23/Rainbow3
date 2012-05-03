/*
	* Utility function code.
	* filename: single_chain.c
	* Based on cuda version.
	* Given an initial password and a chain length (link_count)
	* calculate and save a complete chain for reference
	* Perfomance: Approx. 495000 links/sec
*/

#ifndef __NO_CUDA__
	#define __NO_CUDA__
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <endian.h>

#include "rainbow.h"
//#include "table_utils.h"
#include "fname_gen.h"
#include "freduce.h"
//----------------------------------------------------------------------
void compute_chain_dev(TableEntry *entry, int links);
//----------------------------------------------------------------------
void initHash(uint32_t *h) {
	h[0] = 0x6a09e667;
	h[1] = 0xbb67ae85;
	h[2] = 0x3c6ef372;
	h[3] = 0xa54ff53a;
	h[4] = 0x510e527f;
	h[5] = 0x9b05688c;
	h[6] = 0x1f83d9ab;
	h[7] = 0x5be0cd19;
}
//-----------------------------------------------------------------------------
void sha256_transform(uint32_t *w, uint32_t *H) {
	//working variables 32 bit words
	int i;
	uint32_t a,b,c,d,e,f,g,h,T1,T2;
	
	uint32_t k[64] = {
	   0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	   0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	   0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	   0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	   0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	   0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	   0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	   0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

	a = H[0];
	b = H[1];
	c = H[2];
	d = H[3];
	e = H[4];
	f = H[5];
	g = H[6];
	h = H[7];
	
	//DEBUG
	//printf("single_chain:\n");
	//for(i=0;i<64;i++) printf("k=%08x w=%08x\n", k[i],w[i]);
	//END DEBUG
   
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
      
      //DEBUG
	//printf("single_chain:\n");
	//printf("T1: %08x\n",T1);
	//printf("T2: %08x\n",T2);
	//END DEBUG
	
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

void compute_chain_dev(TableEntry *entry, int links)
{
	// Note - function assumes sufficient memory has been allocated for the results.

	uint8_t  M[64];	// Initial string zero padded and length in bits appended
	uint32_t W[64];	// Expanded Key Schedule
	uint32_t H[8];	// Hash
	int i;			// working index
	uint64_t l = 0; // length of message
	uint8_t  B[64];	// store initial and working trials here to protect original data
	uint chain_idx;	// index

	// copy zero terminated string into B buffer
	i=0;
	while(entry->initial_password[i] != 0x00) {
		B[i] = entry->initial_password[i];
		i++;
	}
	B[i] = 0x00;

	// ---> main loop buffer B contains the zero term string
	for(chain_idx=0; chain_idx<links; chain_idx++) {
		// copy zero terminated string from B to M and note length
		// use this loop to save the new (reduced) password in the Table
		i=0; l=0;
		while(B[i] != 0x00) {
			M[i] = B[i];
			(entry+chain_idx)->initial_password[i] = B[i];
			i++;
			l++;
		}
		M[i] = 0x80;
		(entry+chain_idx)->initial_password[i] = 0x00;
		// inc index and zero fill
		i++;
		while(i < 56) M[i++]=0x00;

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
		uint32_t *q = (uint32_t*)&M[i];
		*q = *(p+1);
		*(q+1) = *p;

		// The 64 bytes in the message block can now be used
		// to initialise the 64 4-byte words in the message schedule W[64]
		uint8_t *r = (uint8_t*)M;
		uint8_t *s = (uint8_t*)W;
		for(i=0;i<64;i++) s[i] = r[i];
		for(i=16;i<64;i++) W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];
	
		// set initial hash values
		initHash(H);
		
		
			
		// Now calc the hash
		sha256_transform(W,H);
		
		// save hash in Table
		for(i=0;i<8;i++) {
			(entry+chain_idx)->final_hash[i] = H[i];
		}

		// Reduce the Hash and store in B using reduce_hash function		
		reduce_hash(H,B,chain_idx,TABLEIDENT);

	} // for chain_idx=0 ...

} // end compute_chain()

//-------------------------------------------------------------------

int main(int argc, char **argv) {
/*	
	workflow:
	1) Input initial password from CL (scanf) to buffer
	2) Allocate memory space for TableHeader
	3) Allocate memory space for TableEntry*links
	4) Initialise Table
	5) Call compute_chain(TableEntry*,link_count)
	6) Write results to file RbowTab_single.rbt
	7) Display table on StdOut
*/
	char buffer[64] = "--DO_NOT_USE--";
	TableHeader *header;
	TableEntry  *entry;
	int i;
	FILE *fp;
	int link_count;

	printf("=====Single Chain Generator.=====\n");
	if(argc != 3) {
		printf("Usage: AB12cdE links\n");
		exit(1);
	}
	strncpy(buffer,argv[1],7);
	buffer[7]='\0';
	link_count = atoi(argv[2]);

	// printf("Enter the initial 7 character password and number of links: ");
	// scanf("%s %d",buffer,&link_count);

	printf("The initial password is %s. The number of links is %d.\n",buffer,link_count);

	header = (TableHeader*)malloc(sizeof(TableHeader));
	entry  = (TableEntry*)malloc(sizeof(TableEntry)*link_count);
	if( (header==NULL) || (entry==NULL) ) {
		printf("Error - Memory allocate has failed.\n");
		exit(1);
	}

	strcpy(&(entry->initial_password[0]),buffer);
	entry->final_hash[0] = 0xffffffff;
	for(i=1; i<link_count; i++) {
		strcpy(&((entry+i)->initial_password[0]),"empty");
		(entry+i)->final_hash[0] = 0xffffffff;
	}

	// call compute - function assumes sufficient memory has been allocated for the results
	compute_chain_dev(entry,link_count);

	// write results
	fname_gen(buffer,"single",T_ENTRIES);
	fp = fopen(buffer,"w");
	if(fp != NULL) {
		fwrite(header,sizeof(TableHeader),1,fp);
		fwrite(entry,sizeof(TableEntry),link_count,fp);
		fclose(fp);
	} else {
		printf("Failed to write results to %s\n", buffer);
	}

#if 1		
	// Display table
	int di;
	for(i=0; i<link_count; i++) {
		printf("Link %d: Password: %s \nHash %d: ", i, (entry+i)->initial_password,i);
		for(di=0;di<8;di++) printf("%08x ", (entry+i)->final_hash[di]);
		printf("\n");
	}
#endif
	
	// Cleanup
	free(entry);
	free(header);
	return(0);
}


