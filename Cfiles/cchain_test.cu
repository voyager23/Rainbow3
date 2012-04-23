/*	
	*
	* cchain_test.cu
	* testbed code to investigate compute_chain
	*
*/

#ifndef __CUDA__
	#define __CUDA__
#endif

//===========================Include code======================================

#include "rainbow.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <endian.h>
#include <time.h>

//=========================Declarations=================================
#ifdef __CUDA__
__global__
#endif
void kernel(TableHeader *header, TableEntry *entry);

// Hash constants
#ifdef __CUDA__
__constant__
#endif
uint32_t k[64] = {
	   0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	   0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	   0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	   0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	   0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	   0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	   0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	   0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
	   
//=========================Include Device Code==========================

#include "freduce.cu"
#include "initHash.cu"
#include "sha256_txfm.cu"
#include "utils.cu"

#define NODEBUG
void hash2uint32(char *hash_str, uint32_t *H) {
	// hash_str must be 64 byte hexadecimal string
	const int words=8;
	char buffer[9], *source=hash_str;
	int i,len;
	len = strlen(hash_str);
	if(len != sizeof(unsigned)*words*2) {
		printf("Error - hash2uint32: hash_str length=%d\n",len);
		exit(1);
	}	
	for(i=0;i<words;i++) {
		strncpy(buffer,source,8);
		buffer[8]='\0';
		sscanf(buffer,"%x",H+i);
		source+=8;
	}
#ifdef DEBUG
	printf("hash2uint32\nHash:%s\nH[8]:",hash_str);
	for(i=0;i<8;i++) printf("%08x",H[i]); 
	printf("\n");
	#undef DEBUG
#endif
}



//=================================Main Code==================================

int main(int argc, char **argv) {

	TableEntry *check;
	
	//void compute_chain(TableEntry *entry, int links)
	check = (TableEntry*)malloc(sizeof(TableEntry)*(LINKS));
	strcpy(check->initial_password, "MM49jhM");
	compute_chain(check,5);
	show_table_entries(check,0,5);
	free(check);
	return(0);
}


