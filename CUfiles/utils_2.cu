/*
 * Version 14May2012
 * 
 * This is a consolidated file
 * holding functions:
 * utils.cu
 * initHash.cu
 * sha256_txfm.cu
 * freduce.cu
 * fname_gen.cu
 * 
 * This file must #included to avoid problems with dependencies
*/

#include "utils_2.h"

//================================Definitions===================================
__device__ __host__
void reduce_hash(uint32_t H[], uint8_t B[], uint32_t link_idx) {

		uint32_t z;
		uint16_t b0,b1;
		const uint16_t mask = 0xffff;
		
		uint32_t offset = link_idx;
		
		z = H[0] + offset;
		b0 = (uint16_t)(z & mask);
		B[0] = (b0 % 26) + 'A';
		z >>= 16;
		b1 = (uint16_t)(z & mask);
		B[1] = (b1 % 26) + 'A';

		z = H[1] + offset;
		b0 = (uint16_t)(z & mask);
		B[2] = (b0 % 10) + '0';
		z >>= 16;
		b1 = (uint16_t)(z & mask);
		B[3] = (b1 % 10) + '0';

		z = H[2] + offset;
		b0 = (uint16_t)(z & mask);
		B[4] = (b0 % 26) + 'a';
		z >>= 16;
		b1 = (uint16_t)(z & mask);
		B[5] = (b1 % 26) + 'a';
	
		z = H[3] + offset;
		b0 = (uint16_t)(z & mask);
		B[6] = (b0 % 26) + 'A';
		B[7] = '\0';
}
//==============================================================================
__device__ __host__
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
//==============================================================================
__device__ __host__
void sha256_transform(uint32_t *w, uint32_t *H) {
	//working variables 32 bit words
	int i;
	uint32_t a,b,c,d,e,f,g,h,T1,T2;
	// 22Apr2012 k[] array added to function
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
//==============================================================================
__host__
void fname_gen(char *buffer, char *type_str, uint32_t tid) {
	// now takes table_id as parameter
	// output form: ./rbt/merge_
	const char *root = "./rbt/";
	const char *rbt  = "rbt";
	char table_id[64];
	
	sprintf(table_id,"0x%08x",tid);
	sprintf(buffer,"%s%s_%s.%s",root,type_str,table_id,rbt);
	
	//printf("\nfname_gen/table_id: %s\n",table_id);
}


//==============================================================================
__host__
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
}
//==============================================================================
__host__
int get_rnd_table_entry(TableEntry *target, FILE * fp) {
	// target is pointer to a single TableEntry struct.
	// fp is pointer to a stored table file.
	// select a chain and link index at random
	// calculate and save the entry in known
	// returns number of entries found (0 on failure);
	
	unsigned chain_idx,link,dx;
	TableHeader *header;
	TableEntry  *chain;

	// allocate and read header
	header = (TableHeader*)malloc(sizeof(TableHeader));
	fread(header,sizeof(TableHeader),1,fp);
	
	show_table_header(header);
	
	// allocate space for a full chain
	chain = (TableEntry*)malloc(sizeof(TableEntry)*LINKS);
	// set initial_password of chain to randomly selected
	// value from main table.
	for(chain_idx=0; chain_idx<=(rand() % header->entries); chain_idx++) 
		fread(chain,sizeof(TableEntry),1,fp);
	// randomly select a link and calculate pass/hash pair
	link = (rand() % LINKS)+1;	
	compute_chain(header,chain,link);
	// display result
	printf("From chain commencing %s and at link %d:\n", chain->initial_password, link-1);
	printf("\nRandomly selected chain data.\nPassword: %s\nHash: ", (chain+link-1)->initial_password);
	for(dx=0;dx<8;dx++) printf("%08x ", (chain+link-1)->final_hash[dx]);
	printf("\n");
	// move that data into target
	strcpy(target->initial_password,(chain+link-1)->initial_password);
	for(dx=0;dx<8;dx++) {
		//printf("%08x ", (chain+link-1)->final_hash[dx]);
		target->final_hash[dx]=(chain+link-1)->final_hash[dx];
	};	
	//------------------------
	free(chain);
	free(header);
	return(1);
}
//==============================================================================
__host__
void make_rnd_target(TableEntry *target) {
	// generate a random password.
	// calculate the associated hash and store in 'target'
	// Random password type 'UUnnllU'
	target->initial_password[0]= (rand() % 26) + 'A';
	target->initial_password[1]= (rand() % 26) + 'A';
	target->initial_password[2]= (rand() % 10) + '0';
	target->initial_password[3]= (rand() % 10) + '0';
	target->initial_password[4]= (rand() % 26) + 'a';
	target->initial_password[5]= (rand() % 26) + 'a';
	target->initial_password[6]= (rand() % 26) + 'A';
	target->initial_password[7]= '\0';
	// DEBUG
	target->final_hash[0] = 0x776f6272;
	compute_chain(NULL,target,1);
}
//==============================================================================
__host__
void show_table_header(TableHeader *header) {
	printf("Header size:	%d\n", header->hdr_size);
	printf("Checksum:	%s\n", header->check_sum);
	printf("Created:	%d\n", header->date);
	printf("Entries:	%d\n", header->entries);
	printf("Links:		%d\n", header->links);
	printf("Index:		%u\n", header->table_id);
	printf("Solutions:	%u\n", header->entries*header->links);
}
//==============================================================================
__host__
void show_table_entries(TableEntry *entry,int first,int last) {
	// display entries from table from first to last inclusive
	int idx,i;
	for(idx=first; idx <=last; idx++) {
		printf("Initial password[%d]: %s\nFinal Hash: ",idx,(entry+idx)->initial_password);
		for(i=0;i<8;i++) printf("%08x ",(entry+idx)->final_hash[i]);
		printf("\nInput hash: ");
		for(i=0;i<8;i++) printf("%08x ",(entry+idx)->input_hash[i]);
		printf("\nSublinks: %d\n\n", (entry+idx)->sublinks);
	}
}
//==============================================================================
__host__
int hash_compare_32bit(void const *p1, void const *p2) {
	// used by bsearch function
	int i;
	uint32_t *left=(uint32_t*)&(((TableEntry*)p1)->final_hash);
	uint32_t *right=(uint32_t*)&(((TableEntry*)p2)->final_hash);

	for(i=0; i<8; i++) {
		if(*(left+i) > *(right+i)) 
			return(1);
		else if(*(left+i) < *(right+i)) 
			return(-1);
		else
			continue;
	}
	return(0);
}
//==============================================================================
__host__
int hash_compare_uint32_t(uint32_t *left, uint32_t *right) {
	int i;
	for(i=0; i<8; i++) {
		if(*(left+i) > *(right+i)) 
			return(1);
		else if(*(left+i) < *(right+i)) 
			return(-1);
		else
			continue;
	}
	return(0);
}
//==============================================================================
__host__
void compute_chain(TableHeader *header, TableEntry *entry, int links) {
	// Calculate and store -in detail- a full chain.
	// Note - function assumes sufficient memory has been allocated for the results.
	
	uint8_t  M[64];	// Initial string zero padded and length in bits appended
	uint32_t W[64];	// Expanded Key Schedule
	uint32_t H[8];	// Hash
	int i;			// working index
	uint64_t l = 0; // length of message
	uint8_t  B[64];	// store initial and working trials here to protect original data
	uint32_t link_idx;	// index

	// copy zero terminated string into B buffer
	i=0;
	while(entry->initial_password[i] != 0x00) {
		B[i] = entry->initial_password[i];
		i++;
	}
	B[i] = 0x00;

	// ---> main loop buffer B contains the zero term string
	for(link_idx=0; link_idx<links; link_idx++) {
		// copy zero terminated string from B to M and note length
		// use this loop to save the new (reduced) password in the Table
		i=0; l=0;
		while(B[i] != 0x00) {
			M[i] = B[i];
			(entry+link_idx)->initial_password[i] = B[i];
			i++;
			l++;
		}
		M[i] = 0x80;
		(entry+link_idx)->initial_password[i] = 0x00;
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
			(entry+link_idx)->final_hash[i] = H[i];
		}

		// Reduce the Hash and store in B using reduce_hash function
		if(header!=NULL)		
			(void)reduce_hash(H,B,(link_idx+header->table_id));

	} // for link_idx=0 ...

} // end compute_chain()
//==============================================================================
