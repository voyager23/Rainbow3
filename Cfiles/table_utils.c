/*	
	* Rainbow table utilities
	* filename: table_utils.cu
	* compile using:	nvcc --linker-options -lm maketable.c md5.c 
	* 24Jan2012 - adding new code for comparing hashes
	* hash_compare_uint32_t() takes pointers directly to a hash
*/

#include "md5.h"
#include "../common/rainbow.h"
#include "table_utils.h"

// Hash constants
const uint32_t k[64] = {
	   0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	   0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	   0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	   0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	   0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	   0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	   0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	   0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };

//==============================================================================
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
//=============================================================================
void sha256_transform(uint32_t *w, uint32_t *H) {
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
//=============================================================================
void show_table_header(TableHeader *header) {
	unsigned solutions;
	printf("Header size:	%d\n", header->hdr_size);
	printf("Checksum:	%s\n", header->check_sum);
	printf("Created:	%d\n", header->date);
	printf("Entries:	%d\n", header->entries);
	printf("Links:		%d\n", header->links);
	printf("Index:		%d\n", header->table_id);
	printf("Solutions:	%u\n", header->entries*header->links);
}
//=============================================================================
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
//=============================================================================
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
//===============New Code=======================================================
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
//==============================================================================
void compute_chain(TableEntry *entry, int links) {
	// Calculate and store -in detail- a full chain.
	// Note - function assumes sufficient memory has been allocated for the results.
	
	uint8_t  M[64];	// Initial string zero padded and length in bits appended
	uint32_t W[64];	// Expanded Key Schedule
	uint32_t H[8];	// Hash
	int i;			// working index
	uint64_t l = 0; // length of message
	uint8_t  B[64];	// store initial and working trials here to protect original data
	uint link_idx;	// index

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
		for(i=0;i<8;i++) (entry+link_idx)->final_hash[i] = H[i];

		// Reduce the Hash and store in B using reduce_hash function		
		(void)reduce_hash(H,B,link_idx);

	} // for link_idx=0 ...

} // end compute_chain()
//======================================================================
#if(0)
int tmerge(char *input,char *new_merge){
	
	TableHeader *hdr_left, *hdr_right, *hdr_merge;
	TableEntry  *ent_left, *ent_right, *ent_merge;
	TableEntry  *l_ptr, *l_end, *r_ptr, *r_end, *merge_ptr, *merge_end;
	const char *merge = "./rbt/RbowTab_merge.rbt";
	FILE *fp_left, *fp_right, *fp_merge;
	unsigned entries_total, entries_merged, entries_count, discarded;
	int i,di,compare,r1,r2,error_flag;
	
	// check that the original merged file is valid for reading (fp_left)
	fp_left = fopen(merge,"r");
	if(fp_left==NULL) {
		printf("Tmerge - invalid file %s\n",merge);
		return(1);
	}
	// check that the newly sorted file is valid for reading (fp_right)
	fp_right = fopen(input,"r");
	if(fp_right==NULL) {
		printf("Tmerge - invalid file %s\n",input);
		fclose(fp_left);
		return(1);
	}	
	
	hdr_left = (TableHeader*)malloc(sizeof(TableHeader));
	fread(hdr_left,sizeof(TableHeader),1,fp_left);	
	
	hdr_right = (TableHeader*)malloc(sizeof(TableHeader));
	fread(hdr_right,sizeof(TableHeader),1,fp_right);

	// allocate for merged table
	entries_total = hdr_left->entries + hdr_right->entries;
	hdr_merge = (TableHeader*)malloc(sizeof(TableHeader));
	ent_merge = (TableEntry*)malloc(sizeof(TableEntry)*entries_total);

	// allocate for source tables
	ent_left  = (TableEntry*)malloc(sizeof(TableEntry)*hdr_left->entries);
	ent_right = (TableEntry*)malloc(sizeof(TableEntry)*hdr_right->entries);

	// read source entries
	// original merged table
	fread(ent_left,sizeof(TableEntry),hdr_left->entries,fp_left);
	fclose(fp_left);
	// new sorted table
	// Read this table via a buffer, discarding duplicates. Use r_end 
	// as working pointer and entries_count.
	r_end = ent_right;
	discarded=0;
	for(entries_count=0; entries_count<hdr_right->entries; entries_count++) {
		fread(r_end,sizeof(TableEntry),1,fp_right);
		if((r_end==ent_right)||(hash_compare_uint32_t(r_end->final_hash,(r_end-1)->final_hash)!=0)){
			r_end++;
		} else {
			discarded++;
		}	
	}	
	fclose(fp_right);
	printf("Discarded %u from sort file\n",discarded);

	// set sentinels.
	l_end = ent_left + hdr_left->entries;
	//r_end = ent_right + hdr_right->entries;
	merge_end = ent_merge + entries_total;

	// set working pointers and counter
	entries_merged = 0;
	discarded = 0;
	l_ptr = ent_left;
	r_ptr = ent_right;
	merge_ptr = ent_merge;

	// Main loop
	// 19feb2012 - Candidate file entries may have duplicated final_hash
	while ((l_ptr < l_end)&&(r_ptr < r_end)) {
		compare = hash_compare_uint32_t(l_ptr->final_hash, r_ptr->final_hash);
		if (compare == 1) {
			// left > right
			*merge_ptr = *r_ptr;
			merge_ptr++;
			r_ptr++;
			entries_merged++;			
		}
		else if (compare == -1) {
			// left < right
			*merge_ptr = *l_ptr;
			merge_ptr++;
			l_ptr++;
			entries_merged++;
		}
		else {
			// left == right - Discard right entry
			// printf("Discarding entry from right table\n");
			*merge_ptr = *l_ptr;
			merge_ptr++;			
			l_ptr++;			
			r_ptr++;
			entries_merged++;
			discarded++;
		}		
		// now test for end conditions
		if(r_ptr == r_end) {
				// write balance of left to merge
				entries_count = (l_end - l_ptr);
				memcpy(merge_ptr, l_ptr, sizeof(TableHeader)*entries_count);
				// adjust count and pointers
				l_ptr = l_end;
				merge_ptr += entries_count;
				entries_merged += entries_count;
		}
		if(l_ptr == l_end) {
				// write balance of right to merge
				entries_count = (r_end - r_ptr);
				memcpy(merge_ptr, r_ptr, sizeof(TableHeader)*entries_count);
				// adjust count and pointers
				r_ptr = r_end;
				merge_ptr += entries_count;
				entries_merged += entries_count;
		}
	} // end_main_loop
	
	// write the merged table to disk
	hdr_merge->hdr_size = sizeof(TableHeader);
	hdr_merge->entries = entries_merged;
	hdr_merge->links = hdr_left->links;
	
	// TODO Recalc the md5sum of the entries.
	strcpy(hdr_merge->check_sum,"No data");
	//==========NEW=CODE====================
	// Calculate the md5sum of the table entries
	md5_state_t state;
	md5_byte_t digest[16];	
	md5_init(&state);
	for(i=0; i<entries_merged; i++)
		md5_append(&state, (const md5_byte_t *)&(ent_merge[i]), sizeof(TableEntry));
	md5_finish(&state, digest);

	// Save the md5sum in check_sum slot
	for (di = 0; di < 16; ++di)
	    sprintf(hdr_merge->check_sum + di * 2, "%02x", digest[di]);
	*(hdr_merge->check_sum + di * 2) = '\0';
	//======================================
	
	//sanity check for destination file
	fp_merge = fopen(new_merge,"w");
	if(fp_merge==NULL) {
		printf("Unable to open %s \n",new_merge);
	} else {
		r1=fwrite(hdr_merge,sizeof(TableHeader),1,fp_merge);
		r2=fwrite(ent_merge,sizeof(TableEntry),entries_merged,fp_merge);
		fclose(fp_merge);
		if((r1==1)&&(r2==entries_merged)) {
			printf("Removing the original sorted table file.\n");
			remove(input);	
			
			// remove previous _merge.rbt.sav
			remove("./rbt/RbowTab_merge.rbt.sav");
			
			// rename _merge.rbt to _merge.rbt.sav
			rename("./rbt/RbowTab_merge.rbt", "./rbt/RbowTab_merge.rbt.sav");
			
			// rename _merge.rbt.new to _merge.rbt
			printf("Renaming new merge file to _merge.rbt\n");
			rename(new_merge, "./rbt/RbowTab_merge.rbt");
			
			error_flag=0;
		} else {
			printf("Tmerge - invalid file %s Nothing saved\n",new_merge);
			error_flag=1;
		}
	}
	
	// Clean up pointers
	free(ent_right);
	free(ent_left);
	free(ent_merge);
	free(hdr_merge);
	free(hdr_right);
	free(hdr_left);	
	// final actions		
	printf("Merged %d entries - adding %d.\n",entries_merged,hdr_right->entries-discarded);
	return(error_flag);
}
#endif
int get_rnd_table_entry(TableEntry *target, FILE * fp) {
	// target is pointer to a single TableEntry struct.
	// fp is pointer to a stored table file.
	// select a chain and link index at random
	// calculate and save the entry in known
	// returns number of entries found (0 on failure);
	
	unsigned chain_idx,link,link_idx,dx;
	TableHeader *header;
	TableEntry  *chain;
	
	// randomise
	srand(time(NULL));
	// allocate and read header
	header = (TableHeader*)malloc(sizeof(TableHeader));
	fread(header,sizeof(TableHeader),1,fp);
	// allocate space for a full chain
	chain = (TableEntry*)malloc(sizeof(TableEntry)*LINKS);
	// set initial_password of chain to randomly selected
	// value from main table.
	for(chain_idx=0; chain_idx<=(rand() % header->entries); chain_idx++) 
		fread(chain,sizeof(TableEntry),1,fp);
	// randomly select a link and calculate pass/hash pair
	link = (rand() % LINKS)+1;
	compute_chain(chain,link);
	// display result
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

void reduce_hash(uint32_t H[], uint8_t B[], int link_idx) {

		uint32_t z;
		uint16_t b0,b1;
		const uint16_t mask = 0xffff;
		
		z = H[0] + link_idx;
		b0 = (uint16_t)(z & mask);
		B[0] = (b0 % 26) + 'A';
		z >>= 16;
		b1 = (uint16_t)(z & mask);
		B[1] = (b1 % 26) + 'A';

		z = H[1] + link_idx;
		b0 = (uint16_t)(z & mask);
		B[2] = (b0 % 10) + '0';
		z >>= 16;
		b1 = (uint16_t)(z & mask);
		B[3] = (b1 % 10) + '0';

		z = H[2] + link_idx;
		b0 = (uint16_t)(z & mask);
		B[4] = (b0 % 26) + 'a';
		z >>= 16;
		b1 = (uint16_t)(z & mask);
		B[5] = (b1 % 26) + 'a';
	
		z = H[3] + link_idx;
		b0 = (uint16_t)(z & mask);
		B[6] = (b0 % 26) + 'A';
		B[7] = '\0';
}
