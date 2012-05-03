//-----Declarations-----
__host__
void hash2uint32(char *hash_str, uint32_t *H);
__host__
int get_rnd_table_entry(TableEntry *target, FILE * fp);
__host__
void make_rnd_target(TableEntry *target);
__host__
void show_table_header(TableHeader *header);
__host__
void show_table_entries(TableEntry *entry,int first,int last);
__host__
int hash_compare_32bit(void const *p1, void const *p2);
__host__
int hash_compare_uint32_t(uint32_t *left, uint32_t *right);
__host__
void compute_chain(TableEntry *entry, int links);

//-----Definitions-----
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
	
	// randomise
	srand(time(NULL));
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
	
	show_table_entries(chain,0,2);
	
	compute_chain(chain,link);
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

__host__
void make_rnd_target(TableEntry *target) {
	// generate a random password.
	// calculate the associated hash and store in 'target'
	srand(time(NULL));
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
	compute_chain(target,1);
}

__host__
void show_table_header(TableHeader *header) {
	printf("Header size:	%d\n", header->hdr_size);
	printf("Checksum:	%s\n", header->check_sum);
	printf("Created:	%d\n", header->date);
	printf("Entries:	%d\n", header->entries);
	printf("Links:		%d\n", header->links);
	printf("Index:		%d\n", header->f1);
	printf("Solutions:	%u\n", header->entries*header->links);
}

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

__host__
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
		for(i=0;i<8;i++) {
			(entry+link_idx)->final_hash[i] = H[i];
		}

		// Reduce the Hash and store in B using reduce_hash function		
		(void)reduce_hash(H,B,link_idx,TABLEIDENT);

	} // for link_idx=0 ...

} // end compute_chain()
