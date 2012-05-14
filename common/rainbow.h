/*
	* Header file for Rainbow tables
	* Filename: rainbow.
	* edited sat 12 may
*/

#ifndef __RAINBOW_H__
	#define __RAINBOW_H__
	#include <stdint.h>
	
	// Define the current table-identification
	// Used by original version of tmerge
	#define TABLEIDENT 903975
	
	// thread blocks - Note Max value 28 to avoid kernel timeout
	#define DIMGRIDX 1
	// threads per block
	#define THREADS  1024
	// Split the total work into Work Units to avoid kernel timeout
	#define WORKUNITS 1
	// Number of links in chain - defining characteristic of table.
	#define LINKS 2048
	// limit password to MAXLENGTH chars
	#define MAXLENGTH 63
	// hash length
	#define HASHLENGTH 64
	// Maximum number of table names
	#define MAXTABS 64
	// number of table entries to calc
	#define T_ENTRIES (WORKUNITS*DIMGRIDX*THREADS)

	#define ROTR(a,b) (((a) >> (b)) | ((a) << (32-(b))))
	#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
	#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
	#define EP0(x) (ROTR(x,2) ^ ROTR(x,13) ^ ROTR(x,22))
	#define EP1(x) (ROTR(x,6) ^ ROTR(x,11) ^ ROTR(x,25))
	#define SIG0(x) (ROTR(x,7) ^ ROTR(x,18) ^ ((x) >> 3))
	#define SIG1(x) (ROTR(x,17) ^ ROTR(x,19) ^ ((x) >> 10))	
	#define SWAP32(x) ((uint32_t)((((uint32_t)(x) & 0xFF000000) >> 24) | \
								  (((uint32_t)(x) & 0x00FF0000) >>  8) | \
								  (((uint32_t)(x) & 0x0000FF00) <<  8) | \
								  (((uint32_t)(x) & 0x000000FF) << 24)))
	//-------------------------------------------------------------------
	typedef struct table_list {
		// fname_gen.c
		int Ntables;
		int idx;
		char table_name[MAXTABS][64];
	} TableList;
	//-------------------------------------------------------------------
	typedef struct dataframe {
		char trial_str[THREADS][MAXLENGTH+1];
		char start_str[THREADS][MAXLENGTH+1];
		uint32_t comp_hash[THREADS][8];
		uint32_t match[THREADS];
	} DataFrame;
	//-------------------------------------------------------------------
	typedef struct theader {
		unsigned int  hdr_size;	// header size in bytes (0x38 or 56 bytes)
		char check_sum[33];		// md5sum of data (table id)
		unsigned int date;		// creation date in epoch seconds (date +%s)
		unsigned int entries; 	// Number of table entries
		unsigned int links;		// Number of links in a chain
		unsigned int table_id;	// Table Index
		unsigned int f2;		// >>>> marker - not used
	} TableHeader;

	typedef struct tentry {
		// table entry parameters
		char initial_password[16];	// password
		uint32_t final_hash[8];		// 256 bit hash
		// subchain parameters
		uint32_t input_hash[8];
		unsigned int sublinks;
		// Padding - separates hash and next password 
		uint16_t pad;				
	} TableEntry;
	//------------------------------------------------------------------
	
#endif

