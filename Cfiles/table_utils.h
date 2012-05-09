/*	
	* Rainbow table utilities
	* filename: table_utils.cu
	* compile using:	nvcc --linker-options -lm maketable.c md5.c 
*/

#ifndef __TABLE_UTILS_H__
#define __TABLE_UTILS_H__
	#include "md5.h"
	#include "../common/rainbow.h"
	#include <math.h>
	#include <stdint.h>
	#include <stdio.h>
	#include <stdlib.h>
	#include <string.h>
	#include <endian.h>
	#include <time.h>
	
#ifdef __cplusplus
extern "C" 
{
#endif
	//=============================================================================
	void show_table_header(TableHeader*);
	void show_table_entries(TableEntry*,int,int);
	int hash_compare_32bit(void const *, void const *);
	void hash2uint32(char *, uint32_t *);
	void compute_chain(TableEntry *entry, int links);
	int hash_compare_uint32_t(uint32_t *left, uint32_t *right);
	int tmerge(char *sort,char *new_merge);
	int get_rnd_table_entry(TableEntry*,FILE*);
	void reduce_hash(uint32_t H[], uint8_t B[], int link_idx);
	//=============================================================================
#ifdef __cplusplus
}  /* end extern "C" */
#endif

#endif
