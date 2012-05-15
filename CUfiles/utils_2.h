#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#ifndef __UTILS_2_H__
#define __UTILS_2_H__
#include "../common/rainbow.h"
//---------------------Declarations---------------------------------------------
__host__
void hash2uint32(char *hash_str, uint32_t *H);
__host__
int get_rnd_table_entry(TableEntry *target, FILE *fp);
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
void compute_chain(TableHeader *header,TableEntry *entry, int links);
__host__ 
void fname_gen(char*,char*,uint32_t);	//output format changed

// Inline code
__device__ __host__
void sha256_transform(uint32_t *w, uint32_t *H);
__device__ __host__
void initHash(uint32_t *h);
__device__ __host__
void reduce_hash(uint32_t H[], uint8_t B[], uint32_t link_idx);

//------------------------------------------------------------------------------
#endif
