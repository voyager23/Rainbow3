// tmerge.h

#ifndef __TMERGE_H__
#define __TMERGE_H__

#include "../common/rainbow.h"

// ------Declarations-----
int hash_compare_uint32_t(uint32_t *left, uint32_t *right);
int tmerge(char *sort);
int tmerge_2(const char *id_str);
char *sort2new(char *buffer);

#endif

