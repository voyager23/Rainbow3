/*
 * Generate a file name of the following format
 * type_table-ident.rbt
 * 
 * Version 2.0 01May2012
 */

#ifndef __FNAME_GEN_H__
#define __FNAME_GEN_H__
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "../common/rainbow.h"

__host__ void fname_gen(char*,char*,uint32_t);	//output format changed
__host__ int fname_read(char*);
__host__ int fname_write(char*);
__host__ int fname_list(TableList* tbl);
#endif
