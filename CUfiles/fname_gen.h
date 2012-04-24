/*
 * Generate a file name of the following format
 * RbowTab_type_epochtimeinseconds.rbt
 * file_name_generate(char *buffer, char *type)
 * Version 1.0 16Jan2012
 */

#ifndef __FNAME_GEN_H__
#define __FNAME_GEN_H__
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "rainbow.h"

#ifdef __cplusplus
extern "C" 
{
#endif
void fname_gen(char*,char*,int);
int fname_read(char*);
int fname_write(char*);
int fname_list(TableList* tbl);
#ifdef __cplusplus
}
#endif

#endif
