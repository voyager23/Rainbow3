/*
	* Include code to modularise the reduction functions
	* freduce.h
	* reduction functions for use by maketable and searchtable
	*
*/
#ifndef __FREDUCE_H__
#define __FREDUCE_H__
#include <stdint.h>

#ifdef __cplusplus
extern "C"
#endif
//======================================================================
#ifdef __CUDA__ 
__device__ 
#endif
void reduce_hash(uint32_t H[], uint8_t B[], int link_idx, uint32_t tab_id);
//======================================================================
#ifdef __cplusplus
}
#endif
#endif
