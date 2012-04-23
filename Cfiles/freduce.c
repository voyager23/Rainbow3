/*
	* Include code to modularise the reduction functions
	* freduce.cu
	* reduction functions for use by maketable and searchtable
	*
*/
#include <stdint.h>
#include "freduce.h"

#ifdef __CUDA__ 
__device__ 
#endif
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
//=============================================================================
