/* tablid.c
* test code for getting/setting the rainbow table_id
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>

#include "../common/rainbow.h"
#include "table_utils.h"

uint32_t set_table_id(TableHeader *, char *tid);

uint32_t set_table_id(TableHeader *header, char *tid) {
	// Expects a pointer to a hex string
	// Returns equivalent uint32_t or zero on failure
	if(tid==NULL) {return(0);}
	header->table_id = strtol(tid,NULL,16);
	return(header->table_id);
}

int main(int argc, char** argv) {
	
	TableHeader* header;
	uint32_t result;
	
	printf("Count: %d	Table_id: %s\n",argc,argv[1]);
	
	if((header=(TableHeader*)malloc(sizeof(TableHeader)))==NULL) {
		perror("TableHeader malloc");
		exit(1);
	}
	
	result=set_table_id(header,argv[1]);
	if(result!=0) {
		printf("set_table_id success: 0x%08x\n",result);
		
	}else{
		printf("set_table_id failed: 0x%08x \n",result);
	}
	
	free(header);

	return(0);
}
