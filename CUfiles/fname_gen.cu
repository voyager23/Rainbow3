/*
 * Generate a file name of the following format
 * RbowTab_type_dimension.rbt
 * file_name_generate(char *buffer, char *type)
 * Version 1.0 16Jan2012
 */

#include "fname_gen.h"
#include "../common/rainbow.h"

// This file is a list of 'zero-term' filenames

char *rbowtab_tables = "./rbt/RbowTab_tables_0.rbt";

__host__
void fname_gen(char *buffer, char *type_str, uint32_t tid) {
	// now takes table_id as parameter
	// output form: ./rbt/merge_
	const char *root = "./rbt/";
	const char *rbt  = "rbt";
	char table_id[64];
	
	sprintf(table_id,"0x%08x",tid);
	sprintf(buffer,"%s%s_%s.%s",root,type_str,table_id,rbt);
	
	printf("\nfname_gen/table_id: %s\n",table_id);
}

__host__
int fname_read(char *buffer) {
	// read table names from RbowTab_tables_0.rbt
	FILE *fp;
	int items=0;
	
	fp = fopen(rbowtab_tables,"r");
	if(fp==NULL) return 1;
	do {
		items = fscanf(fp,"%s",buffer);
		printf("Found: %s\n",buffer);
	} while (items != EOF);
	fclose(fp);	
	return 0;
}

__host__
int fname_write(char *buffer) {
	FILE *fp;

	fp = fopen(rbowtab_tables,"a");
	if(fp==NULL) return 1;
	// write the filename
	if(fwrite(buffer,strlen(buffer),1,fp)==0) return 1;
	// add a newline char
	if(fwrite("\n",sizeof(char),1,fp)==0) return 1;
	fclose(fp);	
	return 0;
}

__host__
int fname_list(TableList* tbl) {
	// --TODO--
	// open the list of table names
	// read names into TableList
	// update values in Ntables and idx
	FILE *fp;
	int result;
	char buffer[128];
	
	fp=fopen(rbowtab_tables,"r");
	if(fp==NULL) {
		printf("Unable to open %s\n",rbowtab_tables);
		return 1;
	}
	
	tbl->idx = 0;
	tbl->Ntables = 0;
	do {
		printf("Getting filename\n");
		result = fscanf(fp,"%s",buffer);
		if(result != EOF) {
			strcpy(tbl->table_name[tbl->Ntables],buffer);
			printf("Found: %s\n",tbl->table_name[tbl->Ntables]);
			tbl->Ntables++;
		}
	} while (result != EOF);
	
	
		
	printf("%d files found.\n",tbl->Ntables);	
	fclose(fp);
	return 0;
}

	  
