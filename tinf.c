#include <stdlib.h>
#include <stdio.h>
#include "rainbow.h"


// read a table name from the CL
// display header info

void show_table_header(TableHeader *header) {
	printf("Header size:	%d\n", header->hdr_size);
	printf("Checksum:	%s\n", header->check_sum);
	printf("Created:	%d\n", header->date);
	printf("Entries:	%d\n", header->entries);
	printf("Links:		%d\n", header->links);
	printf("Index:		%d\n", header->f1);
}

int main(int argc, char **argv) {
	FILE *fp;
	TableHeader *header;
	char table[128];
	
	fp = fopen(argv[1],"r");
	if(fp==NULL) {
		printf("Failed to open %s\n",argv[1]);
		exit(1);
	}
	header = (TableHeader*)malloc(sizeof(TableHeader));
	if(header==NULL) {
		printf("Unable to allocate memory for table header.\n");
		exit(1);
	}
	fread(header,sizeof(TableHeader),1,fp);
	fclose(fp);
	
	show_table_header(header);

	return(0);
}
