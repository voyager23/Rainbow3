/*
 * rbt_ident.c
 * Read the header from a RbowTab merge file
 * and determine the table ident.
 * 
 * Copyright 2012 michael <mike@cuda>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include "rainbow.h"

void show_table_header(TableHeader *header);

void show_table_header(TableHeader *header) {
	printf("Header size:	%d\n", header->hdr_size);
	printf("Checksum:	%s\n", header->check_sum);
	printf("Created:	%d\n", header->date);
	printf("Entries:	%d\n", header->entries);
	printf("Links:		%d\n", header->links);
	printf("Ident:		%d\n", header->f1);
	printf("Solutions:	%u\n", header->entries*header->links);
}

int main(int argc, char **argv)
{
	FILE *fp;
	TableHeader *header;
	if(argc==2) {
		printf("Path to RbowTab_merge: %s\n",argv[1]);	
	} else {
		printf("Error: No table name.\n");
		printf("Usage: rbt_ident path/to/merge/table\n");
		exit(1);
	}
	fp = fopen(argv[1],"r");
	if(fp==NULL) {
		printf("Error: unable to open %s for reading.\n",argv[1]);
		exit(1);
	}
	header = (TableHeader*)malloc(sizeof(TableHeader));
	if(header==NULL) {
		printf("Unable to allocate memory for table header.\n");
		exit(1);
	}
	fread(header,sizeof(TableHeader),1,fp);
	fclose(fp);
	//show_table_header(header);
	printf("Table identifier: %08x\n", header->f1);
	free(header);
	return 0;
}

