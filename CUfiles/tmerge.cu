//      tmerge.cu
//      
//      Copyright 2012 Michael <mike@mike-n110>
//      
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//      
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//      
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.

/*
 * Filename: tmerge.cu
 * develop code for merging two rainbow tables, ignoring any chains
 * that have matching final_hashes. Result is new rainbow table with
 * corrected header.
 * RbowTab_sort_left.rbt  RbowTab_sort_right.rbt
 * 12feb2012 - rewrite as callable function for
 * use in maketable (table_utils?).
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include "tmerge.h"

#include "utils_2.h"
#include "utils_2.cu"
	
// ------Definitions-----

int tmerge_2(const char *id_str) {
	FILE *fp_master, *fp_sorted, *fp_merged;
	TableHeader master_header, sorted_header;
	TableEntry master_entry, sorted_entry, hold_entry, merged_entry;
	char fname_master[128];
	char fname_sorted[128];
	char fname_merged[128];
	int compare,first_pass,entries_written,discarded;
	
	sprintf(fname_master,"./rbt/master_%s.rbt",id_str);
	fp_master=fopen(fname_master,"r");
	sprintf(fname_sorted,"./rbt/sort_%s.rbt",id_str);
	fp_sorted=fopen(fname_sorted,"r");	
	
	if(fp_sorted==NULL) {
		if(fp_master==NULL) {
			perror("No master or sort file");
		} else {
			perror("No sort file");
			fclose(fp_master);
		}
		return(1);
	}
	
	if(fp_master==NULL) {	//fp_sorted is available
		fclose(fp_sorted);
		rename(fname_sorted,fname_master);
		printf("Renamed %s to %s\n",fname_sorted,fname_master);
		return(1);
	}
	
	sprintf(fname_merged,"./rbt/merged_%s.rbt",id_str);
	fp_merged=fopen(fname_merged,"w");
	if(fp_merged==NULL) {
		perror("No writable merge file:");
		fclose(fp_master);
		fclose(fp_sorted);
		return(1);
	}
	
	fread(&master_header,sizeof(TableHeader),1,fp_master);
	fread(&sorted_header,sizeof(TableHeader),1,fp_sorted);
	fread(&master_entry,sizeof(TableEntry),1,fp_master);
	fread(&sorted_entry,sizeof(TableEntry),1,fp_sorted);
	first_pass = 1;	entries_written=discarded=0;
	while((!feof(fp_master))&&(!feof(fp_sorted))) {
		compare = hash_compare_uint32_t(&(master_entry.final_hash[0]),&(sorted_entry.final_hash[0]));
		switch(compare) {
			case(1):	// master > sorted
				//printf("master > sorted hold=sorted\n");
				memcpy(&hold_entry,&sorted_entry,sizeof(TableHeader));
				fread(&sorted_entry,sizeof(TableEntry),1,fp_sorted);
			break;
			case(-1):	// master < sorted
				//printf("master < sorted hold=master\n");
				memcpy(&hold_entry,&master_entry,sizeof(TableEntry));
				fread(&master_entry,sizeof(TableEntry),1,fp_master);			
			break;
			case(0):	// master == sorted
				//printf("master == sorted hold=master\n");
				// copy master to hold
				memcpy(&hold_entry,&master_entry,sizeof(TableEntry));
				fread(&master_entry,sizeof(TableEntry),1,fp_master);
				fread(&sorted_entry,sizeof(TableEntry),1,fp_sorted);
			break;
		} //end switch
		if(first_pass==1) {
			//printf("First pass: hold -> merged -> table\n");
			first_pass=0;
			fwrite(&master_header,sizeof(TableHeader),1,fp_merged);
			memcpy(&merged_entry,&hold_entry,sizeof(TableEntry));			
			fwrite(&merged_entry,sizeof(TableEntry),1,fp_merged);
			entries_written++;
		} else {
			if(hash_compare_uint32_t(&hold_entry.final_hash[0],&merged_entry.final_hash[0])!=0) {
				//printf("hold != merged hold -> merged -> table\n");
				memcpy(&merged_entry,&hold_entry,sizeof(TableEntry));
				fwrite(&merged_entry,sizeof(TableEntry),1,fp_merged);
				entries_written++;
			} else {
				// hold == last write to merged
				//printf("hold == merged discard\n");
				discarded++;
			}
		}		
	} //end while
	if(feof(fp_sorted)) {
		// write balance of master to merged
		// merged_entry has last write to file
		//printf("writing balance of master to merged\n");
		while(!feof(fp_master)) {
			fread(&hold_entry,sizeof(TableEntry),1,fp_master);
			if(hash_compare_uint32_t(&hold_entry.final_hash[0],&merged_entry.final_hash[0])!=0) {
				//printf("hold != merged hold -> merged -> table\n");
				memcpy(&merged_entry,&hold_entry,sizeof(TableEntry));
				fwrite(&merged_entry,sizeof(TableEntry),1,fp_merged);
				entries_written++;
			} else {
				// hold == last write to merged
				//printf("hold == merged discard\n");
				discarded++;
			}
		}
	} else {
		// write balance of sorted to merged
		// merged_entry has last write to file
		//printf("write balance of sorted to merged\n");
		while(!feof(fp_sorted)) {
			fread(&hold_entry,sizeof(TableEntry),1,fp_sorted);
			if(hash_compare_uint32_t(&hold_entry.final_hash[0],&merged_entry.final_hash[0])!=0) {
				//printf("hold != merged hold -> merged -> table\n");
				memcpy(&merged_entry,&hold_entry,sizeof(TableEntry));
				fwrite(&merged_entry,sizeof(TableEntry),1,fp_merged);
				entries_written++;
			} else {
				// hold == last write to merged
				//printf("hold == merged discard\n");
				discarded++;
			}
		}
	}
	// rewind merged, read header, adjust header data, rewind merged, write new header
	rewind(fp_merged);
	fread(&master_header,sizeof(TableHeader),1,fp_merged);
	master_header.entries = entries_written;
	rewind(fp_merged);
	fwrite(&master_header,sizeof(TableHeader),1,fp_merged);
	
	fclose(fp_merged);
	fclose(fp_sorted);
	fclose(fp_master);
	printf("Complete. %d entries written and %d entries discarded.\n",entries_written,discarded);
	printf("Table written to %s\n\n",fname_merged);
	return(0);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int filter(const struct dirent* dp) {
	const char *posn;
	posn = strstr(dp->d_name, "sort_" );
	return((posn != NULL));
}

//+++++++++++++++++++++++++++++++++Main Code++++++++++++++++++++++++++++++++++++
int main(int argc, char** argv)
{	
	char filename_1[128], filename_2[128];
	
	
	printf("========= Tmerge =========\n");
	// Required parameter is the table identifier
	// Supplied as hex string of the form 0x12ab34cd
	// Stored internally as uint32_t
	// String form used to generate the table name
	// of the form "sort_0x12ab34cd.rbt
	
	if(argc != 2) {
		printf("Table Identifier missing.\nUsage: tmerge 0x1234abcd\n");
		exit(1);
	}

	// call function
	if(tmerge_2(argv[1])!=0) exit(1);
	
	// read a .rbt file into memory and print out header
	FILE *fp;
	TableHeader header;	
	sprintf(filename_1,"./rbt/merged_%s.rbt",argv[1]);
	fp=fopen(filename_1,"r");
	fread(&header,sizeof(TableHeader),1,fp);
	show_table_header(&header);
	fclose(fp);
	
	// remove sorted
	sprintf(filename_1,"./rbt/sort_%s.rbt",argv[1]);
	if( remove(filename_1 ) != 0 )
		perror( "Error deleting sorted file" );
	else
		puts( "Sorted file successfully deleted" );

	// master -> saved
	sprintf(filename_1,"./rbt/master_%s.rbt",argv[1]);
	sprintf(filename_2,"./rbt/saved_%s.rbt",argv[1]);
	if( rename(filename_1,filename_2) != 0 )
		perror( "Error renaming master file" );
	else
		puts( "Master file renamed to save" );
	
	// merged -> master
	sprintf(filename_2,"./rbt/merged_%s.rbt",argv[1]);
	if( rename(filename_2,filename_1) != 0 )
		perror( "Error renaming merged file" );
	else
		puts( "Merged file renamed to master" );
	
	return 0;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

