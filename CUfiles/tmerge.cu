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
	
// ------Definitions-----
int tmerge(char *sort){
	
	TableHeader *hdr_left, *hdr_right, *hdr_merge;
	TableEntry  *ent_left, *ent_right, *ent_merge;
	TableEntry  *l_ptr, *l_end, *r_ptr, *r_end, *merge_ptr, *merge_end;	
	unsigned entries_total, entries_merged, entries_count, discarded;
	int compare,r1,r2,error_flag;
	
	const char *merge = "./rbt/RbowTab_merge.rbt";
	const char *new_merge = "./rbt/RbowTab_merge.rbt.new";
	const char *rbt_sav = "./rbt/RbowTab_merge.rbt.sav";
	
	FILE *fp_left, *fp_right, *fp_merge;
	
	// check that the original merged file is valid for reading (fp_left)
	fp_left = fopen(merge,"r");
	if(fp_left==NULL) {
		printf("Tmerge - invalid master merged file %s\n",merge);
		return(1);
	}
	// check that the newly sorted file is valid for reading (fp_right)
	fp_right = fopen(sort,"r");
	if(fp_right==NULL) {
		printf("Tmerge - invalid candidate file %s\n",sort);
		fclose(fp_left);
		return(1);
	}	
	
	hdr_left = (TableHeader*)malloc(sizeof(TableHeader));
	fread(hdr_left,sizeof(TableHeader),1,fp_left);	
	
	hdr_right = (TableHeader*)malloc(sizeof(TableHeader));
	fread(hdr_right,sizeof(TableHeader),1,fp_right);

	// allocate for merged table
	entries_total = hdr_left->entries + hdr_right->entries;
	hdr_merge = (TableHeader*)malloc(sizeof(TableHeader));
	ent_merge = (TableEntry*)malloc(sizeof(TableEntry)*entries_total);

	// allocate for source tables
	ent_left  = (TableEntry*)malloc(sizeof(TableEntry)*hdr_left->entries);
	ent_right = (TableEntry*)malloc(sizeof(TableEntry)*hdr_right->entries);

	// read source entries
	fread(ent_left,sizeof(TableEntry),hdr_left->entries,fp_left);
	fclose(fp_left);
	fread(ent_right,sizeof(TableEntry),hdr_right->entries,fp_right);
	fclose(fp_right);

	// set sentinels.
	l_end = ent_left + hdr_left->entries;
	r_end = ent_right + hdr_right->entries;
	merge_end = ent_merge + entries_total;

	// set working pointers and counter
	entries_merged = 0;
	discarded = 0;
	l_ptr = ent_left;
	r_ptr = ent_right;
	merge_ptr = ent_merge;

	// Main loop
	while ((l_ptr < l_end)&&(r_ptr < r_end)) {
		compare = hash_compare_uint32_t(l_ptr->final_hash, r_ptr->final_hash);
		if (compare == 1) {
			// left > right
			*merge_ptr = *r_ptr;
			entries_merged++;
			r_ptr++;
			merge_ptr++;
		}
		else if (compare == -1) {
			// left < right
			*merge_ptr = *l_ptr;
			entries_merged++;
			l_ptr++;
			merge_ptr++;			
		}
		else {
			// left == right - Discard right entry
			// printf("Discarding entry from right table\n");
			*merge_ptr = *l_ptr;
			entries_merged++;
			discarded++;
			l_ptr++;
			merge_ptr++;
			r_ptr++;
		}		
		// now test for end conditions
		if(r_ptr == r_end) {
				// write balance of left to merge
				entries_count = (l_end - l_ptr);
				memcpy(merge_ptr, l_ptr, sizeof(TableHeader)*entries_count);
				// adjust count and pointers
				l_ptr = l_end;
				merge_ptr += entries_count;
				entries_merged += entries_count;
		}
		if(l_ptr == l_end) {
				// write balance of right to merge
				entries_count = (r_end - r_ptr);
				memcpy(merge_ptr, r_ptr, sizeof(TableHeader)*entries_count);
				// adjust count and pointers
				r_ptr = r_end;
				merge_ptr += entries_count;
				entries_merged += entries_count;
		}
	} // end_main_loop
	
	// write the merged table to disk
	hdr_merge->hdr_size = sizeof(TableHeader);
	hdr_merge->entries = entries_merged;
	hdr_merge->links = hdr_left->links;
	//sanity check for destination file
	fp_merge = fopen(new_merge,"w");
	if(fp_merge!=NULL) {
		r1=fwrite(hdr_merge,sizeof(TableHeader),1,fp_merge);
		r2=fwrite(ent_merge,sizeof(TableEntry),entries_merged,fp_merge);
		fclose(fp_merge);
		if((r1==1)&&(r2==entries_merged)) {
			printf("Removing the original sorted table file.\n");
			remove(sort);	// not required anymore	
			
			// remove previous merge.rbt.sav ...
			remove(rbt_sav);
			
			// ... and save old merge.rbt to merge.rbt.sav
			rename(merge, rbt_sav);
			
			// rename _merge.rbt.new to _merge.rbt
			printf("Renaming new merge file to _merge.rbt\n");
			rename(new_merge, merge);
			
			error_flag=0;
		} else {
			printf("Tmerge - invalid file %s Nothing saved\n",new_merge);
			error_flag=1;
		}
	}
	
	// Clean up pointers
	free(ent_right);
	free(ent_left);
	free(ent_merge);
	free(hdr_merge);
	free(hdr_right);
	free(hdr_left);	
	// final actions		
	printf("Merged %d entries and discarded %d.\n",entries_merged,discarded);
	return(error_flag);
}

int hash_compare_uint32_t(uint32_t *left, uint32_t *right) {
	int i;
	for(i=0; i<8; i++) {
		if(*(left+i) > *(right+i)) 
			return(1);
		else if(*(left+i) < *(right+i)) 
			return(-1);
		else
			continue;
	}
	return(0);
}

int filter(const struct dirent* dp) {
	const char *posn;
	posn = strstr(dp->d_name, "RbowTab_sort64_" );
	return((posn != NULL));
}

// ------Main Code------

int main(int argc, char** argv)
{	
    struct dirent **namelist;
    int n;
    char buffer[256];

    n = scandir("./rbt", &namelist, filter, alphasort);
    if (n < 0)
        perror("scandir");
    else {
        while (n--) {
			sprintf(buffer,"./rbt/%s",namelist[n]->d_name);
            printf("Calling tmerge against: %s\n", buffer);
            tmerge(buffer);
            free(namelist[n]);
        }
        free(namelist);
    }
	return 0;
}
