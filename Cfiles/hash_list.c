/*
 * hash_list.c
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
#include <string.h>
#include "../common/rainbow.h"

#define BUFFERSIZE 128
#define MAXHASHES 128

typedef struct {
	int count;	// no. of entries
	int idx;	// working index
	TableEntry target_list[MAXHASHES];	//array of TableEntry
}TargetList;


int filter_str(char *);
void hash2uint32(char *hash_str, uint32_t *H);
int make_target_list(char *fpath, TargetList *tl);

//------------------------------------------------------------------------------
int filter_str(char *hash_str) {
	// Read the string to an internal buffer
	// dropping all chars not in "0-9a-fA-F".
	// Copy result string back to origin.
	// Return string length.
	const char *charset = "1234567890abcdefABCDEF";
	char buffer[BUFFERSIZE];
	int hash_idx,buff_idx,len;
	
	// sanity check
	len=strlen(hash_str);
	if((len > BUFFERSIZE-1)||(len < 64)) return(0);
	
	// scan the input string
	hash_idx=buff_idx=0;
	while(hash_idx < len) {
		if(strchr(charset,hash_str[hash_idx])!=NULL) {
			buffer[buff_idx++]=hash_str[hash_idx++];
		} else {
			hash_idx++;
		}
	}
	// add the termination
	buffer[buff_idx]='\0';
	// copy back result
	strcpy(hash_str,buffer);
	return(buff_idx);
}
//------------------------------------------------------------------------------
void hash2uint32(char *hash_str, uint32_t *H) {
	// hash_str must be 64 byte hexadecimal string
	const int words=8;
	char buffer[9], *source=hash_str;
	int i,len;
	len = strlen(hash_str);
	if(len != sizeof(unsigned)*words*2) {
		printf("Error - hash2uint32: hash_str length=%d\n",len);
		exit(1);
	}	
	for(i=0;i<words;i++) {
		strncpy(buffer,source,8);
		buffer[8]='\0';
		sscanf(buffer,"%x",H+i);
		source+=8;
	}
}
//------------------------------------------------------------------------------
int make_target_list(char *fpath, TargetList *t_list) {
	
	// Parameters: path to text file, pointer to TargetList
	// Requires filter_str() and hash2uint32()
		
	FILE *fp_list;
	char hash_str[BUFFERSIZE];
	int len;
	
	// Sanity checks
	if(t_list != NULL) {
		t_list->count=t_list->idx=0;
	} else {
		printf("Error - TargetList pointer is NULL.\n");
		return(0);
	}	
	fp_list=fopen(fpath,"r");
	if(fp_list==NULL) {
		perror("hash_list.txt");
		return(0);
	}
	// scan file and save data in TargetList
	fgets(hash_str,BUFFERSIZE,fp_list);
	while(!feof(fp_list)&&(t_list->count<MAXHASHES)) {
		len = filter_str(hash_str);
		if (len == 64) {
			//printf("Hash string: %s  %d\n",hash_str,len);
			hash2uint32(hash_str,(t_list->target_list + t_list->count)->final_hash);
			strcpy((t_list->target_list+t_list->count)->initial_password,"hash_list");
			t_list->count++;
		}		
		fgets(hash_str,BUFFERSIZE,fp_list);
	}
	fclose(fp_list);
	return(t_list->count);
}
//==============================================================================
int main(int argc, char **argv)
{
	TargetList tlist;
	int targets,dx;
	
	targets=make_target_list("./hash_list.txt", &tlist);
	if(targets>0) {
		printf("%d targets found.\n",targets);	
		for(tlist.idx=0;tlist.idx<tlist.count;tlist.idx++) {
			for(dx=0;dx<8;dx++) printf("%08x ", (tlist.target_list+tlist.idx)->final_hash[dx]);
			printf("\n");
		}
	}
	return 0;
}
//==============================================================================
