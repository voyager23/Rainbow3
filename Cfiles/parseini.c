/*
 * parseini.c
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
#include <iniparser.h>

#define PATHSIZE 128
#define NPATHS 128

typedef struct mfl {
	char (*path_ptr) [PATHSIZE];
	char master_file [NPATHS][PATHSIZE];
	int count;
}MasterFileList;

int main(int argc, char **argv)
{
	char *ini_name = "rainbow.ini";
	char **key_str;	
	int n_keys,i;	
	dictionary *ini;
	MasterFileList mfl;
	
	// initialise masterfilelist
	mfl.path_ptr = &mfl.master_file[0];
	mfl.count = 0;
	
	// open .ini file
	ini = iniparser_load(ini_name);
    if (ini==NULL) {
        fprintf(stderr, "cannot parse file: %s\n", ini_name);
        return -1 ;
    }    
    // debug dump file contents
    // iniparser_dump(ini, stderr);    
    
    // get number of keys and pointer to first key
    n_keys = iniparser_getsecnkeys(ini,"masterfiles");
    key_str= iniparser_getseckeys(ini,"masterfiles");
    
    // loop through key/value pairs copying
    // value to masterfilelist and incrementing
    // the count.
    for(i=0; i<n_keys;i++) {	
		printf("%s\n",*key_str);
		printf("%s\n",iniparser_getstring(ini,*key_str,NULL));
		strcpy((char*)mfl.path_ptr, iniparser_getstring(ini,*key_str++,NULL));
		mfl.path_ptr++;	mfl.count++;
    }
    printf("\n");
    // reset the pointer in mfl and loop through values recovered
	mfl.path_ptr = &mfl.master_file[0];
	while(mfl.count-->0) {
		printf("->%s<-\n",mfl.path_ptr++);
	}
	
	// Cleanup
	iniparser_freedict(ini);	
	return 0;
}

