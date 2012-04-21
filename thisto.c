//      thisto.c
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


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "rainbow.h"
#include "table_utils.h"
#include "freduce.h"
#include "fname_gen.h"
#include "md5.h"

// number of histogram buckets
#define BUCKETS 128
// number of trial chains to generate
#define TRIALS	256
// scaling factor
#define WIDTH 72

int main(int argc, char** argv)
{
	int x,p,counts[BUCKETS];
	TableEntry *target;
	int value,v_max,stars;
	const int f6 = 1;
	const int f5 = f6*26;
	const int f4 = f5*26;
	const int f3 = f4*10;
	const int f2 = f3*10;
	const int f1 = f2*26;
	const int f0 = f1*26;
	char *B;
	
	// clear histogram
	for(x=0;x<BUCKETS;x++) counts[x]=0;
	// allocate space for 1 full chain
	target=(TableEntry*)malloc(sizeof(TableEntry)*LINKS);
	// randomise
	srand(time(NULL));
	// generate some trial chains for analysis
	
	for(x=0;x<TRIALS;x++) {
		// Random password type 'UUnnllU'
		target->initial_password[0]= (rand() % 26) + 'A';
		target->initial_password[1]= (rand() % 26) + 'A';
		target->initial_password[2]= (rand() % 10) + '0';
		target->initial_password[3]= (rand() % 10) + '0';
		target->initial_password[4]= (rand() % 26) + 'a';
		target->initial_password[5]= (rand() % 26) + 'a';
		target->initial_password[6]= (rand() % 26) + 'A';
		target->initial_password[7]= '\0';		
		// compute a chain in detail
		compute_chain(target,LINKS);
		// scan chain for -reduced- passwords
		for(p=1;p<LINKS;p++) {
			B=(target+p)->initial_password;
			// determine a numerical value for a histogram
			value = 0;			
			value =  (B[6]-'A')*f6;
			value += (B[5]-'a')*f5;
			value += (B[4]-'a')*f4;
			value += (B[3]-'0')*f3;
			value += (B[2]-'0')*f2;
			value += (B[1]-'A')*f1;
			value += (B[0]-'A')*f0;
			// save in histo data
			counts[(value%BUCKETS)]+=1;
		} // next link
	} // next trial
	// print the histogram data
	v_max=0;
	// find max value
	for(x=0;x<BUCKETS;x++) if(counts[x] > v_max) v_max=counts[x];
	// print a scaled line for each value
	for(x=0;x<BUCKETS;x++) {
		printf("%2d: ",x);
		stars=(counts[x]*WIDTH) / v_max;
		for(p=0;p<stars;p++) printf("=");
		printf("\n");
	}	
	return(0);
} 
