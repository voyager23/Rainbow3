# CUfiles makefile - 23Apr2012

objects =  table_utils.o freduce.o fname_gen.o md5.o tmerge.o

objs = $(addprefix ./obj/,table_utils.o fname_gen.o md5.o)

deps = rainbow.h table_utils.h freduce.h fname_gen.h md5.h

all :  mktab tmerge search
	
#-----------------------------------------------------------------------

thisto : $(objs)
	gcc -Wall -lm thisto.c $^ -o ./bin/$@
	
search: searchtable_v5.cu utils_2.cu utils_2.h
	nvcc searchtable_v5.cu -o ./bin/search
	
mktab : maketable_v4.cu utils_2.cu utils_2.h md5.cu
	nvcc maketable_v4.cu md5.cu -o ./bin/mktab

tmerge : tmerge.cu utils_2.cu utils_2.h tmerge.h
	nvcc tmerge.cu  -o bin/tmerg
	
#-----------------------------------------------------------------------

./obj/%.o : %.cu $(deps)
	nvcc -c $< -o $@

#-----------------------------------------------------------------------

