# CUfiles makefile - 23Apr2012

objects =  table_utils.o freduce.o fname_gen.o md5.o tmerge.o

objs = $(addprefix ./obj/,table_utils.o fname_gen.o md5.o)

deps = rainbow.h table_utils.h freduce.h fname_gen.h md5.h

all : search mktab tmerge
	
#-----------------------------------------------------------------------

thisto : $(objs)
	gcc -Wall -lm thisto.c $^ -o ./bin/$@
	
search: $(objs)
	nvcc searchtable_v5.cu -o ./bin/search
	
mktab : maketable_v3.cu $(objs)
	nvcc maketable_v3.cu ./obj/fname_gen.o ./obj/md5.o -o ./bin/mktab

tmerge : tmerge.cu $(objs)
	nvcc tmerge.cu -o ./bin/tmerge
	
#-----------------------------------------------------------------------

./obj/%.o : %.cu $(deps)
	nvcc -c $< -o $@

#-----------------------------------------------------------------------

