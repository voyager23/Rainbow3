# CUfiles makefile - 23Apr2012

objects =  table_utils.o freduce.o fname_gen.o md5.o

objs = $(addprefix ./obj/, table_utils.o  fname_gen.o md5.o)

deps = rainbow.h table_utils.h freduce.h fname_gen.h md5.h

all : search
	
#-----------------------------------------------------------------------

thisto : $(objs)
	gcc -Wall -lm thisto.c $^ -o ./bin/$@
	
search: $(objs)
	nvcc searchtable_v4.cu ./bin/search
	
#-----------------------------------------------------------------------
	
./obj/%.o : %.cu $(deps)
	nvcc -c $< -o $@

#-----------------------------------------------------------------------

