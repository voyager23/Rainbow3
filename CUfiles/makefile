# CUfiles makefile - 23Apr2012

objects =  table_utils.o freduce.o fname_gen.o md5.o

objs = $(addprefix ./obj/, table_utils.o  fname_gen.o md5.o)

deps = rainbow.h table_utils.h freduce.h fname_gen.h md5.h

all : search mktab
	
#-----------------------------------------------------------------------

thisto : $(objs)
	gcc -Wall -lm thisto.c $^ -o ./bin/$@
	
search: $(objs)
	nvcc searchtable_v4.cu -o ./bin/search
	
mktab : maketable_v3.cu md5.c md5.h fname_gen.c fname_gen.h
	nvcc -c fname_gen.cu -o ./obj/fname_gen.o
	nvcc maketable_v3.cu md5.c ./obj/fname_gen.o -o ./bin/mktab
	
#-----------------------------------------------------------------------

./obj/%.o : %.cu $(deps)
	nvcc -c $< -o $@

#-----------------------------------------------------------------------

