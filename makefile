# New makefile 02Apr2012

objects =  table_utils.o freduce.o fname_gen.o md5.o

objs = $(addprefix ./obj/, table_utils.o  fname_gen.o md5.o)

deps = rainbow.h table_utils.h freduce.h fname_gen.h md5.h

all : search csearch

cmaketab : $(objs)
	gcc -Wall -pthread -lm maketable_v7.c $^ -o ./bin/$@
	
csearch : $(objs)
	gcc -Wall -pthread -lm  searchtable_v7.c $^ -o ./bin/$@
	
#-----------------------------------------------------------------------

thisto : $(objs)
	gcc -Wall -lm thisto.c $^ -o ./bin/$@
	
search: $(objs)
	nvcc -g -G searchtable_v4.cu ./obj/table_utils.o ./obj/md5.o -o ./bin/search
	
#-----------------------------------------------------------------------
./obj/%.o : %.c $(deps)
	nvcc -c $< -o $@
	
./obj/%.o : %.cu $(deps)
	nvcc -c $< -o $@
#-----------------------------------------------------------------------

