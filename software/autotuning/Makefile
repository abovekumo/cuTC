SWCC = sw5cc.new #-Wall
CC = mpicc #-Wall
CFLAGS = -host -O3 -g -OPT:Olimit=0 -lm -lm_slave -OPT:IEEE_arith=2
CFLAG_SLAVE = -slave -O3 -msimd -g -OPT:Olimit=0 -OPT:IEEE_arith=2
LIBS = -hybrid -g -lslave
#LINK = -lm ./cblas_LINUX0324.a ./libswblas0324.a 

TARGET = ga

$(TARGET): fitness.o ga.o init.o mt_mpi.o report.o
	$(CC) $(LIBS) $^ -o $@ $(LINK)

ga.o: ga.c config.h fitness.h ga.h init.h mt_mpi.h report.h types.h common.h
	$(CC) $(CFLAGS) -host -c ga.c

fitness.o: fitness.c config.h fitness.h types.h
	$(CC) $(CFLAGS) -host -c fitness.c

init.o: init.c config.h init.h types.h mt_mpi.h report.h
	$(CC) $(CFLAGS) -c init.c

mt_mpi.o: mt_mpi.c mt_mpi.h
	$(CC) $(CFLAGS) -c mt_mpi.c

report.o: report.c config.h report.h types.h
	$(CC) $(CFLAGS) -c report.c

clean:
	rm *.o $(TARGET)
distclean:
	rm *.o ga
