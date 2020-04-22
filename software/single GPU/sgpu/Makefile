CC = icc

NVCC = nvcc

CFLAGS = -O3 -g  -std=c99 -xHost -D_GNU_SOURCE -D CISS_DEBUG
#NCFLAGS = -O2 -g -G -arch=sm_70 -D ALSAS_DEBUG
NCFLAGS = -O2 -g -arch=sm_70
#-D ALS_DEBUG


TARGET = tc

$(TARGET):tc.o base.o io.o util.o sptensor.o matrixprocess.o ciss.o loss.o ccd.o als.o sgd.o completion.o
	$(CC) $(CFLAGS) $^ -o $@ -L/usr/local/cuda-10.2/lib64 -lcudart -lcuda -lcusolver

tc.o: tc.c
	$(CC) -o $@ -c $(CFLAGS) $<
base.o: base.c
	$(CC) -o $@ -c $(CFLAGS) $<
#csf.o: csf.c
#	$(CC) -o $@ -c $(CFLAGS) $<
util.o:util.c
	$(CC) -o $@ -c $(CFLAGS) $<
matrixprocess.o:matrixprocess.c 
	$(CC) -o $@ -c $(CFLAGS) $<  
io.o: io.c 
	$(CC) -o $@ -c $(CFLAGS) $<
sptensor.o: sptensor.c
	$(CC) -o $@ -c $(CFLAGS) $<
ciss.o: ciss.c
	$(CC) -o $@ -c $(CFLAGS) $<
completion.o: completion.c als.cu ccd.cu sgd.cu
	$(CC) -o $@ -c $(CFLAGS) $<
loss.o: loss.cu
	$(NVCC) -o $@ -c $(NCFLAGS) $< -L/usr/local/cuda-10.2/lib64 -lcudart -lcuda
als.o: als.cu 
	$(NVCC) -o $@ -c $(NCFLAGS) $< -L/usr/local/cuda-10.2/lib64 -lcudart -lcuda -lcusolver
sgd.o: sgd.cu 
	$(NVCC) -o $@ -c $(NCFLAGS) $< -L/usr/local/cuda-10.2/lib64 -lcudart -lcuda
ccd.o: ccd.cu 
	$(NVCC) -o $@ -c $(NCFLAGS) $< -L/usr/local/cuda-10.2/lib64 -lcudart -lcuda

#cpd.o: cpd.c
#	$(CC) -o $@ -c $(CFLAGS) $<
#matrixnormal_host.o: matrixnormal_host.c
#	$(CC) -o $@ -c $(CFLAGS) $<
#matrixnormal_slave.o: matrixnormal_slave.c
#	$(CC) -o $@ -c $(CFLAG_SLAVE) $<
#mttkrp.o: mttkrp.c
#	$(CC) -o $@ -c $(CFLAGS) $<
#mttkrp_slave.o:mttkrp_slave.c
#	$(CC) -o $@ -c $(CFLAG_SLAVE) $<
clean:
	rm *.o $(TARGET)
