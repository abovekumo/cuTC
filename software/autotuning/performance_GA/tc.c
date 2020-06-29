//include
#include "base.h"
#include "matrixprocess.h"
#include "sptensor.h"
#include "io.h"
#include "completion.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include<sys/time.h>


/*
The code number of algorithms:
0 ALS | 1 SGD | 2 CCD+
*/

idx_t tc_main(char* input_traina, char* input_trainb, char* input_trainc, char* input_validation, char *input_test, int SGD_DEFAULT_BLOCKSIZE, int SGD_DEFAULT_T_TILE_LENGTH)
{
  int flag;
  idx_t nmodes = 2;
  int j = 0;
  //int algorithm_index = atoi(argv[6]);
  //printf("algorithm: %d\n", algorithm_index);
  //int SGD_DEFAULT_BLOCKSIZE = atoi(argv[6]);
  //int SGD_DEFAULT_T_TILE_LENGTH = atoi(argv[7]);
  //printf("blocksize = %d tilesize = %d\n", SGD_DEFAULT_BLOCKSIZE, SGD_DEFAULT_T_TILE_LENGTH);
  int algorithm_index = 1;
  double cost;
  //clock_t start;
  //clock_t end;
  idx_t exectime;
  struct timeval start, end;
  gettimeofday(&start, NULL);

  #ifdef DEBUG
  printf("%d\n",argc);
  #endif

  //load tensor for train and validate
  sptensor_t * traina, * trainb, * trainc;
  sptensor_t * validation;
  sptensor_t * test;
  
  traina = tt_read(input_traina);
  trainb = tt_read(input_trainb);
  trainc = tt_read(input_trainc);
  validation = tt_read(input_validation); 
  test = tt_read(input_test);
    
  exectime = tc_main_ciss(traina, trainb, trainc, validation, test, algorithm_index, SGD_DEFAULT_BLOCKSIZE, SGD_DEFAULT_T_TILE_LENGTH);
  
  tt_free(traina); 
  tt_free(trainb);
  tt_free(trainc); 
  tt_free(validation);
  tt_free(test);
  gettimeofday(&end, NULL);
  double timeuse = 1000000*(end.tv_sec - start.tv_sec) + end.tv_usec-start.tv_usec;
  //printf("total time = %f s\n", timeuse/1000000);
  //printf("kernel exectime = %ld\n", exectime);

  return exectime;
}

