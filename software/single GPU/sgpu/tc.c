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

/*
The code number of algorithms:
0 ALS | 1 SGD | 2 CCD+
*/

int main(int argc, char** argv)
{
  int flag;
  idx_t nmodes = 3;
  int j = 0;
  int algorithm_index = 1 ;
  double cost;
  clock_t start;
  clock_t end;

  #ifdef DEBUG
  printf("%d\n",argc);
  #endif

  //load tensor for train and validate
  sptensor_t* traina, * trainb, * trainc;
  sptensor_t* validation;
  sptensor_t* test;
  #ifdef CISS_DEBUG
  printf("%s\n",argv[1]);
  printf("%s\n",argv[2]);
  printf("%s\n",argv[3]);  
  #endif
  traina = tt_read(argv[1]);
  trainb = tt_read(argv[2]);
  trainc = tt_read(argv[3]);
  validation = tt_read(argv[4]); 
  test = tt_read(argv[5]);
  
  #ifdef DEBUG
  printf("finish tensor reading\n");
  #endif

  tc_main_ciss(traina, trainb, trainc, validation, test, algorithm_index);
  
  tt_free(traina); 
  tt_free(trainb);
  tt_free(trainc);
  tt_free(validation);
  tt_free(test);
  
  
  return 0;
}

