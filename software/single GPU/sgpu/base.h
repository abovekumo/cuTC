#ifndef BASE_H
#define BASE_H

//includes
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <assert.h>

#include "/usr/local/cuda-10.2/include/cuda.h"
#include "/usr/local/cuda-10.2/include/cuda_runtime.h"



//constants
typedef uint64_t idx_t;
#define DEFAULT_ITERATE 50
#define DEFAULT_NFACTORS 16
#define DEFAULT_MAX_ITERATE 1000
#define DEFAULT_ERROR_ALS 1e-8
#define DEFAULT_ERROR_SGD 1e-8
#define DEFAULT_ERROR_CCD 1e-8
#define DEFAULT_RANK 16 
#define MAX_NMODES 4

//for evaluation
#define NUM_INNER 1
#define MAX_SECONDS -1
#define MAX_BADEPOCHS 3
#define NBADEPOCHS 0
#define BEST_EPOCH 0
#define BEST_RMSE 10000000
#define TOLERANCE 1e-4

//regularization parameters
#define SGD_REGULARIZATION 1e-3
#define CCD_REGULARIZATION 100
#define ALS_REGULARIZATION 10 
#define SGD_MODIFICATIONA 1e-5
#define SGD_MODIFICATIONB 1e-5
#define SGD_MODIFICATIONC 1e-5

//warp size 
#define ALS_WARPSIZE 32
#define CCD_WARPSIZE 32
#define SGD_WARPSIZE 16

//for as-als
#define ALS_BLOCKSIZE 256

//for als
#define ALS_BUFSIZE 2048

//for sgd
#define LEARN_RATE 1e-3


#define DEFAULT_BLOCKSIZE 256
#define DEFAULT_T_TILE_LENGTH 32
#define UNIT_LENGTH 134217728 //how many uint64_t/double in a GB ‬
#define DEFAULT_T_TILE_WIDTH 3

#ifdef WITHMPI
#define  DEFAULT_MPI_PROCESS  1
#endif

//macro functions
#define SS_MIN(x,y) ((x) < (y) ? (x) : (y))
#define SS_MAX(x,y) ((x) > (y) ? (x) : (y))


//basic functions
idx_t argmax_elem(idx_t const* const arr,
                     idx_t const N);           //scan a list and return the index of the maximum valued element.

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
                    cudaGetErrorString( err ), file, line, err );
        fflush(stdout);
        exit(-1);
    }
}

#endif
