#ifndef TC_LOSS_H
#define TC_LOSS_H

/**
 @brief For the loss computation. 
 @version Single GPU
**/

extern "C"
{
    #include "matrixprocess.h"
    #include "sptensor.h"  
}

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tc_loss_gpu(sptensor_t * test,
                            ordi_matrix ** mats,
                            double * loss);

__global__ void tc_frob_gpu(double regularization_index,
                            ordi_matrix ** mats,
                            double * frob);


#endif