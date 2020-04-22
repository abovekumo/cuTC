extern "C"
{
    #include "matrixprocess.h"
    #include "sptensor.h" 
    #include "base.h"
    #include <stdio.h>
    #include <sys/time.h>
    #include <stdlib.h>
    #include <time.h> 
}

#include <cuda.h>
#include <cuda_runtime.h>

//macros
#define DEFAULT_NNZNUM 4

/**
  @brief To compute loss on GPU, with segment scan and warp shuffle
  @version Single GPU
**/
__global__ void tc_loss_gpu(sptensor_t * test,
                            ordi_matrix ** mats,
                            double * loss
                            )
{
  __shared__ double warpsum[((idx_t)DEFAULT_BLOCKSIZE) / 32];
  __shared__ uint32_t warpmask[((idx_t)DEFAULT_BLOCKSIZE) / 32];
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  idx_t warpid = threadIdx.x / 32;
  idx_t laneid = threadIdx.x % 32;
  idx_t nnzbatch = test->nnz / ((idx_t) DEFAULT_NNZNUM) + 1;
  double sum_l = 0.0;

  //intialize the mask
  if(laneid % 32 == 0) warpmask[warpid] = 0xffffffff;
  //prepare the masks 
  idx_t basicposition = tileid * DEFAULT_NNZNUM;
  if(basicposition == test->nnz)
  {
    uint32_t newmask = 0xffffffff;
    newmask = __brevll((newmask << (32-laneid)));
    warpmask[warpid] = newmask;
  }

  __syncwarp();

  uint32_t selfmask = warpmask[warpid];

  if(tileid < nnzbatch)
  {
      
    //inter thread operation
    for(int j = 0; j < DEFAULT_NNZNUM; j++)
    {
      if(basicposition >= test->nnz) break;
      double rmse_l = test->vals[basicposition];
      idx_t a = test->ind[0][basicposition] - 1;
      idx_t b = test->ind[1][basicposition] - 1;
      idx_t c = test->ind[2][basicposition] - 1;
      for(int i = 0; i < DEFAULT_NFACTORS ; i++)
      {
        rmse_l -= mats[0]->values[a * DEFAULT_NFACTORS + i] * mats[1]->values[b * DEFAULT_NFACTORS + i] * mats[2]->values[c * DEFAULT_NFACTORS + i];
      }
      rmse_l = rmse_l * rmse_l;
      sum_l += rmse_l;
      basicposition++;
    }
    __syncwarp(selfmask);
    //inter warp operation
    if(((laneid % 2) == 0)) 
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 1 , 32);
      sum_l += sum_n;
    }

    if(((laneid % 4) == 0))
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 2 , 32);
      sum_l += sum_n;
    }

    if(((laneid % 8) == 0))
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 4 , 32);
      sum_l += sum_n;
    }

    if(((laneid % 16) == 0))
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 8 , 32);
      sum_l += sum_n;
    }

    if(((laneid % 32) == 0))
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 16 , 32);
      sum_l += sum_n;
      warpsum[warpid] = sum_l;
    }

    __syncthreads();

    //inter block operation
    if(tid == 0)
    {
      double sum_f = 0;
      for(int j = 0; j < ((idx_t)DEFAULT_BLOCKSIZE) / 32; j++)
      {
        sum_f += warpsum[j];
      }
      //inter device operation
      atomicAdd(loss, sum_f);  
    }
      
  }
 
}


//macros
#define DEFAULT_MATNUM 8

/**
  @brief To compute frobenius norm on GPU, with segment scan and warp shuffle
  @version Single GPU
**/
__global__ void tc_frob_gpu(double regularization_index,
                            ordi_matrix * mats,
                            double * frob)
{
  __shared__ double warpsum[((idx_t)DEFAULT_BLOCKSIZE) / 32];
  __shared__ uint32_t warpmask[((idx_t)DEFAULT_BLOCKSIZE) / 32];
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  idx_t warpid = threadIdx.x / 32;
  idx_t laneid = threadIdx.x % 32;
  idx_t nnz = (mats->I) * (mats->J);
  
  idx_t nnzbatch = (nnz) / ((idx_t) DEFAULT_MATNUM) + 1;
  double sum_l = 0.0; 

  //intialize the mask
  if(laneid % 32 == 0) warpmask[warpid] = 0xffffffff;
  idx_t basicposition = tileid * DEFAULT_MATNUM;

  //prepare the masks 
  if(basicposition == nnz)
  {
    uint32_t newmask = 0xffffffff;
    newmask = __brevll((newmask << (32-laneid)));
    warpmask[warpid] = newmask;
  }

  __syncwarp();
  uint32_t selfmask = warpmask[warpid];

  if(tileid < nnzbatch)
  {
    double frob_l = 0.0;
    for(int j = 0; j < DEFAULT_MATNUM; j++)
    {
      if(basicposition >= nnz) break;
      double mat_tmp = mats->values[basicposition];
      frob_l += mat_tmp * mat_tmp * regularization_index;
      sum_l += frob_l;
      basicposition++;
    }
    
    __syncwarp(selfmask);

    //inter warp operation
    if(((laneid % 2) == 0)) 
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 1 , 32);
      sum_l += sum_n;
    }

    if(((laneid % 4) == 0))
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 2 , 32);
      sum_l += sum_n;
    }

    if(((laneid % 8) == 0))
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 4 , 32);
      sum_l += sum_n;
    }

    if(((laneid % 16) == 0))
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 8 , 32);
      sum_l += sum_n;
    }

    if(((laneid % 32) == 0))
    {
      double sum_n = __shfl_down_sync(selfmask, sum_l, 16 , 32);
      sum_l += sum_n;
      warpsum[warpid] = sum_l;
    }

    __syncthreads();

    //inter block operation
    if(tid == 0)
    {
      double sum_f = 0;
      for(int j = 0; j < ((idx_t)DEFAULT_BLOCKSIZE) / 32; j++)
      {
        sum_f += warpsum[j];
      }
      //inter device operation
      atomicAdd(frob, sum_f);  
    }

  } 
}