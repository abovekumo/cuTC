extern "C"
{
#include "completion.h"
#include "base.h"
#include "ciss.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
}

#include "als.cuh"
#include "loss.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <omp.h>
#include "nccl.h"


#define HANDLE_SOLVERERR( err ) (HandleSolverErr( err, __FILE__, __LINE__ ))

static void HandleSolverErr( cusolverStatus_t err, const char *file, int line )
{
    if(err != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: in %s at line %d (error-code %d)\n",
                    file, line, err );
        fflush(stdout);
        exit(-1);
    }
}

#define HANDLE_NCCLERR(cmd) do{   \
  ncclResult_t r=cmd;    \
  if(r!=ncclSuccess){     \
      printf("Failed, NCCL error %s:%d '%s'\n",__FILE__,__LINE__,ncclGetErrorString(r));   \
      exit(1);    \
  }                        \
} while(0)


/*
/**
* @brief Compute the Cholesky decomposition of the normal equations and solve
*        for out_row. We only compute the upper-triangular portion of 'neqs',
*        so work with the lower-triangular portion when column-major
*        (for Fortran).
*
* @param neqs The NxN normal equations.
* @param[out] out_row The RHS of the equation. Updated in place.
* @param N The rank of the problem.

static inline void p_invert_row(
    double * const restrict neqs,
    double * const restrict out_row,
    idx_t const N)
{
  char uplo = 'L';
  int order = (int) N;
  int lda = (int) N;
  int info;
  LAPACK_DPOTRF(&uplo, &order, neqs, &lda, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRF returned %d\n", info);
  }


  int nrhs = 1;
  int ldb = (int) N;
  LAPACK_DPOTRS(&uplo, &order, &nrhs, neqs, &lda, out_row, &ldb, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRS returned %d\n", info);
  }
}



/**
* @brief Compute DSYRK: out += A^T * A, a rank-k update. Only compute
*        the upper-triangular portion.
*
* @param A The input row(s) to update with.
* @param N The length of 'A'.
* @param nvecs The number of rows in 'A'.
* @param nflush Then number of times this has been performed (this slice).
* @param[out] out The NxN matrix to update.

static inline void p_vec_oprod(
		double * const restrict A,
    idx_t const N,
    idx_t const nvecs,
    idx_t const nflush,
    double * const restrict out)
{
  char uplo = 'L';
  char trans = 'N';
  int order = (int) N;
  int k = (int) nvecs;
  int lda = (int) N;
  int ldc = (int) N;
  double alpha = 1;
  double beta = (nflush == 0) ? 0. : 1.;
  LAPACK_DSYRK(&uplo, &trans, &order, &k, &alpha, A, &lda, &beta, out, &ldc);
}


static void p_process_slice3(
    csf_sptensor * csf,
    idx_t const tile,
    idx_t const i,
    double * A,
    double * B,
    idx_t const DEFAULT_NFACTORS,
    double *  out_row,
    double * accum,
    double *  neqs,
    double *  neqs_buf,
    idx_t * const nflush)
{
  csf_sparsity const * const pt = csf->pt + tile;
  idx_t const * const restrict sptr = pt->fptr[0];
  idx_t const * const restrict fptr = pt->fptr[1];
  idx_t const * const restrict fids = pt->fids[1];
  idx_t const * const restrict inds = pt->fids[2];
  double const * const restrict vals = pt->vals;

  double * hada = neqs_buf;

  idx_t bufsize = 0;

  /* process each fiber 
  for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
    double const * const restrict av = A  + (fids[fib] * DEFAULT_NFACTORS);

    /* first entry of the fiber is used to initialize accum 
    idx_t const jjfirst  = fptr[fib];
    double const vfirst   = vals[jjfirst];
    double const * const restrict bv = B + (inds[jjfirst] * DEFAULT_NFACTORS);
    for(idx_t r=0; r < DEFAULT_NFACTORS; ++r) {
      accum[r] = vfirst * bv[r];
      hada[r] = av[r] * bv[r];
    }

    hada += DEFAULT_NFACTORS;
    if(++bufsize == ALS_BUFSIZE) {
      /* add to normal equations 
      p_vec_oprod(neqs_buf, DEFAULT_NFACTORS, bufsize, (*nflush)++, neqs);
      bufsize = 0;
      hada = neqs_buf;
    }

    /* foreach nnz in fiber 
    for(idx_t jj=fptr[fib]+1; jj < fptr[fib+1]; ++jj) {
      double const v = vals[jj];
      double const * const restrict bv = B + (inds[jj] * DEFAULT_NFACTORS);
      for(idx_t r=0; r < DEFAULT_NFACTORS; ++r) {
        accum[r] += v * bv[r];
        hada[r] = av[r] * bv[r];
      }

      hada += DEFAULT_NFACTORS;
      if(++bufsize == ALS_BUFSIZE) {
        /* add to normal equations 
        p_vec_oprod(neqs_buf, DEFAULT_NFACTORS, bufsize, (*nflush)++, neqs);
        bufsize = 0;
        hada = neqs_buf;
      }
    }

    /* accumulate into output row 
    for(idx_t r=0; r < DEFAULT_NFACTORS; ++r) {
      out_row[r] += accum[r] * av[r];
    }

  } /* foreach fiber 

  /* final flush 
  p_vec_oprod(neqs_buf, DEFAULT_NFACTORS, bufsize, (*nflush)++, neqs);
}


//private function TODO: in gpu
/**
* @brief Compute the i-ith row of the MTTKRP, form the normal equations, and
*        store the new row.*
* @param neq for inverse part
* @param out_row for mttkrp part
* @param i The row to update.
* @param reg Regularization parameter for the i-th row.

static void p_update_slice(
    sptensor_t * train,
    idx_t const i,
    double const regularization_index,
    idx_t DEFAULT_NFACTORS
    )
{
  idx_t const nmodes = train->nmodes;
  
  /* fid is the row we are actually updating 
  idx_t const fid = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
  double * const restrict out_row = model->factors[csf->dim_perm[0]] +
      (fid * DEFAULT_NFACTORS);
  double * const restrict accum = ws->thds[tid].scratch[1];
  double * const restrict neqs  = ws->thds[tid].scratch[2];

  idx_t bufsize = 0; /* how many hada vecs are in mat_accum 
  idx_t nflush = 0;  /* how many times we have flushed to add to the neqs
  double * const restrict mat_accum  = ws->thds[tid].scratch[3];

  double * hada = mat_accum;
  double * const restrict hada_accum  = ws->thds[tid].scratch[4];

  /* clear out buffers 
  for(idx_t m=0; m < nmodes; ++m) {
    for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
      accum[f + (m*DEFAULT_NFACTORS)] = 0.;
    }
    for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
      hada_accum[f + (m*DEFAULT_NFACTORS)] = 0.;
    }
  }
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    out_row[f] = 0;
  }

  /* grab factors
  double * mats[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = model->factors[csf->dim_perm[m]];
  }

  /* do MTTKRP + dsyrk
  p_process_slice(csf, 0, i, mats, DEFAULT_NFACTORS, out_row, accum, neqs, mat_accum,
      hada_accum, &nflush);

  /* add regularization to the diagonal
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    neqs[f + (f * DEFAULT_NFACTORS)] += reg;
  }

  /* solve! 
  p_invert_row(neqs, out_row, DEFAULT_NFACTORS);
}

/**
 * @brief To update the factor matrices in als

static void p_update_als(
    sptensor_t * train,
    ordi_matrix ** mats,
    double regularization_index,
    idx_t DEFAULT_NFACTORS
)
{
    
    //for (i in all i number):
    //    p_update_slice();
    
} */

// gpu global function
/**
 * @brief For computing the mttkrp in als
 * @version Now only contains the atomic operation
*/
__global__ void p_mttkrp_gpu(cissbasic_t* d_traina, 
                             ordi_matrix * d_factora, 
                             ordi_matrix * d_factorb, 
                             ordi_matrix * d_factorc, 
                             double * d_hbuffer, 
                             idx_t tilenum
                            )
{
  //get thread and block index
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  uint8_t flag;
  double * entries = d_traina -> entries;
  idx_t localtile = tileid * ((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  double __align__(256)  localtbuffer[6];
  double __align__(256)  localmbuffer[2 * DEFAULT_NFACTORS];
  
  
  
  //do the mttkrp
  if(tileid < tilenum)
  {
    //get supportive information for tiles
    idx_t f_id = (idx_t)(entries[localtile] * (-1)) ;
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1)) ;
    idx_t bitmap = (idx_t)(entries[localtile+2]);
    //if(bitmap == 0) break;
    #ifdef DEBUG
    if(tileid == 0)
    {
      printf("f_id %ld, l_id %ld, bitmap %ld\n", f_id, l_id, bitmap);
    }
    #endif
    bitmap = __brevll(bitmap);
    while((bitmap & 1) == 0) {bitmap = bitmap >> 1;}
    bitmap = bitmap >> 1;
    localtile += DEFAULT_T_TILE_WIDTH;
    #ifdef DEBUG
    if(tileid == 0)
    {
      printf("f_id %ld, l_id %ld, bitmap %ld\n", f_id, l_id, bitmap);
    }
    #endif
    //load in vectorize
    for(int m = 0; m < ((idx_t)DEFAULT_T_TILE_LENGTH) / 2; m++ )
    {
      //unroll loop and load
      //((double2*)localtbuffer)[0] = ((double2*)(entries+localtile))[0];
      //((double2*)localtbuffer)[1] = ((double2*)(entries+localtile))[1];
      //((double2*)localtbuffer)[2] = ((double2*)(entries+localtile))[2];
      localtbuffer[0] = entries[localtile];
      localtbuffer[1] = entries[localtile + 1];
      localtbuffer[2] = entries[localtile + 2];
      localtbuffer[3] = entries[localtile + 3];
      localtbuffer[4] = entries[localtile + 4];
      localtbuffer[5] = entries[localtile + 5];
      
           
      //do the mttkrp for the first
      f_id = f_id + (!(bitmap & 1));
      idx_t tmpi = d_traina->directory[f_id];
      tmpi--;
      #ifdef DEBUG
      printf("the fid is %d\n", f_id);
      #endif
      bitmap = bitmap >> 1;
      if((localtbuffer[0] == -1) && (localtbuffer[1] == -1)) break;
      for(int j = 0; j < DEFAULT_NFACTORS; j++)
      {
        double b = d_factorb->values[((idx_t)localtbuffer[0]*DEFAULT_NFACTORS - DEFAULT_NFACTORS ) + j];
        double c = d_factorc->values[((idx_t)localtbuffer[1]*DEFAULT_NFACTORS - DEFAULT_NFACTORS) + j];
        localmbuffer[j] = b * c;
        atomicAdd(&(d_factora->values[tmpi * DEFAULT_NFACTORS + j]), localmbuffer[j] * localtbuffer[2]);        
      }

      
      //if(localtbuffer[0] == -1 && localtbuffer[1] == -1) break;
      /*for(int j = 0; j < DEFAULT_NFACTORS; j++)
      {
        idx_t b = d_factorb->values[(idx_t)(localtbuffer[0]*DEFAULT_NFACTORS - DEFAULT_NFACTORS) + j];
        idx_t c = d_factorc->values[(idx_t)(localtbuffer[1]*DEFAULT_NFACTORS - DEFAULT_NFACTORS) + j];
        localmbuffer[j] = b * c;
        atomicAdd(&(d_factora->values[tmpi * DEFAULT_NFACTORS + j]), localmbuffer[j] * localtbuffer[2]);        
      }*/

      //do the mttkrp for the second
      flag = !(bitmap & 1);
      f_id = f_id + (!(bitmap & 1));
      #ifdef DEBUG
      printf("the fid is %d\n", f_id);
      #endif
      tmpi = d_traina->directory[f_id];
      tmpi--;
      bitmap = bitmap >> 1;
      if((localtbuffer[0] == -1) && (localtbuffer[1] == -1)) break;
      for(int j = 0; j < DEFAULT_NFACTORS; j++)
      {
        idx_t b = d_factorb->values[((idx_t)localtbuffer[3]*DEFAULT_NFACTORS - DEFAULT_NFACTORS) + j];
        idx_t c = d_factorc->values[((idx_t)localtbuffer[4]*DEFAULT_NFACTORS - DEFAULT_NFACTORS) + j];
        localmbuffer[DEFAULT_NFACTORS + j] = b * c;
        atomicAdd(&(d_factora->values[tmpi * DEFAULT_NFACTORS + j]), localmbuffer[DEFAULT_NFACTORS + j] * localtbuffer[5]);        
      }

      //compute the HTH for the first
      //compute the HTH for the second
      if(flag)
      {
        for(int i = 0; i < DEFAULT_NFACTORS; i++)
      {
        for(int j = 0; j <=i ; j++)
        {
          double presult1 = localmbuffer[i] * localmbuffer[j];
          double presult2 = localmbuffer[DEFAULT_NFACTORS + i] * localmbuffer[DEFAULT_NFACTORS + j];
          atomicAdd(&(d_hbuffer[(f_id - flag) * DEFAULT_NFACTORS * DEFAULT_NFACTORS + i * DEFAULT_NFACTORS + j]), presult1); 
          atomicAdd(&(d_hbuffer[f_id * DEFAULT_NFACTORS * DEFAULT_NFACTORS + i * DEFAULT_NFACTORS + j]), presult2); 
        }
      }
      }
      else
      {
        for(int i = 0; i < DEFAULT_NFACTORS; i++)
      {
        for(int j = 0; j <=i ; j++)
        {
          double presult = localmbuffer[i] * localmbuffer[j] + localmbuffer[DEFAULT_NFACTORS + i] * localmbuffer[DEFAULT_NFACTORS + j];
          atomicAdd(&(d_hbuffer[f_id * DEFAULT_NFACTORS * DEFAULT_NFACTORS + i * DEFAULT_NFACTORS + j]), presult); 
        }
      }
      }
               
      localtile += 2*DEFAULT_T_TILE_WIDTH;
    }
  }

}

/**
 * @brief For computing the mttkrp in als, only one element on one thread
 * @version Now reduce atmoic add with segment scan
*/
__global__ void p_mttkrp_gpu_as(cissbasic_t* d_traina, 
                                ordi_matrix * d_factora, 
                                ordi_matrix * d_factorb, 
                                ordi_matrix * d_factorc, 
                                double * d_hbuffer, 
                                //double * d_hthbuffer,
                                idx_t tilenum)
{
  //get block, warp and thread index
  __shared__ uint32_t warpmask[((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)];
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t warpid = tid / ((idx_t)ALS_WARPSIZE);
  idx_t laneid = tid % ((idx_t)ALS_WARPSIZE);
  idx_t tileid = bid * ((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE) + warpid;
  double * entries = d_traina -> entries;
  idx_t localtile = tileid * ((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  double __align__(256)  localtbuffer[3];
  double __align__(256)  localmbuffer[DEFAULT_NFACTORS];
  double mytmp = 0, myntmp = 0;

  //initialize the warp mask
  if(laneid == 0) warpmask[warpid] = 0xffffffff;
  if((tileid * DEFAULT_T_TILE_LENGTH + laneid) == d_traina->nnz)
  {
      //redefine the mask
      warpmask[warpid] = __brev((warpmask[warpid]<<(32-laneid)));      
  }
  
  __syncwarp();

  uint32_t mymask =  warpmask[warpid];

  #ifdef ALSAS_DEBUG
  //printf("now the mymask and mynnz id in thread %ld are %x and %ld\n", tid, mymask, (tileid * DEFAULT_T_TILE_LENGTH + laneid));
  #endif

  if((tileid < tilenum) && ((tileid * DEFAULT_T_TILE_LENGTH + laneid)<d_traina->nnz))
  {
    //initialize the information for tile and local entry
    idx_t f_id = (idx_t)(entries[localtile] * (-1)) ;
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1)) ;
    idx_t bitmap = (idx_t)(entries[localtile+2]);
    if(bitmap != 0)
    {
      bitmap = __brevll(bitmap);
      while((bitmap & 1) == 0) {bitmap = bitmap >> 1;}
      bitmap = bitmap >> 1;
      idx_t itercounter = __popcll(bitmap) - (bitmap & 1);
      #ifdef ALSAS_DEBUG
      //if(laneid == 0)
      //printf("now the itercounter is %ld\n", itercounter);
      #endif
            
      idx_t myfid = f_id + laneid - __popcll((bitmap << (63-laneid))) + 1;
      #ifdef ALSAS_DEBUG
      //printf("now the myfid in thread %ld is %ld\n", tid, myfid);
      #endif
      idx_t mybit = ((bitmap >> (laneid)) & 1);    
      idx_t mylbit = mybit;
      if(laneid == 0) 
      {
        mylbit = 0;
        mybit = 1;
      }
    
      //inter thread computation
      localtbuffer[0] = entries[localtile + (laneid + 1) * DEFAULT_T_TILE_WIDTH];
      localtbuffer[1] = entries[localtile + (laneid + 1) * DEFAULT_T_TILE_WIDTH + 1];
      localtbuffer[2] = entries[localtile + (laneid + 1) * DEFAULT_T_TILE_WIDTH + 2];

      idx_t tmpi = d_traina->directory[myfid] - 1;
      idx_t b = (idx_t)localtbuffer[0] - 1;
      idx_t c = (idx_t)localtbuffer[1] - 1;

      //for the hadamard
      #ifdef ALSAS_DEBUG
      //printf("now the myposition for hthbuffer in thread %ld is %ld\n", tid, (tileid * DEFAULT_T_TILE_LENGTH + laneid));
      #endif
      for(int m = 0; m < DEFAULT_NFACTORS; m++)
      {
        localmbuffer[m] = d_factorb->values[b * DEFAULT_NFACTORS + m] * d_factorc->values[c * DEFAULT_NFACTORS + m]; 
        //d_hthbuffer[(tileid * DEFAULT_T_TILE_LENGTH + laneid)*DEFAULT_NFACTORS + m] = localmbuffer[m];
      }
      
      __syncwarp(mymask);
      
    //reduction in hth
    //mytmp: final partial result; myntmp: messages
    
    for(int m = 0; m < DEFAULT_NFACTORS; m++)
    {
      for(int j = 0; j <=m ; j++)
      {
        mytmp = localmbuffer[m] * localmbuffer[j];
        myntmp = mybit * mytmp;
        __syncwarp(mymask);
        //now the reduction
        for(int i = 0; i < itercounter; i++)
        {
          mytmp = (__shfl_down_sync(mymask, myntmp, 1, (int)ALS_WARPSIZE)) + (!(mylbit)) * mytmp;
          myntmp = mybit * mytmp;      
          __syncwarp(mymask);              
        }
        if(!mybit)
        {
          atomicAdd(&(d_hbuffer[myfid * DEFAULT_NFACTORS * DEFAULT_NFACTORS + m * DEFAULT_NFACTORS + j]), mytmp); 
        }
        __syncwarp(mymask);
      }
    }
    __syncwarp(mymask); 

    //reduction in mttkrp
      for(int m = 0; m < DEFAULT_NFACTORS; m++)
      {
        mytmp = localmbuffer[m] * localtbuffer[2];
        myntmp = mybit * mytmp;
        __syncwarp(mymask);
        //now the reduction
        for(int i = 0; i < itercounter; i++)
        {
          mytmp = (__shfl_down_sync(mymask, myntmp, 1, (int)ALS_WARPSIZE)) + (!(mylbit)) * mytmp;
          myntmp = mybit * mytmp; 
          __syncwarp(mymask);                   
        }      
        if(!mybit)
        {
          atomicAdd(&(d_factora->values[tmpi * DEFAULT_NFACTORS + m]), mytmp);  
        }
        __syncwarp(mymask);
    }

  }
  }
  __syncthreads();
}


/**
 * @brief For update the H matrices and prepare for inversion as well as equation
 * @version Warp shuffle
**/
__global__ void p_hth_update_as(cissbasic_t * d_traina, 
                                double * d_hthbuffer, 
                                double * d_value_a, 
                                double * d_hbuffer, 
                                double ** d_hbufptr, 
                                double ** d_factptr, 
                                idx_t dlength, 
                                double regularization_index)
{
  __shared__ double blkmbuffer[((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE) * (idx_t)DEFAULT_NFACTORS];
  //get block, warp and thread index
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t warpid = tid / ((idx_t)ALS_WARPSIZE);
  idx_t laneid = tid % ((idx_t)ALS_WARPSIZE);
  idx_t tileid = bid * ((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE) + warpid;
  double __align__(256)  localhthbuffer[DEFAULT_NFACTORS]={0};
    
  if(tileid < dlength && laneid < DEFAULT_NFACTORS)
  {
    
    idx_t dcounter = d_traina->dcounter[tileid+1] - d_traina->dcounter[tileid];
    #ifdef ALSAS_DEBUG
    if(laneid == 0) printf("my dcounter is %ld\n and my tileid is %ld\n", dcounter, tileid);
    #endif
    idx_t basicposition = d_traina->dcounter[tileid];
    idx_t basicsposition = warpid * DEFAULT_NFACTORS;
    for(idx_t i = 0; i < dcounter; i++)
    {
      double localvalue = d_hthbuffer[(basicposition + i) * DEFAULT_NFACTORS + laneid];
      blkmbuffer[basicsposition + laneid] = localvalue;
      __syncwarp();
      for(idx_t j = 0; j < DEFAULT_NFACTORS; j++)
      {
        localhthbuffer[j] += localvalue * blkmbuffer[basicsposition + j];
      }
    }
    __syncwarp();
    localhthbuffer[laneid] += regularization_index;
    for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
    {
      d_hbuffer[tileid * DEFAULT_NFACTORS * DEFAULT_NFACTORS + laneid * DEFAULT_NFACTORS + i] = localhthbuffer[i];
    }
    __syncwarp();
    //prepare for ptrs
    if(laneid == 0) 
    {
      idx_t fid = d_traina->directory[tileid] - 1;
      d_factptr[tileid] = d_value_a + fid * DEFAULT_NFACTORS;
      d_hbufptr[tileid] = d_hbuffer + tileid * DEFAULT_NFACTORS * DEFAULT_NFACTORS;
    }
  }
  __syncwarp();

}
                

/**
 * @brief Compute the inverse and finish the final update
 * @version Now only with coarse grain
*/
 __global__ void p_update_als_gpu(cissbasic_t * d_traina,
                                  ordi_matrix * d_factora, 
                                  double * d_hbuffer,
                                  idx_t dlength,
                                  double regularization_index
                                )
{
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;  
  idx_t basicposition = tileid * DEFAULT_NFACTORS * DEFAULT_NFACTORS;
  double lv[DEFAULT_NFACTORS * DEFAULT_NFACTORS]={0};
  
  if(tileid < dlength)
  {
    //compute the inverse
    idx_t tmpi = d_traina->directory[tileid];
    tmpi--;
    double *av = d_hbuffer + basicposition;
        
    idx_t i = 0;
    idx_t j = 0;
    idx_t k = 0;
    for (i = 0; i < DEFAULT_NFACTORS; ++i) 
    {
      for (j = 0; j <= i; ++j) 
      {
        double inner = 0;
        for (k = 0; k < j; ++k) 
        {
          inner += lv[k+(i*DEFAULT_NFACTORS)] * lv[k+(j*DEFAULT_NFACTORS)];
        }

        if(i == j) 
        {
          lv[j+(i*DEFAULT_NFACTORS)] = sqrt(av[i+(i*DEFAULT_NFACTORS)] - inner + regularization_index);
        } 
        else 
        {  
          lv[j+(i*DEFAULT_NFACTORS)] = 1.0 / lv[j+(j*DEFAULT_NFACTORS)] * (av[j+(i*DEFAULT_NFACTORS)] - inner);
        }
      }
    }  
   
    for(i = 0; i< DEFAULT_NFACTORS * DEFAULT_NFACTORS; i++)
    {
      av[i] = 0;
    }
    idx_t n = 0;
    for(n=0; n<DEFAULT_NFACTORS; n++) //get identity matrix
    {
      av[n+(n*DEFAULT_NFACTORS)] = 1.0;
    }
   
    //forward solve
    i = 1; //define counters outside the loop
    j = 0;
    idx_t f = 0;
    for(j=0; j < DEFAULT_NFACTORS; ++j) 
    {
      av[j] /= lv[0];
    }
 
    for(i=1; i < DEFAULT_NFACTORS; ++i) 
    {
    /* X(i,f) = B(i,f) - \sum_{j=0}^{i-1} L(i,j)X(i,j) */   
     for(j=0; j < i; ++j) 
     {
       for(f=0; f < DEFAULT_NFACTORS; ++f) 
       {
         av[f+(i*DEFAULT_NFACTORS)] -= lv[j+(i*DEFAULT_NFACTORS)] * av[f+(j*DEFAULT_NFACTORS)];
       }
     }
     for(f=0; f <DEFAULT_NFACTORS; ++f) 
     {
       av[f+(i*DEFAULT_NFACTORS)] /= lv[i+(i*DEFAULT_NFACTORS)];
     }
   }
 
  for(i=0; i < DEFAULT_NFACTORS; ++i) 
  {
    for(j=i+1; j < DEFAULT_NFACTORS; ++j) 
    {
      lv[j+(i*DEFAULT_NFACTORS)] = lv[i+(j*DEFAULT_NFACTORS)];
      lv[i+(j*DEFAULT_NFACTORS)] = 0.0;
    }
  }

  //backsolve
  f = 0;  //set counters
  j = 0;
  idx_t row = 2;
  
  /* last row of X is easy */
  for(f=0; f < DEFAULT_NFACTORS; ++f) {
    i = DEFAULT_NFACTORS - 1;
    av[f+(i*DEFAULT_NFACTORS)] /= lv[i+(i*DEFAULT_NFACTORS)];
  }

  /* now do backward substitution */
  for(row=2; row <= DEFAULT_NFACTORS; ++row) 
  {    
    i = DEFAULT_NFACTORS - row;
    /* X(i,f) = B(i,f) - \sum_{j=0}^{i-1} R(i,j)X(i,j) */
    for( j=i+1; j < DEFAULT_NFACTORS; ++j) 
    {
      for( f=0; f < DEFAULT_NFACTORS; ++f) 
      {
        av[f+(i*DEFAULT_NFACTORS)] -= lv[j+(i*DEFAULT_NFACTORS)] * av[f+( j * DEFAULT_NFACTORS )];
      }
    }
    for(f=0; f < DEFAULT_NFACTORS; ++f) 
    {
      av[f+(i*DEFAULT_NFACTORS)] /= lv[i+(i*DEFAULT_NFACTORS)];
    }
  }

  //now do the final update
  double * mvals = d_factora->values + tmpi * DEFAULT_NFACTORS;
  for(i = 0; i < DEFAULT_NFACTORS; i++)
  {
    lv[i] = 0;
    for(j = 0; j < DEFAULT_NFACTORS; j++)
    {
      lv[i] += mvals[j] * av[i * DEFAULT_NFACTORS + j];
    }
  }

  //the final transmission
  for(i = 0; i < DEFAULT_NFACTORS/2; i++)
  {
    ((double2*)mvals)[i] = ((double2*)lv)[i]; 
  }
   
  }

}

/**
 * @brief Update the matrice
 * @version Now only with coarse grain
*/
__global__ void p_update_matrice(cissbasic_t * d_traina,
                            double * d_value_a, 
                            double * d_hbuffer,
                            double ** d_hbufptr, 
                            double ** d_factptr,
                            idx_t  dlength,
                            double regularization_index)
{
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;  
  idx_t basicposition = tileid * DEFAULT_NFACTORS * DEFAULT_NFACTORS;
  
  
  if(tileid < dlength)
  {
    idx_t tmpi = d_traina->directory[tileid] - 1;
    for(idx_t f = 0; f < DEFAULT_NFACTORS; f++)
    {
      d_hbuffer[basicposition + f*DEFAULT_NFACTORS + f] += regularization_index;  
    }
    d_hbufptr[tileid] = d_hbuffer + basicposition;
    d_factptr[tileid] = d_value_a + tmpi * DEFAULT_NFACTORS;
  }
}

void p_cholecheck(double * d_factora, 
                  double * d_hbuffer, 
                  double ** d_hbufptr, 
                  double ** d_factptr, 
                  idx_t dlength)
{

}


extern "C"{

/**
 * @brief The main function for tensor completion in als
 * @param train The tensor for generating factor matrices
 * @param validation The tensor for validation(RMSE)
 * @param test The tensor for testing the quality
 * @param regularization_index Lambda
*/
void tc_als(sptensor_t * traina, 
            sptensor_t * trainb,
            sptensor_t * trainc,
            sptensor_t * validation,
            sptensor_t * test,
            ordi_matrix ** mats,
            ordi_matrix ** best_mats, 
            idx_t algorithm_index,
            double regularization_index, 
            double * best_rmse, 
            double * tolerance, 
            idx_t * nbadepochs, 
            idx_t * bestepochs, 
            idx_t * max_badepochs)
{
    idx_t const nmodes = traina->nmodes;
    #ifdef CISS_DEBUG
    printf("enter the als\n");
    #endif
    
    //initialize the devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int n;
    #ifdef DMMODIFY
    deviceCount = DEFAULT_GPUNUM;
    #endif
    //print the GPU status
    for(n = 0; n < deviceCount; n++)
    {
      cudaDeviceProp dprop;
      cudaGetDeviceProperties(&dprop, n);
      printf("   %d: %s\n", n, dprop.name);
    }
    omp_set_num_threads(deviceCount);
    //prepare the tensor in TB-COO
    ciss_t * h_cissta = ciss_alloc(traina, 1, deviceCount);
    ciss_t * h_cisstb = ciss_alloc(trainb, 2, deviceCount);
    ciss_t * h_cisstc = ciss_alloc(trainc, 3, deviceCount);
    #ifdef MCISS_DEBUG
    fprintf(stdout, "the new tensors for mode 0\n");
    cissbasic_display(h_cissta->cissunits[0]);
    cissbasic_display(h_cissta->cissunits[1]);
    #endif
    struct timeval start;
    struct timeval end;
    idx_t diff;
    
      
    
    cissbasic_t ** d_traina = (cissbasic_t**)malloc(deviceCount * sizeof(cissbasic_t*));
    cissbasic_t ** d_trainb = (cissbasic_t**)malloc(deviceCount * sizeof(cissbasic_t*)); 
    cissbasic_t ** d_trainc = (cissbasic_t**)malloc(deviceCount * sizeof(cissbasic_t*));
    idx_t ** d_directory_a = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_directory_b = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_directory_c = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_counter_a = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_counter_b = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_counter_c = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_dims_a = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_dims_b = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_dims_c = (idx_t**)malloc(deviceCount * sizeof(idx_t*)); 
    double ** d_entries_a = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_entries_b = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_entries_c = (double**)malloc(deviceCount * sizeof(double*)); 
    double ** d_hbuffer = (double**)malloc(deviceCount * sizeof(double*));
    //double ** d_hthbuffer = (double**)malloc(deviceCount * sizeof(double*));
    int ** d_infoarray = (int**)malloc(deviceCount * sizeof(int*));
    double *** d_hbufptr = (double***)malloc(deviceCount * sizeof(double**));
    double *** d_factptr = (double***)malloc(deviceCount * sizeof(double**));

    ordi_matrix ** d_factora = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    ordi_matrix ** d_factorb = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    ordi_matrix ** d_factorc = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    double ** d_value_a = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_b = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_c = (double**)malloc(deviceCount * sizeof(double*));

    idx_t * maxdlength = (idx_t*)malloc(deviceCount * sizeof(idx_t));
    idx_t * maxnnz = (idx_t*)malloc(deviceCount * sizeof(idx_t));
    #ifdef DMMODIFY
    cusolverDnHandle_t handles[DEFAULT_GPUNUM];
    ncclComm_t comms[DEFAULT_GPUNUM];
    //initialize the unique id for nccl
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    #else
    cusolverDnHandle_t handle0, handle1;
    cudaSetDevice(0);
    HANDLE_SOLVERERR(cusolverDnCreate((&handle0)));
    cudaSetDevice(1);
    HANDLE_SOLVERERR(cusolverDnCreate((&handle1)));
    #endif

    
  #pragma omp parallel
  {
    //prepare the threads
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();

    //set gpus
    int gpu_id = -1;
    cudaSetDevice(cpu_thread_id % deviceCount);  // "% num_gpus" allows more CPU threads than GPU devices
    cudaGetDevice(&gpu_id);
    idx_t * d_itemp1, *d_itemp2, *d_itemp3;
    double * d_ftemp;
    //initialize the cusolver
    #ifdef DMMODIFY
    HANDLE_SOLVERERR(cusolverDnCreate(&(handles[gpu_id])));
    HANDLE_NCCLERR(ncclCommInitRank(&(comms[gpu_id]), deviceCount, id, gpu_id));
    #endif

    //malloc and copy the tensors + matrices to gpu
    
    cissbasic_t * h_traina = h_cissta->cissunits[gpu_id];
    cissbasic_t * h_trainb = h_cisstb->cissunits[gpu_id];
    cissbasic_t * h_trainc = h_cisstc->cissunits[gpu_id];
    //copy tensor for mode-1
    HANDLE_ERROR(cudaMalloc((void**)&(d_traina[gpu_id]), sizeof(cissbasic_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_directory_a[gpu_id]), h_traina->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_counter_a[gpu_id]), (h_traina->dlength + 1) * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_entries_a[gpu_id]), h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_dims_a[gpu_id]), nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_counter_a[gpu_id], h_traina->dcounter, (h_traina->dlength + 1)*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_directory_a[gpu_id], h_traina->directory, h_traina->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_a[gpu_id], h_traina->entries, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_a[gpu_id], h_traina->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_traina->directory;
    d_itemp2 = h_traina->dims;
    d_itemp3 = h_traina->dcounter;
    d_ftemp = h_traina->entries;
    h_traina->directory = d_directory_a[gpu_id];
    h_traina->dims = d_dims_a[gpu_id];
    h_traina->entries = d_entries_a[gpu_id];
    h_traina->dcounter = d_counter_a[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_traina[gpu_id], h_traina, sizeof(cissbasic_t), cudaMemcpyHostToDevice));
    h_traina->directory = d_itemp1;
    h_traina->dims = d_itemp2;
    h_traina->entries = d_ftemp;
    h_traina->dcounter = d_itemp3;
    //copy tensor for mode-2
    HANDLE_ERROR(cudaMalloc((void**)&(d_trainb[gpu_id]), sizeof(cissbasic_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_directory_b[gpu_id]), h_trainb->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_counter_b[gpu_id]), (h_trainb->dlength + 1) * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_entries_b[gpu_id]), h_trainb->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_dims_b[gpu_id]), nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_b[gpu_id], h_trainb->directory, h_trainb->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_counter_b[gpu_id], h_trainb->dcounter, (h_trainb->dlength + 1)*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_b[gpu_id], h_trainb->entries, h_trainb->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_b[gpu_id], h_trainb->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_trainb->directory;
    d_itemp2 = h_trainb->dims;
    d_itemp3 = h_trainb->dcounter;
    d_ftemp = h_trainb->entries;
    h_trainb->directory = d_directory_b[gpu_id];
    h_trainb->dims = d_dims_b[gpu_id];
    h_trainb->entries = d_entries_b[gpu_id];
    h_trainb->dcounter = d_counter_b[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_trainb[gpu_id], h_trainb, sizeof(cissbasic_t), cudaMemcpyHostToDevice));
    h_trainb->directory = d_itemp1;
    h_trainb->dims = d_itemp2;
    h_trainb->entries = d_ftemp;
    h_trainb->dcounter = d_itemp3;
    //copy tensor for mode-3
    HANDLE_ERROR(cudaMalloc((void**)&(d_trainc[gpu_id]), sizeof(cissbasic_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_directory_c[gpu_id]), h_trainc->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_counter_c[gpu_id]), (h_trainc->dlength + 1) * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_entries_c[gpu_id]), h_trainc->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_dims_c[gpu_id]), nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_c[gpu_id], h_trainc->directory, h_trainc->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_counter_c[gpu_id], h_trainc->dcounter, (h_trainc->dlength + 1)*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_c[gpu_id], h_trainc->entries, h_trainc->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_c[gpu_id], h_trainc->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_trainc->directory;
    d_itemp2 = h_trainc->dims;
    d_ftemp = h_trainc->entries;
    d_itemp3 = h_trainc->dcounter;
    h_trainc->directory = d_directory_c[gpu_id];
    h_trainc->dims = d_dims_c[gpu_id];
    h_trainc->entries = d_entries_c[gpu_id];
    h_trainc->dcounter = d_counter_c[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_trainc[gpu_id], h_trainc, sizeof(cissbasic_t), cudaMemcpyHostToDevice));
    h_trainc->directory = d_itemp1;
    h_trainc->dims = d_itemp2;
    h_trainc->entries = d_ftemp;
    h_trainc->dcounter = d_itemp3;

    //buffer for HTH
    maxdlength[gpu_id] = SS_MAX(SS_MAX(h_traina->dlength, h_trainb->dlength),h_trainc->dlength);
    maxnnz[gpu_id] = SS_MAX(SS_MAX(h_traina->nnz, h_trainb->nnz),h_trainc->nnz);

    #ifdef ALSAS_DEBUG
    fprintf(stdout, "now in thread %d the cpu maxnnz is %ld\n", cpu_thread_id,maxnnz[gpu_id]); 
    #endif   
      
    HANDLE_ERROR(cudaMalloc((void**)&(d_hbuffer[gpu_id]), DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength[gpu_id] * sizeof(double)));
    //HANDLE_ERROR(cudaMalloc((void**)&(d_hthbuffer[gpu_id]), DEFAULT_NFACTORS * maxnnz[gpu_id] * sizeof(double)));
    
    //HANDLE_ERROR(cudaMalloc((void**)&d_invbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)));
    //buffer for inversion
    HANDLE_ERROR(cudaMalloc((void**)&(d_hbufptr[gpu_id]), maxdlength[gpu_id] * sizeof(double*)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_factptr[gpu_id]), maxdlength[gpu_id] * sizeof(double*)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_infoarray[gpu_id]), maxdlength[gpu_id] * sizeof(int)));  

    
    HANDLE_ERROR(cudaMalloc((void**)&(d_factora[gpu_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_a[gpu_id]), mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_a[gpu_id], mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    #pragma omp critical
    {
      d_ftemp = mats[0]->values;
    mats[0]->values = d_value_a[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_factora[gpu_id], mats[0], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[0]->values = d_ftemp;
    }
    #pragma omp barrier

    HANDLE_ERROR(cudaMalloc((void**)&(d_factorb[gpu_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_b[gpu_id]), mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_b[gpu_id], mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    #pragma omp critical
    {
      d_ftemp = mats[1]->values;
      mats[1]->values = d_value_b[gpu_id];
      HANDLE_ERROR(cudaMemcpy(d_factorb[gpu_id], mats[1], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
      mats[1]->values = d_ftemp;
    }
    #pragma omp barrier
    HANDLE_ERROR(cudaMalloc((void**)&(d_factorc[gpu_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_c[gpu_id]), mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_c[gpu_id], mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    #pragma omp critical
    {
      d_ftemp = mats[2]->values;
      mats[2]->values = d_value_c[gpu_id];
      HANDLE_ERROR(cudaMemcpy(d_factorc[gpu_id], mats[2], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
      mats[2]->values = d_ftemp;
    }
  }
    
    #ifdef CUDA_LOSS //to be done
    sptensor_gpu_t * d_test, * d_validate;    
    #else
    double loss = tc_loss_sq(traina, mats, algorithm_index);
    double frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    #endif
    tc_converge(traina, validation, mats, best_mats, algorithm_index, loss, frobsq, 0, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);

    //step into the kernel
    
       
        
    idx_t mode_i, mode_n, m;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    for(idx_t e=1; e < DEFAULT_MAX_ITERATE+1; ++e) {
      
      gettimeofday(&start,NULL);
      //can set random variables  
      srand(time(0));
      mode_i = rand()%3;
      #ifdef ALSAS_DEBUG
      mode_i = 0;
      fprintf(stdout, "now the mode_i is %d\n", mode_i);
      #endif
      for(m=0; m < 3; m++) {
        #pragma omp parallel
        {  
          unsigned int cpu_thread_id = omp_get_thread_num();
          
          cudaSetDevice(cpu_thread_id % deviceCount);  // "% num_gpus" allows more CPU threads than GPU devices
          #ifdef DMMODIFY
          cusolverDnHandle_t handle = handles[cpu_thread_id];
          cudaStream_t s;
          HANDLE_ERROR(cudaStreamCreate(&s));
          #else
          cusolverDnHandle_t handle;
          if(!cpu_thread_id) handle = handle1;
          else handle = handle0;
          #endif
          cissbasic_t * h_traina = h_cissta->cissunits[cpu_thread_id];
          cissbasic_t * h_trainb = h_cisstb->cissunits[cpu_thread_id];
          cissbasic_t * h_trainc = h_cisstc->cissunits[cpu_thread_id];
          
          idx_t mymode_n = (mode_i + m)%3;
          idx_t blocknum_u, blocknum_h, nnz, tilenum, blocknum_m;

                 
          HANDLE_ERROR(cudaMemset(d_hbuffer[cpu_thread_id], 0, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength[cpu_thread_id] * sizeof(double)));
          //HANDLE_ERROR(cudaMemcpy(d_invbuffer, h_invbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)),cudaMemcpyHostToDevice);
          switch (mymode_n)
          {
            case 0:
              {  
                nnz = h_traina->nnz;
                tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
                blocknum_m = tilenum/(((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)) + 1;
                #ifdef ALSAS_DEBUG
                fprintf(stdout, "now in thread %d, nnz is %d, blocknum_m is %d, tilenum is %d\n", cpu_thread_id, nnz, blocknum_m, tilenum);    
                #endif           
                HANDLE_ERROR(cudaMemset(d_value_a[cpu_thread_id], 0, mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
                blocknum_u = h_traina->dlength / DEFAULT_BLOCKSIZE + 1;
                blocknum_h = h_traina->dlength / (((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)) + 1;
                p_mttkrp_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_traina[cpu_thread_id], d_factora[cpu_thread_id], d_factorb[cpu_thread_id], d_factorc[cpu_thread_id], d_hbuffer[cpu_thread_id], tilenum);
                HANDLE_ERROR(cudaDeviceSynchronize());
                #ifdef ALSAS_DEBUG
                fprintf(stdout, "now in thread %d ends mttkrp\n", cpu_thread_id);    
                fprintf(stdout, "now in thread %d the blocknum for hth update is %ld and the dlength is %ld\n", cpu_thread_id, blocknum_h, h_traina->dlength);
                #endif
                //p_hth_update_as<<<blocknum_h,DEFAULT_BLOCKSIZE,0>>>(d_traina[cpu_thread_id], d_hthbuffer[cpu_thread_id], d_value_a[cpu_thread_id], d_hbuffer[cpu_thread_id], d_hbufptr[cpu_thread_id], d_factptr[cpu_thread_id], h_traina->dlength, regularization_index);                               
                p_update_matrice<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(d_traina[cpu_thread_id], d_value_a[cpu_thread_id], d_hbuffer[cpu_thread_id], d_hbufptr[cpu_thread_id], d_factptr[cpu_thread_id], h_traina->dlength, regularization_index); 
                HANDLE_ERROR(cudaDeviceSynchronize());
                #ifdef ALS_DEBUG
                p_cholecheck(d_value_a[cpu_thread_id], d_hbuffer[cpu_thread_id], d_hbufptr[cpu_thread_id], d_factptr[cpu_thread_id], h_traina->dlength);
                #endif
                HANDLE_SOLVERERR(cusolverDnDpotrfBatched(handle, uplo, DEFAULT_NFACTORS, d_hbufptr[cpu_thread_id], DEFAULT_NFACTORS, d_infoarray[cpu_thread_id], (int)h_traina->dlength));
                HANDLE_SOLVERERR(cusolverDnDpotrsBatched(handle, uplo, DEFAULT_NFACTORS, 1, d_hbufptr[cpu_thread_id], DEFAULT_NFACTORS, d_factptr[cpu_thread_id], DEFAULT_NFACTORS, d_infoarray[cpu_thread_id], (int)h_traina->dlength)); 
                HANDLE_ERROR(cudaDeviceSynchronize());

                

                #pragma omp barrier
                //update the final results
                #ifdef DMMODIFY
                HANDLE_NCCLERR(ncclAllReduce((const void*)d_value_a[cpu_thread_id], (void*)d_value_a[cpu_thread_id], mats[0]->I * DEFAULT_NFACTORS, ncclDouble, ncclSum,
        comms[cpu_thread_id], s));
                HANDLE_ERROR(cudaStreamSynchronize(s));
                #else
                HANDLE_ERROR(cudaMemcpy(mats[0]->values + (h_cissta->d_ref[cpu_thread_id] -1) * DEFAULT_NFACTORS, d_value_a[cpu_thread_id] + (h_cissta->d_ref[cpu_thread_id] -1) * DEFAULT_NFACTORS, (h_cissta->d_ref[cpu_thread_id + 1] - h_cissta->d_ref[cpu_thread_id]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaDeviceSynchronize());
                HANDLE_ERROR(cudaMemcpy(d_value_a[cpu_thread_id] + (h_cissta->d_ref[(cpu_thread_id + 1)% deviceCount] - 1) * DEFAULT_NFACTORS, mats[0]->values + (h_cissta->d_ref[(cpu_thread_id + 1) % deviceCount] -1 ) * DEFAULT_NFACTORS,  (h_cissta->d_ref[(cpu_thread_id + 1) % deviceCount + 1] - h_cissta->d_ref[(cpu_thread_id + 1) % deviceCount]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
                HANDLE_ERROR(cudaDeviceSynchronize());
                #endif
                                
                break;
              }
            case 1:
              {  
                           
                nnz = h_trainb->nnz;
                tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
                blocknum_m = tilenum/(((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)) + 1;
                HANDLE_ERROR(cudaMemset(d_value_b[cpu_thread_id], 0, mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
                blocknum_u = h_trainb->dlength / DEFAULT_BLOCKSIZE + 1;
                blocknum_h = h_trainb->dlength / (((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)) + 1;
                p_mttkrp_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainb[cpu_thread_id], d_factorb[cpu_thread_id], d_factorc[cpu_thread_id], d_factora[cpu_thread_id], d_hbuffer[cpu_thread_id], tilenum);
                HANDLE_ERROR(cudaDeviceSynchronize());
                //p_hth_update_as<<<blocknum_h,DEFAULT_BLOCKSIZE,0>>>(d_trainb[cpu_thread_id], d_hthbuffer[cpu_thread_id], d_value_b[cpu_thread_id], d_hbuffer[cpu_thread_id], d_hbufptr[cpu_thread_id], d_factptr[cpu_thread_id], h_trainb->dlength, regularization_index);
                HANDLE_ERROR(cudaDeviceSynchronize());
                
                p_update_matrice<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(d_trainb[cpu_thread_id], d_value_b[cpu_thread_id], d_hbuffer[cpu_thread_id], d_hbufptr[cpu_thread_id], d_factptr[cpu_thread_id], h_trainb->dlength, regularization_index); 
                HANDLE_ERROR(cudaDeviceSynchronize());
                HANDLE_SOLVERERR(cusolverDnDpotrfBatched(handle, uplo, DEFAULT_NFACTORS, d_hbufptr[cpu_thread_id], DEFAULT_NFACTORS, d_infoarray[cpu_thread_id], (int)h_trainb->dlength));
                HANDLE_SOLVERERR(cusolverDnDpotrsBatched(handle, uplo, DEFAULT_NFACTORS, 1, d_hbufptr[cpu_thread_id], DEFAULT_NFACTORS, d_factptr[cpu_thread_id],DEFAULT_NFACTORS, d_infoarray[cpu_thread_id], (int)h_trainb->dlength)); 
                HANDLE_ERROR(cudaDeviceSynchronize());

                

                #pragma omp barrier
                //update the final results
                #ifdef DMMODIFY
                HANDLE_NCCLERR(ncclAllReduce((const void*)d_value_b[cpu_thread_id], (void*)d_value_b[cpu_thread_id], mats[1]->I * DEFAULT_NFACTORS, ncclDouble, ncclSum,
                comms[cpu_thread_id], s));
                HANDLE_ERROR(cudaStreamSynchronize(s));
                #else
                HANDLE_ERROR(cudaMemcpy(mats[1]->values + (h_cisstb->d_ref[cpu_thread_id] - 1) * DEFAULT_NFACTORS, d_value_b[cpu_thread_id] + (h_cisstb->d_ref[cpu_thread_id] - 1)* DEFAULT_NFACTORS, (h_cisstb->d_ref[cpu_thread_id + 1] - h_cisstb->d_ref[cpu_thread_id]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaDeviceSynchronize());
                HANDLE_ERROR(cudaMemcpy(d_value_b[cpu_thread_id] + (h_cisstb->d_ref[(cpu_thread_id + 1)% deviceCount] - 1)* DEFAULT_NFACTORS, mats[1]->values + (h_cisstb->d_ref[(cpu_thread_id + 1) % deviceCount] - 1)* DEFAULT_NFACTORS,  (h_cisstb->d_ref[(cpu_thread_id + 1) % deviceCount + 1] - h_cisstb->d_ref[(cpu_thread_id + 1) % deviceCount]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
                HANDLE_ERROR(cudaDeviceSynchronize());
                #endif

                break;
              }
            default:
              {
                
                nnz = h_trainc->nnz;
                tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
                blocknum_m = tilenum/(((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)) + 1;
                HANDLE_ERROR(cudaMemset(d_value_c[cpu_thread_id], 0, mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
                blocknum_u = h_trainc->dlength / DEFAULT_BLOCKSIZE + 1;
                blocknum_h = h_trainc->dlength / (((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)) + 1;
                p_mttkrp_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainc[cpu_thread_id], d_factorc[cpu_thread_id], d_factora[cpu_thread_id], d_factorb[cpu_thread_id], d_hbuffer[cpu_thread_id], tilenum);
                HANDLE_ERROR(cudaDeviceSynchronize());

                //p_hth_update_as<<<blocknum_h,DEFAULT_BLOCKSIZE,0>>>(d_trainc[cpu_thread_id], d_hthbuffer[cpu_thread_id], d_value_c[cpu_thread_id], d_hbuffer[cpu_thread_id], d_hbufptr[cpu_thread_id], d_factptr[cpu_thread_id], h_trainc->dlength, regularization_index);
                                
                p_update_matrice<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(d_trainc[cpu_thread_id], d_value_c[cpu_thread_id], d_hbuffer[cpu_thread_id], d_hbufptr[cpu_thread_id], d_factptr[cpu_thread_id], h_trainc->dlength, regularization_index); 
                HANDLE_ERROR(cudaDeviceSynchronize());
                HANDLE_SOLVERERR(cusolverDnDpotrfBatched(handle, uplo, DEFAULT_NFACTORS, d_hbufptr[cpu_thread_id], DEFAULT_NFACTORS, d_infoarray[cpu_thread_id], (int)h_trainc->dlength));
                HANDLE_SOLVERERR(cusolverDnDpotrsBatched(handle, uplo, DEFAULT_NFACTORS, 1, d_hbufptr[cpu_thread_id], DEFAULT_NFACTORS, d_factptr[cpu_thread_id],DEFAULT_NFACTORS, d_infoarray[cpu_thread_id], (int)h_trainc->dlength));
                HANDLE_ERROR(cudaDeviceSynchronize()); 

                
                
                #pragma omp barrier
                //update the final results
                #ifdef DMMODIFY
                HANDLE_NCCLERR(ncclAllReduce((const void*)d_value_c[cpu_thread_id], (void*)d_value_c[cpu_thread_id], mats[2]->I * DEFAULT_NFACTORS, ncclDouble, ncclSum,
                comms[cpu_thread_id], s));
                HANDLE_ERROR(cudaStreamSynchronize(s));
                #else
                HANDLE_ERROR(cudaMemcpy(mats[2]->values + (h_cisstc->d_ref[cpu_thread_id] -1 ) * DEFAULT_NFACTORS, d_value_c[cpu_thread_id] + (h_cisstc->d_ref[cpu_thread_id] - 1) * DEFAULT_NFACTORS, (h_cisstc->d_ref[cpu_thread_id + 1] - h_cisstc->d_ref[cpu_thread_id]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                HANDLE_ERROR(cudaDeviceSynchronize());
                HANDLE_ERROR(cudaMemcpy(d_value_c[cpu_thread_id] + (h_cisstc->d_ref[(cpu_thread_id + 1)% deviceCount] -1) * DEFAULT_NFACTORS, mats[2]->values + (h_cisstc->d_ref[(cpu_thread_id + 1) % deviceCount] -1)* DEFAULT_NFACTORS,  (h_cisstc->d_ref[(cpu_thread_id + 1) % deviceCount + 1] - h_cisstc->d_ref[(cpu_thread_id + 1) % deviceCount]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
                HANDLE_ERROR(cudaDeviceSynchronize());
                #endif

                break;
              }
          //p_update_als(train, mats, m, DEFAULT_NFACTORS, regularization_index);
          
          }
          #ifdef DMMODIFY
          cudaStreamDestroy(s);
          #endif
        }
        }
        //copy the results to CPU
        #ifdef DMMODIFY
        cudaSetDevice(0);
        HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a[0], mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_b[0], mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_c[0], mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaDeviceSynchronize()); 
        #endif
        
        #ifdef DEBUG
        matrix_display(mats[0]);
        matrix_display(mats[1]);
        matrix_display(mats[2]);
        #endif
        gettimeofday(&end,NULL);
        diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
        printf("this time cost %ld\n",diff);
     

    /* compute new obj value, print stats, and exit if converged */
    loss = tc_loss_sq(traina, mats, algorithm_index);
    frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    if(tc_converge(traina, validation, mats, best_mats, algorithm_index, loss, frobsq, e, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs)) {
      break;
    }
    

  } /* foreach iteration */
 
  #ifdef DMMODIFY
  
  #else
  cudaSetDevice(0);
  HANDLE_SOLVERERR(cusolverDnDestroy(handle0));
  cudaSetDevice(1);
  HANDLE_SOLVERERR(cusolverDnDestroy(handle1));
  #endif

  #pragma omp parallel
  {
    unsigned int cpu_thread_id = omp_get_thread_num();

    cudaSetDevice(cpu_thread_id % deviceCount); 
    //end the cusolver
    //HANDLE_SOLVERERR(cusolverDnDestroy(handle));
    //free the cudabuffer
    #ifdef DMMODIFY
    HANDLE_NCCLERR(ncclCommDestroy(comms[cpu_thread_id]));
    HANDLE_SOLVERERR(cusolverDnDestroy(handles[cpu_thread_id]));
    #endif
    cudaFree(d_counter_a[cpu_thread_id]);
    cudaFree(d_directory_a[cpu_thread_id]);
    cudaFree(d_dims_a[cpu_thread_id]);
    cudaFree(d_entries_a[cpu_thread_id]);
    cudaFree(d_counter_b[cpu_thread_id]);
    cudaFree(d_directory_b[cpu_thread_id]);
    cudaFree(d_dims_b[cpu_thread_id]);
    cudaFree(d_entries_b[cpu_thread_id]);
    cudaFree(d_counter_c[cpu_thread_id]);
    cudaFree(d_directory_c[cpu_thread_id]);
    cudaFree(d_dims_c[cpu_thread_id]);
    cudaFree(d_entries_c[cpu_thread_id]);
    cudaFree(d_hbuffer[cpu_thread_id]);
    cudaFree(d_hbufptr[cpu_thread_id]);
    //cudaFree(d_hthbuffer[cpu_thread_id]);
    cudaFree(d_factptr[cpu_thread_id]);
    cudaFree(d_infoarray[cpu_thread_id]);
    cudaFree(d_value_a[cpu_thread_id]);
    cudaFree(d_value_b[cpu_thread_id]);
    cudaFree(d_value_c[cpu_thread_id]);
    cudaFree(d_traina[cpu_thread_id]);
    cudaFree(d_trainb[cpu_thread_id]);
    cudaFree(d_trainc[cpu_thread_id]);
    cudaFree(d_factora[cpu_thread_id]);
    cudaFree(d_factorb[cpu_thread_id]);
    cudaFree(d_factorc[cpu_thread_id]);
    //cudaFree(d_hthbuffer[cpu_thread_id]);
    
    cudaDeviceReset();
}
    ciss_free(h_cissta, deviceCount);
    ciss_free(h_cisstb, deviceCount);
    ciss_free(h_cisstc, deviceCount);
    free(d_traina); 
    free(d_trainb); 
    free(d_trainc); 
    free(d_directory_a); 
    free(d_directory_b); 
    free(d_directory_c); 
    free(d_counter_a); 
    free(d_counter_b); 
    free(d_counter_c);
    free(d_dims_a); 
    free(d_dims_b); 
    free(d_dims_c);
    free(d_entries_a); 
    free(d_entries_b); 
    free(d_entries_c); 
    free(d_hbuffer); 
    //free(d_hthbuffer); 
    free(d_hbufptr); 
    free(d_infoarray); 
    free(d_factptr); 
    //free(handle);

    free(d_factora); 
    free(d_factorb); 
    free(d_factorc); 
    free(d_value_a); 
    free(d_value_b); 
    free(d_value_c); 
    free(maxdlength);
    free(maxnnz);
}


}
