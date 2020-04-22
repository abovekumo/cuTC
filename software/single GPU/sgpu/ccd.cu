extern "C"
{
#include "completion.h"
#include "ciss.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
}

#include "ccd.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
//#include "loss.h"



/*static double p_update_residual(
    sptensor_t * rnewtensor,
    ordi_matrix ** mats,
    idx_t const f,
    double const mult)
{
  
  idx_t const I = rnewtensor->dims[0];
  idx_t const J = rnewtensor->dims[1];
  idx_t const K = rnewtensor->dims[2];
  idx_t nnz = rnewtensor->nnz;
  double * avals = mats[0]+(f*I);
  double * bvals = mats[1]+(f*J);
  double * cvals = mats[2]+(f*K);
  double * residual = rnewtensor->vals;
  double myloss = 0;

  /* predict value 
  val_t predicted = 0;
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    predicted += arow[f] * brow[f] * crow[f];
  }
  val_t const loss = train->vals[x] - predicted;

  for(idx_t n = 0; n < nnz; n++)
  {
      idx_t a_id = rnewtensor->ind[0][n];
      idx_t b_id = rnewtensor->ind[1][n];
      idx_t c_id = rnewtensor->ind[2][n];
      double aval = avals[a_id];
      double bval = bvals[b_id];
      double cval = cvals[c_id];
      residual[n] += mult * aval * bval * cval;
      myloss += residual[n] * residual[n];
  }

  /* update residual in parallel 
  for(idx_t tile=0; tile < csf->ntiles; ++tile) {
    for(idx_t i=0; i < pt->nfibs[0]; ++i) {
      idx_t const a_id = (pt->fids[0] == NULL) ? i : pt->fids[0][i];
      double const aval = avals[a_id];

      for(idx_t fib=sptr[i]; fib < sptr[i+1]; ++fib) {
        double const bval = bvals[fids[fib]];
        for(idx_t jj=fptr[fib]; jj < fptr[fib+1]; ++jj){
          double const cval = cvals[inds[jj]];

          residual[jj] += mult * aval * bval * cval;
          myloss += residual[jj] * residual[jj];
       

  return myloss;
}

/**
 * @brief Compute the alpha and beta, need to be parallelized in the future
 * 
 * @param rtensor the residual
 * @param m the mode

 p_ccd_update(sptensor_t * rtensor, 
              idx_t m, 
              idx_t f, 
              idx_t nmodes, 
              ordi_matrix ** mats,
              double * nomin,
              double * denomin)
{
    idx_t nnz = rtensor->nnz;
    idx_t const I = rnewtensor->dims[m];
    idx_t const J = rnewtensor->dims[(m+1)%nmodes)];
    idx_t const K = rnewtensor->dims[(m+2)%nmodes)];
    double * avals = mats[m]+(f*I);
    double * bvals = mats[(m+1)%nmodes)]+(f*J);
    double * cvals = mats[(m+2)%nmodes)]+(f*K);
    for(idx_t n = 0; n<nnz; n++)
    {
        idx_t a_id = rtensor->inds[m][n];
        idx_t b_id = rtensor->inds[(m+1)%nmodes)][n];
        idx_t c_id = rtensor->inds[(m+2)%nmodes)][n];
        double rval = rtensor->vals[n];
        double bval = bval[b_id];
        double cval = cval[c_id];
        nomin[a_id] += rval * bval * cval;
        dnomin[a_id] += bval * bval * cval * cval;
    }
}



/**
* @brief Finalize the new f-th column of factors[m] after computing the new
*        numerator/denominator.
*
* @param m The mode to update.
* @param f The column to update.

static inline void p_compute_newcol(
    sptensor_t * rtensor,
    ordi_matrix ** mats,
    double * nmoin,
    double * denomin,
    double regularization_index,
    idx_t  m,
    idx_t f)
{
  idx_t const dim = rtensor->dims[m];
  double * avals = mats[m] + (f * dim);
  idx_t const offset = 0;

  double * numer = nmoin + offset;
  double * denom = denomin + offset;

  for(idx_t i=0; i < dim; ++i) {
    avals[i] = numer[i] / (regularization_index + denom[i]);
  }

  
}

*/

/**
* @brief Transpose a model's factor matrices.
*
* @param model The model to transpose.
*/
static void p_transpose_model(
    idx_t nmodes,
    ordi_matrix ** mats)
{
  double * buf = mats[MAX_NMODES]->values;

  for(idx_t m=0; m < nmodes; ++m) {
    idx_t const nrows = mats[m]->I;
    double * factor = mats[m]->values;
    idx_t const ncols = mats[m]->J;
    for(idx_t j=0; j < ncols; ++j) {
      for(idx_t i=0; i < nrows; ++i) {
        buf[i + (j*nrows)] = factor[j + (i*ncols)];
      }
    }

    memcpy(factor, buf, nrows * ncols * sizeof(*factor));
  }
  
}

//gpu kernels
/**
 * @brief Compute the loss
 * @version Now contains the segment scan
*/
__global__ void update_residual_gpu(ciss_t * d_traina,
                                    idx_t tilenum, 
                                    double* loss)
{
  //__shared__ double accum[DEFAULT_BLOCKSIZE];
  
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double * entries = d_traina->entries;
  
  double localloss = 0;
  if(tileid < tilenum)
  {
    idx_t localtile = tileid * DEFAULT_T_TILE_LENGTH * DEFAULT_T_TILE_WIDTH + DEFAULT_T_TILE_WIDTH;
    for(idx_t i = 0; i<DEFAULT_T_TILE_LENGTH; i++)
    {
      if(entries[localtile] < 0 && entries[localtile+1]<0) break;
      localloss+= entries[localtile + 2];
      localtile+= DEFAULT_T_TILE_WIDTH;
    }
    atomicAdd(loss, localloss);
  }


}


/**
 * @brief Compute the nomin and denomin of the fraction with warp shuffle
 * @version Now reduces the atomic operation 
*/
__global__ void update_frac_gpu_as(ciss_t * d_traina, 
                                   ordi_matrix * d_factora, 
                                   ordi_matrix * d_factorb, 
                                   ordi_matrix * d_factorc, 
                                   double * d_nominbuffer, 
                                   double * d_denominbuffer,
                                   idx_t tilenum)
{
  //get block, warp and thread index
  __shared__ uint32_t warpmask[((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)CCD_WARPSIZE)];
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t warpid = tid / ((idx_t)CCD_WARPSIZE);
  idx_t laneid = tid % ((idx_t)CCD_WARPSIZE);
  idx_t tileid = bid * ((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)CCD_WARPSIZE) + warpid;
  double * entries = d_traina -> entries;
  idx_t localtile = tileid * ((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  double __align__(256)  localtbuffer[3];
  double __align__(256)  localmbuffer[DEFAULT_NFACTORS];
  double mytmp = 0, myntmp = 0;

  #ifdef CCDAS_DEBUG
  printf("my warp id %ld\n", warpid);
  #endif

  //initialize the warp mask
  if(laneid == 0) warpmask[warpid] = 0xffffffff;
  if((tileid * DEFAULT_T_TILE_LENGTH + laneid) == d_traina->nnz)
  {
    //redefine the mask
    warpmask[warpid] = __brev((warpmask[warpid]<<(32-laneid)));      
  }

  __syncwarp();

  uint32_t mymask =  warpmask[warpid];

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
      #ifdef CCDAS_DEBUG
      if(laneid == 0)
      printf("now the itercounter is %ld\n", itercounter);
      #endif

      idx_t myfid = f_id + laneid - __popcll((bitmap << (63-laneid))) + 1;

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

      //for the loss and denomin
      for(int m = 0; m < DEFAULT_NFACTORS; m++)
      {
        localmbuffer[m] = d_factorb->values[b * DEFAULT_NFACTORS + m] * d_factorc->values[c * DEFAULT_NFACTORS + m]; 
        localtbuffer[2] -= d_factora->values[tmpi * DEFAULT_NFACTORS + m] * localmbuffer[m];
      }

      __syncwarp(mymask);

      //reduction in denomin and nomin
      for(int m = 0; m < DEFAULT_NFACTORS; m++)
      {
        mytmp = localmbuffer[m] * localmbuffer[m];
        myntmp = mybit * mytmp;
        __syncwarp(mymask);
        //now the reduction
        for(int i = 0; i < itercounter; i++)
        {
          mytmp = (__shfl_down_sync(mymask, myntmp, 1, (int)CCD_WARPSIZE)) + (!(mylbit)) * mytmp;
          myntmp = mybit * mytmp; 
          __syncwarp(mymask);                   
        }      
        if(!mybit)
        {
          atomicAdd(&(d_denominbuffer[myfid * DEFAULT_NFACTORS + m]), mytmp);  
        }
        __syncwarp(mymask);

        //now the nomins
        mytmp = localmbuffer[m] * localtbuffer[2];
        myntmp = mybit * mytmp;
        __syncwarp(mymask);
        //now the reduction
        for(int i = 0; i < itercounter; i++)
        {
          mytmp = (__shfl_down_sync(mymask, myntmp, 1, (int)CCD_WARPSIZE)) + (!(mylbit)) * mytmp;
          myntmp = mybit * mytmp; 
          __syncwarp(mymask);                   
        }      
        if(!mybit)
        {
          atomicAdd(&(d_nominbuffer[myfid * DEFAULT_NFACTORS + m]), mytmp);  
        }
        __syncwarp(mymask);
}

}
}
__syncthreads();

}

/**
 * @brief Compute the nomin and denomin of the fraction
 * @version Now contains the atomic operation 
*/
__global__ void update_frac_gpu(ciss_t * d_traina, 
                                ordi_matrix * d_factora, 
                                ordi_matrix * d_factorb, 
                                ordi_matrix * d_factorc, 
                                double * d_nominbuffer,
                                double * d_denominbuffer, 
                                idx_t tilenum)
{
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double * entries = d_traina -> entries;
  idx_t localtile = tileid*((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  double __align__(64) localloss[2] = {0, 0};
  double __align__(256) localtbuffer[6];
  double __align__(256) localmbuffer[2 * DEFAULT_NFACTORS];
  idx_t b, c;

  if(tileid < tilenum)
  {
    //get the indices and value
    idx_t f_id = (idx_t)(entries[localtile] * (-1));
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1));
    idx_t bitmap = (idx_t)(entries[localtile+2]);
    bitmap = __brevll(bitmap);
    while((bitmap & 1) == 0) {bitmap = bitmap >> 1;}
    bitmap = bitmap >> 1;
    localtile += DEFAULT_T_TILE_WIDTH;

    for(idx_t j = 0; j < DEFAULT_T_TILE_LENGTH/2; j++)
    {
      //unroll loop and load
      localtbuffer[0] = entries[localtile];
      localtbuffer[1] = entries[localtile + 1];
      localtbuffer[2] = entries[localtile + 2];
      localtbuffer[3] = entries[localtile + 3];
      localtbuffer[4] = entries[localtile + 4];
      localtbuffer[5] = entries[localtile + 5];

      //for the first
      f_id += (!(bitmap & 1));
      bitmap = bitmap >> 1;
      idx_t tmpi = d_traina->directory[f_id] - 1;
      b = (idx_t)localtbuffer[0] - 1;
      c = (idx_t)localtbuffer[1] - 1;
      localloss[0] = localtbuffer[2];
      if(localtbuffer[0] == -1 && localtbuffer[1] == -1) break;
      //load the factor matrices
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        ((double2*)localmbuffer)[i] = ((double2*)d_factorb->values)[(b * DEFAULT_NFACTORS) / 2 + i];
        ((double2*)localmbuffer)[i + DEFAULT_NFACTORS / 2] = ((double2*)d_factorc->values)[(c * DEFAULT_NFACTORS)/2 + i];
      }
      //compute the loss and denomin
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        localloss[0] -= (d_factora->values)[(tmpi * DEFAULT_NFACTORS) + i] * localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS];
        double denomin = (localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS]) * (localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS]);
        atomicAdd(&(d_denominbuffer[f_id * DEFAULT_NFACTORS + i]), denomin); 
      }
      //compute the nomin
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        double nomin = localloss[0] * localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS];
        atomicAdd(&(d_nominbuffer[f_id * DEFAULT_NFACTORS + i]), nomin);
      }

      //for the second
      f_id += (!(bitmap & 1));
      bitmap = bitmap >> 1;
      b = (idx_t)localtbuffer[3] -1 ;
      c = (idx_t)localtbuffer[4] - 1;
      tmpi = d_traina->directory[f_id] - 1;
      localloss[1] = localtbuffer[5];
      if(localtbuffer[3] == -1 && localtbuffer[4] == -1) break;
      //load the factor matrices
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        ((double2*)localmbuffer)[i] = ((double2*)d_factorb->values)[(b * DEFAULT_NFACTORS) / 2 + i];
        ((double2*)localmbuffer)[i + DEFAULT_NFACTORS / 2] = ((double2*)d_factorc->values)[(c * DEFAULT_NFACTORS)/2 + i];
      }
      //compute the loss and denomin
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        localloss[1] -= (d_factora->values)[(tmpi * DEFAULT_NFACTORS) + i]* localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS];
        double denomin = (localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS]) * (localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS]);
        atomicAdd(&(d_denominbuffer[f_id * DEFAULT_NFACTORS + i]), denomin); 
      }
      //compute the nomin
      for(idx_t i = 0; i < DEFAULT_NFACTORS; i++)
      {
        double nomin = localloss[1] * localmbuffer[i] * localmbuffer[i+DEFAULT_NFACTORS];
        atomicAdd(&(d_nominbuffer[f_id * DEFAULT_NFACTORS + i]), nomin);
      }
      localtile += 2 * DEFAULT_T_TILE_WIDTH;
    }
    
  }

}


/**
 * @brief Finally update the column for factor matrices
 * @version preliminary
*/ 
__global__ void update_ccd_gpu(ciss_t * d_traina, 
                               ordi_matrix * d_factora, 
                               double * d_nominbuffer,
                               double * d_denominbuffer, 
                               idx_t  dlength, 
                               double regularization_index)
{
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double __align__(256) nomin[DEFAULT_NFACTORS];
  double __align__(256) denomin[DEFAULT_NFACTORS];
  double * value = d_factora->values;
  
  if(tileid < dlength)
  {
    idx_t localtile = tileid * DEFAULT_NFACTORS;
    idx_t localid = d_traina->directory[tileid] - 1;
    for(idx_t i = 0; i<DEFAULT_NFACTORS;i++)
    {
      //((double2*)nomin)[i] = ((double2*)d_nominbuffer)[(localtile)/2 + i];
      //((double2*)denomin)[i] = ((double2*)d_denominbuffer)[localtile/2 + i];
      nomin[i] = d_nominbuffer[localtile + i];
      denomin[i] = d_denominbuffer[localtile + i];
    }
    for(idx_t i = 0; i<DEFAULT_NFACTORS;i++)
    {
      value[localid * DEFAULT_NFACTORS + i] = (nomin[i])/(regularization_index+denomin[i]);
    }
  }
}


/**
 * @brief The main function for tensor completion in ccd
 * @param train The tensor for generating factor matrices
 * @param validation The tensor for validation(RMSE)
 * @param test The tensor for testing the quality
 * @param regularization_index Lambda
*/
extern "C"{
void tc_ccd(sptensor_t * traina, 
            sptensor_t * trainb,
            sptensor_t * trainc,
            sptensor_t * validation,
            sptensor_t * test,
            ordi_matrix ** mats, 
            ordi_matrix ** best_mats,
            int algorithm_index,
            double regularization_index, 
            double * best_rmse, 
            double * tolerance, 
            idx_t * nbadepochs, 
            idx_t * bestepochs, 
            idx_t * max_badepochs)
{
    idx_t const nmodes = traina->nmodes;
    //pay attention to this
    //p_transpose_model(nmodes, mats);

    //for the residual
    //sptensor_t * rtensor = tt_copy(train);
    int const rank = 0;

    #ifdef CUDA_LOSS
    //to be done
    #else
    /* initialize residual, to be done in gpu */
    //for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    //    p_update_residual(rtensor, mats, DEFAULT_NFACTORS, f, -1);
    //}
    #endif
    
    //initialize the devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(0);
    //prepare the tensor in TB-COO
    ciss_t * h_traina = ciss_alloc(traina, 1);
    ciss_t * h_trainb = ciss_alloc(trainb, 2);
    ciss_t * h_trainc = ciss_alloc(trainc, 3);
    #ifdef CCD_DEBUG
    printf("finish the allocation\n");
    printf("mode2:dimension in train %d, dimension in mats %d\n", traina->dims[1],mats[1]->I);
    #endif
    struct timeval start;
    struct timeval end;
    idx_t diff;
    
    //malloc and copy the tensors + matrices to gpu
    ciss_t * d_traina, * d_trainb, * d_trainc;
    idx_t * d_directory_a, * d_directory_b, * d_directory_c;
    idx_t * d_dims_a, * d_dims_b, * d_dims_c; 
    idx_t * d_itemp1, *d_itemp2;
    double * d_entries_a , * d_entries_b, * d_entries_c; 
    double * d_ftemp, * d_nominbuffer;
    //copy tensor for mode-1
    HANDLE_ERROR(cudaMalloc((void**)&d_traina, sizeof(ciss_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_directory_a, h_traina->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_entries_a, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_dims_a, nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_a, h_traina->directory, h_traina->dlength*sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_a, h_traina->entries, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_a, h_traina->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_traina->directory;
    d_itemp2 = h_traina->dims;
    d_ftemp = h_traina->entries;
    h_traina->directory = d_directory_a;
    h_traina->dims = d_dims_a;
    h_traina->entries = d_entries_a;
    HANDLE_ERROR(cudaMemcpy(d_traina, h_traina, sizeof(ciss_t), cudaMemcpyHostToDevice));
    h_traina->directory = d_itemp1;
    h_traina->dims = d_itemp2;
    h_traina->entries = d_ftemp;
    //copy tensor for mode-2
    HANDLE_ERROR(cudaMalloc((void**)&d_trainb, sizeof(ciss_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_directory_b, h_trainb->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_entries_b, h_trainb->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_dims_b, nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_b, h_trainb->directory, h_trainb->dlength*sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_b, h_trainb->entries, h_trainb->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_b, h_trainb->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_trainb->directory;
    d_itemp2 = h_trainb->dims;
    d_ftemp = h_trainb->entries;
    h_trainb->directory = d_directory_b;
    h_trainb->dims = d_dims_b;
    h_trainb->entries = d_entries_b;
    HANDLE_ERROR(cudaMemcpy(d_trainb, h_trainb, sizeof(ciss_t), cudaMemcpyHostToDevice));
    h_trainb->directory = d_itemp1;
    h_trainb->dims = d_itemp2;
    h_trainb->entries = d_ftemp;
    //copy tensor for mode-3
    HANDLE_ERROR(cudaMalloc((void**)&d_trainc, sizeof(ciss_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_directory_c, h_trainc->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_entries_c, h_trainc->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_dims_c, nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_c, h_trainc->directory, h_trainc->dlength*sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_c, h_trainc->entries, h_trainc->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_c, h_trainc->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_trainc->directory;
    d_itemp2 = h_trainc->dims;
    d_ftemp = h_trainc->entries;
    h_trainc->directory = d_directory_c;
    h_trainc->dims = d_dims_c;
    h_trainc->entries = d_entries_c;
    HANDLE_ERROR(cudaMemcpy(d_trainc, h_trainc, sizeof(ciss_t), cudaMemcpyHostToDevice));
    h_trainc->directory = d_itemp1;
    h_trainc->dims = d_itemp2;
    h_trainc->entries = d_ftemp;

    //buffer for nomin and denomin
    idx_t maxdlength = SS_MAX(SS_MAX(h_traina->dlength, h_trainb->dlength),h_trainc->dlength);
    double * h_nominbuffer = (double *)malloc(maxdlength * DEFAULT_NFACTORS *  sizeof(double));
    double * h_denominbuffer = (double *)malloc(DEFAULT_NFACTORS * maxdlength * sizeof(double));
    HANDLE_ERROR(cudaMalloc((void**)&d_nominbuffer, DEFAULT_NFACTORS * maxdlength * sizeof(double)));
    double* d_denominbuffer; 
    HANDLE_ERROR(cudaMalloc((void**)&d_denominbuffer, DEFAULT_NFACTORS *  maxdlength * sizeof(double)));

    //copy the factor matrices
    ordi_matrix * d_factora, * d_factorb, * d_factorc;
    double * d_value_a, * d_value_b, * d_value_c;
    HANDLE_ERROR(cudaMalloc((void**)&d_factora, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_a, (mats[0]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values, (mats[0]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[0]->values;
    mats[0]->values = d_value_a;
    HANDLE_ERROR(cudaMemcpy(d_factora, mats[0], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[0]->values = d_ftemp;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorb, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_b, (mats[1]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values, (mats[1]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[1]->values;
    mats[1]->values = d_value_b;
    HANDLE_ERROR(cudaMemcpy(d_factorb, mats[1], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[1]->values = d_ftemp;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorc, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_c, (mats[2]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values, (mats[2]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[2]->values;
    mats[2]->values = d_value_c;
    HANDLE_ERROR(cudaMemcpy(d_factorc, mats[2], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[2]->values = d_ftemp;

    
    
    #ifdef CUDA_LOSS //to be done
    sptensor_gpu_t * d_test, * d_validate;    
    #else
    double loss = tc_loss_sq(traina, mats, algorithm_index);
    double frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    tc_converge(traina, validation, mats, best_mats, algorithm_index, loss, frobsq, 0, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);
    #endif

    //double * nomin = (double*)malloc(argmax_elem(spnewtensor->dims, nmodes)*sizeof(double));
    //double * dnomin = (double*)malloc(argmax_elem(spnewtensor->dims, nmodes)*sizeof(double));

    //step into the kernel
    idx_t nnz = traina->nnz;
    idx_t tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
    idx_t blocknum_m = tilenum/((idx_t)DEFAULT_BLOCKSIZE) + 1;

    idx_t mode_n, mode_i;
    /* foreach epoch */
    for(idx_t e=1; e < DEFAULT_MAX_ITERATE+1; ++e) {
       HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values,mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
       HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values,mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
       HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values,mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
       loss = 0;
       srand(time(0));
       mode_i = rand()%3;
       gettimeofday(&start,NULL);
      /*
        for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
        /* add current component to residual 
        p_update_residual(rtensor, mats, DEFAULT_NFACTORS, f, 1);

        for(idx_t inner=0; inner < NUM_INNER; ++inner) {

          /* compute new column 'f' for each factor 
          for(idx_t m=0; m < nmodes; ++m) {
            memcpy(nomin, 0, rtensor->dims[m] * sizeof(double));
            memcpy(dnomin, 0, rtensor->dims[m] * sizeof(double));
            p_ccd_update(rtensor, m, f, nmodes, DEFAULT_NFACTORS, mats, nomin, dnomin);
            
            /* numerator/denominator are now computed; update factor column 
            static inline void p_compute_newcol(rtensor, mats, nmoin, denomin, regularization_index, m, f);

          } /* foreach mode 
        } /* foreach inner iteration 

        /* subtract new rank-1 factor from residual 
        update_residual_gpu(rtensor, mats, DEFAULT_NFACTORS, f, -1);

      } /* foreach factor */
      //the GPU version, update the factors all at once
      //HANDLE_ERROR(cudaMemset(d_nominbuffer, 0, DEFAULT_NFACTORS * maxdlength * sizeof(double)));
      //HANDLE_ERROR(cudaMemset(d_denominbuffer, 0, DEFAULT_NFACTORS * maxdlength*sizeof(double)));
      for(idx_t m = 0; m < nmodes; m++)
      {
        mode_n = (mode_i + m)%3;
        HANDLE_ERROR(cudaMemset(d_nominbuffer, 0, DEFAULT_NFACTORS * maxdlength * sizeof(double)));
        HANDLE_ERROR(cudaMemset(d_denominbuffer, 0, DEFAULT_NFACTORS * maxdlength*sizeof(double)));
        switch(mode_n)
        {
          case 0: //for the first mode
          {
            idx_t blocknum_u = h_traina->dlength / DEFAULT_BLOCKSIZE + 1;      
            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_traina, d_factora, d_factorb, d_factorc, d_nominbuffer, d_denominbuffer, tilenum);
            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_traina, d_factora, d_nominbuffer,d_denominbuffer, h_traina->dlength, regularization_index); 
            break;
          }

          case 1: //for the second mode
          {
            idx_t blocknum_u = h_trainb->dlength / DEFAULT_BLOCKSIZE + 1;      
            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainb, d_factorb, d_factorc, d_factora, d_nominbuffer, d_denominbuffer, tilenum);
            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_trainb, d_factorb, d_nominbuffer,d_denominbuffer, h_trainb->dlength, regularization_index);
            break;
          } 

        //for the third mode
          default:
          {
            idx_t blocknum_u = h_trainc->dlength / DEFAULT_BLOCKSIZE + 1;      
            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainc, d_factorc, d_factora, d_factorb, d_nominbuffer, d_denominbuffer, tilenum);
            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_trainc, d_factorc, d_nominbuffer,d_denominbuffer, h_trainc->dlength, regularization_index);
            break;
          }
        }

        HANDLE_ERROR(cudaDeviceSynchronize());
      }

        gettimeofday(&end,NULL);
        diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
        printf("this time cost %ld\n",diff);

        HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaDeviceSynchronize()); 
        #ifdef CCD_DEBUG
        matrix_display(mats[0]);
        matrix_display(mats[1]);
        matrix_display(mats[2]);
        #endif
        gettimeofday(&end,NULL);
        diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
        printf("this time cost %ld\n",diff);
      
    
   /* compute RMSE and adjust learning rate */
    //the first element is used to store the final loss
    //idx_t blocknum_u = h_traina->dlength / DEFAULT_BLOCKSIZE + 1; 
    //HANDLE_ERROR(cudaMemset(d_nominbuffer, 0, sizeof(double))); 
    //update_residual_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_traina, tilenum, d_nominbuffer);
    //HANDLE_ERROR(cudaMemcpy(&loss, d_nominbuffer, sizeof(double), cudaMemcpyDeviceToHost));
    //frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    loss = tc_loss_sq(traina, mats, algorithm_index);
    frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    double obj = loss + frobsq;
    if(tc_converge(traina, validation, mats, best_mats, algorithm_index, loss, frobsq, e, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs)) {
      break;
    }

  } /* foreach epoch */

  /* print times */
  //p_transpose_model(mats);
  //p_transpose_model(ws->best_model);

  /* cleanup */
  //tt_free(rtensor);
  //free(nomin);
  //free(denomin);
  cudaFree(d_directory_a);
  cudaFree(d_dims_a);
  cudaFree(d_entries_a);
  cudaFree(d_directory_b);
  cudaFree(d_dims_b);
  cudaFree(d_entries_b);
  cudaFree(d_directory_c);
  cudaFree(d_dims_c);
  cudaFree(d_entries_c);
  cudaFree(d_nominbuffer);
  cudaFree(d_denominbuffer);
  cudaFree(d_value_a);
  cudaFree(d_value_b);
  cudaFree(d_value_c);
  cudaFree(d_traina);
  cudaFree(d_trainb);
  cudaFree(d_trainc);
  cudaFree(d_factora);
  cudaFree(d_factorb);
  cudaFree(d_factorc);

  ciss_free(h_traina);
  ciss_free(h_trainb);
  ciss_free(h_trainc);
  cudaDeviceReset();
}


}
