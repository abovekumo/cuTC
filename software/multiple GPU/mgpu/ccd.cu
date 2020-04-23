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
#include <omp.h>
//#include "loss.h"





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
__global__ void update_residual_gpu(cissbasic_t * d_traina,
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
 * @brief Compute the nomin and denomin of the fraction
 * @version Now contains the atomic operation 
*/
__global__ void update_frac_gpu(cissbasic_t * d_traina, 
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
    #ifdef CCDAS_DEBUG
    //printf("my thread id %ld\n", tid);
    #endif
    idx_t f_id = (idx_t)(entries[localtile] * (-1));
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1));
    idx_t bitmap = (idx_t)(entries[localtile+2]);
    if(bitmap != 0) 
    {
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
        #ifdef CCDAS_DEBUG
        //printf("in thread id %ld, f_id %ld, tmpi %ld, b %ld, c %ld\n", tid, f_id, tmpi, b,c);
        #endif
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
        #ifdef CCDAS_DEBUG
        //printf("in thread id %ld, f_id %ld, tmpi %ld, b %ld, c %ld\n", tid, f_id, tmpi, b, c);
        #endif
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
  __syncthreads();

}


/**
 * @brief Compute the nomin and denomin of the fraction with warp shuffle
 * @version Now reduces the atomic operation 
*/
__global__ void update_frac_gpu_as(cissbasic_t * d_traina, 
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
 * @brief Finally update the column for factor matrices
 * @version preliminary
*/ 
__global__ void update_ccd_gpu(cissbasic_t * d_traina, 
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
    int n;
    //print the GPU status
    for(n = 0; n < deviceCount; n++)
    {
      cudaDeviceProp dprop;
      cudaGetDeviceProperties(&dprop, n);
      fprintf(stdout, "   %d: %s\n", n, dprop.name);
    }
    omp_set_num_threads(deviceCount);
    //prepare the tensor in TB-COO
    
    ciss_t * h_cissta = ciss_alloc(traina, 1, deviceCount);
    ciss_t * h_cisstb = ciss_alloc(trainb, 2, deviceCount);
    ciss_t * h_cisstc = ciss_alloc(trainc, 3, deviceCount);
    #ifdef MCISS_DEBUG 
    fprintf(stdout, "now printf the new tensors\n");
    cissbasic_display(h_cisstb->cissunits[0]);
    cissbasic_display(h_cisstb->cissunits[1]);
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
    idx_t ** d_dims_a = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_dims_b = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_dims_c = (idx_t**)malloc(deviceCount * sizeof(idx_t*)); 
    double ** d_entries_a = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_entries_b = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_entries_c = (double**)malloc(deviceCount * sizeof(double*)); 
    double ** d_nominbuffer = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_denominbuffer = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_ha = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_hb = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_hc = (double**)malloc(deviceCount * sizeof(double*));

    ordi_matrix ** d_factora = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    ordi_matrix ** d_factorb = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    ordi_matrix ** d_factorc = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    double ** d_value_a = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_b = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_c = (double**)malloc(deviceCount * sizeof(double*));

    idx_t * maxdlength = (idx_t*)malloc(deviceCount * sizeof(idx_t));

    //malloc and copy the tensors + matrices to gpu
  #pragma omp parallel
  {
    //prepare the threads
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();

    int gpu_id = -1;
    cudaSetDevice(cpu_thread_id % deviceCount);  // "% num_gpus" allows more CPU threads than GPU devices
    cudaGetDevice(&gpu_id);
    
    #ifdef CCDAS_DEBUG
    fprintf(stdout, "my cpu_thread_id is %d, my gpu_id is %d\n", cpu_thread_id, gpu_id);
    #endif

    idx_t * d_itemp1, *d_itemp2;
    double * d_ftemp;
    cissbasic_t * h_traina = h_cissta->cissunits[gpu_id];
    cissbasic_t * h_trainb = h_cisstb->cissunits[gpu_id];
    cissbasic_t * h_trainc = h_cisstc->cissunits[gpu_id];
    maxdlength[gpu_id] = SS_MAX(SS_MAX(h_traina->dlength, h_trainb->dlength),h_trainc->dlength);
    //copy tensor for mode-1
    HANDLE_ERROR(cudaMalloc((void**)&(d_traina[gpu_id]), sizeof(cissbasic_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_directory_a[gpu_id]), h_traina->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_entries_a[gpu_id]), h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_dims_a[gpu_id]), nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_a[gpu_id], h_traina->directory, h_traina->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_a[gpu_id], h_traina->entries, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_a[gpu_id], h_traina->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_traina->directory;
    d_itemp2 = h_traina->dims;
    d_ftemp = h_traina->entries;
    h_traina->directory = d_directory_a[gpu_id];
    h_traina->dims = d_dims_a[gpu_id];
    h_traina->entries = d_entries_a[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_traina[gpu_id], h_traina, sizeof(cissbasic_t), cudaMemcpyHostToDevice));
    h_traina->directory = d_itemp1;
    h_traina->dims = d_itemp2;
    h_traina->entries = d_ftemp;
    //copy tensor for mode-2
    HANDLE_ERROR(cudaMalloc((void**)&(d_trainb[gpu_id]), sizeof(cissbasic_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_directory_b[gpu_id]), h_trainb->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_entries_b[gpu_id]), h_trainb->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_dims_b[gpu_id]), nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_b[gpu_id], h_trainb->directory, h_trainb->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_b[gpu_id], h_trainb->entries, h_trainb->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_b[gpu_id], h_trainb->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_trainb->directory;
    d_itemp2 = h_trainb->dims;
    d_ftemp = h_trainb->entries;
    h_trainb->directory = d_directory_b[gpu_id];
    h_trainb->dims = d_dims_b[gpu_id];
    h_trainb->entries = d_entries_b[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_trainb[gpu_id], h_trainb, sizeof(cissbasic_t), cudaMemcpyHostToDevice));
    h_trainb->directory = d_itemp1;
    h_trainb->dims = d_itemp2;
    h_trainb->entries = d_ftemp;
    //copy tensor for mode-3
    HANDLE_ERROR(cudaMalloc((void**)&(d_trainc[gpu_id]), sizeof(cissbasic_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_directory_c[gpu_id]), h_trainc->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_entries_c[gpu_id]), h_trainc->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_dims_c[gpu_id]), nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_c[gpu_id], h_trainc->directory, h_trainc->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_c[gpu_id], h_trainc->entries, h_trainc->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_c[gpu_id], h_trainc->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_trainc->directory;
    d_itemp2 = h_trainc->dims;
    d_ftemp = h_trainc->entries;
    h_trainc->directory = d_directory_c[gpu_id];
    h_trainc->dims = d_dims_c[gpu_id];
    h_trainc->entries = d_entries_c[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_trainc[gpu_id], h_trainc, sizeof(cissbasic_t), cudaMemcpyHostToDevice));
    h_trainc->directory = d_itemp1;
    h_trainc->dims = d_itemp2;
    h_trainc->entries = d_ftemp;

    //buffer for nomin and denomin
    //double * h_nominbuffer = (double *)malloc(maxdlength * DEFAULT_NFACTORS *  sizeof(double));
    //double * h_denominbuffer = (double *)malloc(DEFAULT_NFACTORS * maxdlength * sizeof(double));
    HANDLE_ERROR(cudaMalloc((void**)&(d_nominbuffer[gpu_id]), DEFAULT_NFACTORS * maxdlength[gpu_id] * sizeof(double)));
     
    HANDLE_ERROR(cudaMalloc((void**)&(d_denominbuffer[gpu_id]), DEFAULT_NFACTORS *  maxdlength[gpu_id] * sizeof(double)));

    
    //copy the factor matrices
    
    HANDLE_ERROR(cudaMalloc((void**)&(d_factora[gpu_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_a[gpu_id]), (mats[0]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_a[gpu_id], mats[0]->values, (mats[0]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    
    #pragma omp critical
    {
      d_ftemp = mats[0]->values;
      mats[0]->values = d_value_a[gpu_id];    
      HANDLE_ERROR(cudaMemcpy(d_factora[gpu_id], mats[0], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
      mats[0]->values = d_ftemp;
    }

    #pragma omp barrier

    HANDLE_ERROR(cudaMalloc((void**)&(d_factorb[gpu_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_b[gpu_id]), (mats[1]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_b[gpu_id], mats[1]->values, (mats[1]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    #pragma omp critical
    {
    d_ftemp = mats[1]->values;
    mats[1]->values = d_value_b[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_factorb[gpu_id], mats[1], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[1]->values = d_ftemp;
    }

    #pragma omp barrier

    HANDLE_ERROR(cudaMalloc((void**)&(d_factorc[gpu_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_c[gpu_id]), (mats[2]->I) * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_c[gpu_id], mats[2]->values, (mats[2]->I) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[2]->values;
    #pragma omp critical
    {mats[2]->values = d_value_c[gpu_id];
    HANDLE_ERROR(cudaMemcpy(d_factorc[gpu_id], mats[2], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[2]->values = d_ftemp;
        }
    #pragma omp barrier
    HANDLE_ERROR(cudaDeviceSynchronize());
  }  
    
    
    double loss = tc_loss_sq(traina, mats, algorithm_index);
    double frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    tc_converge(traina, validation, mats, best_mats, algorithm_index, loss, frobsq, 0, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);
        
    idx_t mode_i;
    

    //step into the kernel
    #ifdef CCDAS_DEBUG
    fprintf(stdout, "start iteration!\n");
    #endif
    
    /* foreach epoch */
    for(idx_t e=1; e < DEFAULT_MAX_ITERATE+1; ++e) {
       loss = 0;
       srand(time(0));
       mode_i = rand()%3;
       gettimeofday(&start,NULL);
       #ifdef CCDAS_DEBUG
       mode_i = 2;
       fprintf(stdout, "in %d th iteration the mode_i is %d\n", e, mode_i);
       #endif
             
       
      for(idx_t m = 0; m < nmodes; m++)
      {
        #pragma omp parallel 
        {
        unsigned int cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(cpu_thread_id);  // "% num_gpus" allows more CPU threads than GPU devices
        #ifdef CCDAS_DEBUG
        int gpu_id;
        cudaGetDevice(&gpu_id);
        fprintf(stdout, "my cpu_thread_id is %d, my gpu is %d\n", cpu_thread_id, gpu_id);
        #endif
        cissbasic_t * h_traina = h_cissta->cissunits[cpu_thread_id];
        cissbasic_t * h_trainb = h_cisstb->cissunits[cpu_thread_id];
        cissbasic_t * h_trainc = h_cisstc->cissunits[cpu_thread_id];
        idx_t nnz, tilenum, blocknum_m;
        idx_t mode_n;
        mode_n = (mode_i + m)%3;
        HANDLE_ERROR(cudaMemset(d_nominbuffer[cpu_thread_id], 0, DEFAULT_NFACTORS * maxdlength[cpu_thread_id] * sizeof(double)));
        HANDLE_ERROR(cudaMemset(d_denominbuffer[cpu_thread_id], 0, DEFAULT_NFACTORS * maxdlength[cpu_thread_id]*sizeof(double)));
        switch(mode_n)
        {
          case 0: //for the first mode
          {
            nnz = h_traina->nnz;
            tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
            //blocknum_m = tilenum/(((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)CCD_WARPSIZE)) + 1;
            blocknum_m = tilenum/((idx_t)DEFAULT_BLOCKSIZE) + 1;
            idx_t blocknum_u = h_traina->dlength / DEFAULT_BLOCKSIZE + 1;      
            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_traina[cpu_thread_id], d_factora[cpu_thread_id], d_factorb[cpu_thread_id], d_factorc[cpu_thread_id], d_nominbuffer[cpu_thread_id], d_denominbuffer[cpu_thread_id], tilenum);
            HANDLE_ERROR(cudaDeviceSynchronize());
            #ifdef CCDAS_DEBUG
            printf("for mode %d, the dlength is %d\n", mode_n, h_trainb->dlength);
            #endif
            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_traina[cpu_thread_id], d_factora[cpu_thread_id], d_nominbuffer[cpu_thread_id],d_denominbuffer[cpu_thread_id], h_traina->dlength, regularization_index); 

            HANDLE_ERROR(cudaDeviceSynchronize());
            #pragma omp barrier
            {//update the final results
            HANDLE_ERROR(cudaMemcpy(mats[0]->values + (h_cissta->d_ref[cpu_thread_id] - 1)* DEFAULT_NFACTORS, d_value_a[cpu_thread_id] + (h_cissta->d_ref[cpu_thread_id] - 1) * DEFAULT_NFACTORS, (h_cissta->d_ref[cpu_thread_id + 1] - h_cissta->d_ref[cpu_thread_id]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaDeviceSynchronize());
            }
            #pragma omp barrier
            {HANDLE_ERROR(cudaMemcpy(d_value_a[cpu_thread_id] + (h_cissta->d_ref[(cpu_thread_id + 1)% deviceCount] - 1) * DEFAULT_NFACTORS, mats[0]->values + (h_cissta->d_ref[(cpu_thread_id + 1) % deviceCount] - 1) * DEFAULT_NFACTORS,  (h_cissta->d_ref[(cpu_thread_id + 1) % deviceCount + 1] - h_cissta->d_ref[(cpu_thread_id + 1) % deviceCount]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaDeviceSynchronize());
            }
            #ifdef CCDAS_DEBUG
            printf("in cpu_thread_id, for mode %d, the synchronize finishes\n", cpu_thread_id, mode_n);
            #endif
            break;
          }

          case 1: //for the second mode
          {
            nnz = h_trainb->nnz;
            tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
            //blocknum_m = tilenum/(((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)CCD_WARPSIZE)) + 1;
            blocknum_m = tilenum/((idx_t)DEFAULT_BLOCKSIZE) + 1;
            idx_t blocknum_u = h_trainb->dlength / DEFAULT_BLOCKSIZE + 1;      
            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainb[cpu_thread_id], d_factorb[cpu_thread_id], d_factorc[cpu_thread_id], d_factora[cpu_thread_id], d_nominbuffer[cpu_thread_id], d_denominbuffer[cpu_thread_id], tilenum);
            HANDLE_ERROR(cudaDeviceSynchronize());
            #ifdef CCDAS_DEBUG
            printf("for mode %d, the dlength is %d\n", mode_n, h_trainb->dlength);
            #endif
            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_trainb[cpu_thread_id], d_factorb[cpu_thread_id], d_nominbuffer[cpu_thread_id],d_denominbuffer[cpu_thread_id], h_trainb->dlength, regularization_index);

            HANDLE_ERROR(cudaDeviceSynchronize());

            //update the final results
            #pragma omp barrier
            {
            HANDLE_ERROR(cudaMemcpy(mats[1]->values + (h_cisstb->d_ref[cpu_thread_id] - 1)  * DEFAULT_NFACTORS, d_value_b[cpu_thread_id] + (h_cisstb->d_ref[cpu_thread_id] - 1)* DEFAULT_NFACTORS, (h_cisstb->d_ref[cpu_thread_id + 1] - h_cisstb->d_ref[cpu_thread_id]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaDeviceSynchronize());
            }
            #pragma omp barrier
            {
            HANDLE_ERROR(cudaMemcpy(d_value_b[cpu_thread_id] + (h_cisstb->d_ref[(cpu_thread_id + 1)% deviceCount] - 1) * DEFAULT_NFACTORS, mats[1]->values + (h_cisstb->d_ref[(cpu_thread_id + 1) % deviceCount] - 1) * DEFAULT_NFACTORS,  (h_cisstb->d_ref[(cpu_thread_id + 1) % deviceCount + 1] - h_cisstb->d_ref[(cpu_thread_id + 1) % deviceCount]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaDeviceSynchronize());
            }
            #ifdef CCDAS_DEBUG
            printf("in cpu_thread_id, for mode %d, the synchronize finishes\n", cpu_thread_id, mode_n);
            #endif
            break;
          } 

        //for the third mode
          default:
          {
            nnz = h_trainc->nnz;
            tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
            //blocknum_m = tilenum/(((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)CCD_WARPSIZE)) + 1;
            blocknum_m = tilenum/((idx_t)DEFAULT_BLOCKSIZE) + 1;
            #ifdef CCDAS_DEBUG
            fprintf(stdout, "now cpu thread %d start the ccd update\n", cpu_thread_id);
            fprintf(stdout, "in cpu thread %d, nnz is %ld, blocknum_m %ld, tilenum %ld\n", cpu_thread_id, nnz, blocknum_m, tilenum);
            #endif
            idx_t blocknum_u = h_trainc->dlength / DEFAULT_BLOCKSIZE + 1; 
            
            update_frac_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainc[cpu_thread_id], d_factorc[cpu_thread_id], d_factora[cpu_thread_id], d_factorb[cpu_thread_id], d_nominbuffer[cpu_thread_id], d_denominbuffer[cpu_thread_id], tilenum);
            HANDLE_ERROR(cudaDeviceSynchronize()); 
            #ifdef CCDAS_DEBUG
            fprintf(stdout, "in cpu thread %d, for mode %d, the dlength is %d\n", cpu_thread_id, mode_n, h_trainc->dlength);
            #endif
            update_ccd_gpu<<<blocknum_u, DEFAULT_BLOCKSIZE,0>>>(d_trainc[cpu_thread_id], d_factorc[cpu_thread_id], d_nominbuffer[cpu_thread_id],d_denominbuffer[cpu_thread_id], h_trainc->dlength, regularization_index);

            HANDLE_ERROR(cudaDeviceSynchronize());
            #ifdef CCDAS_DEBUG
            fprintf(stdout, "now cpu thread %d finish the ccd update\n", cpu_thread_id);
            fprintf(stdout, "now the positions for d_ref are %ld, %ld, %ld\n",
            cpu_thread_id, cpu_thread_id + 1, (cpu_thread_id + 1) % deviceCount);
            fprintf(stdout, "for cpu thread %d, the d_ref are %ld, %ld, %ld\n", cpu_thread_id, h_cisstc->d_ref[1], h_cisstc->d_ref[2],h_cisstc->d_ref[0]);
            fprintf(stdout, "for cpu thread %d, the d_ref are %ld, %ld, %ld\n",cpu_thread_id, h_cisstc->d_ref[cpu_thread_id], h_cisstc->d_ref[cpu_thread_id + 1],h_cisstc->d_ref[(cpu_thread_id + 1) % deviceCount]);
            #endif
            #pragma omp barrier
            {
            //update the final results
            HANDLE_ERROR(cudaMemcpy(mats[2]->values + (h_cisstc->d_ref[cpu_thread_id] - 1)* DEFAULT_NFACTORS, d_value_c[cpu_thread_id] + (h_cisstc->d_ref[cpu_thread_id] - 1)* DEFAULT_NFACTORS, (h_cisstc->d_ref[cpu_thread_id + 1] - h_cisstc->d_ref[cpu_thread_id]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaDeviceSynchronize());
            }
            #pragma omp barrier
            {
            HANDLE_ERROR(cudaMemcpy(d_value_c[cpu_thread_id] + (h_cisstc->d_ref[(cpu_thread_id + 1)% deviceCount] - 1) * DEFAULT_NFACTORS, mats[2]->values + (h_cisstc->d_ref[(cpu_thread_id + 1) % deviceCount] - 1) * DEFAULT_NFACTORS,  (h_cisstc->d_ref[(cpu_thread_id + 1) % deviceCount + 1] - h_cisstc->d_ref[(cpu_thread_id + 1) % deviceCount]) * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaDeviceSynchronize());
            }
            #ifdef CCDAS_DEBUG
            printf("in cpu_thread_id, for mode %d, the synchronize finishes\n", cpu_thread_id, mode_n);
            #endif
            break;
          }
          
        }
        #pragma omp barrier
        
      }

      }

        HANDLE_ERROR(cudaDeviceSynchronize()); 
        
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
#pragma omp parallel
{
  unsigned int cpu_thread_id = omp_get_thread_num();
  cudaSetDevice(cpu_thread_id % deviceCount); 
  cudaFree(d_directory_a[cpu_thread_id]);
  cudaFree(d_dims_a[cpu_thread_id]);
  cudaFree(d_entries_a[cpu_thread_id]);
  cudaFree(d_directory_b[cpu_thread_id]);
  cudaFree(d_dims_b[cpu_thread_id]);
  cudaFree(d_entries_b[cpu_thread_id]);
  cudaFree(d_directory_c[cpu_thread_id]);
  cudaFree(d_dims_c[cpu_thread_id]);
  cudaFree(d_entries_c[cpu_thread_id]);
  cudaFree(d_nominbuffer[cpu_thread_id]);
  cudaFree(d_denominbuffer[cpu_thread_id]);
  cudaFree(d_value_a[cpu_thread_id]);
  cudaFree(d_value_b[cpu_thread_id]);
  cudaFree(d_value_c[cpu_thread_id]);
  cudaFree(d_traina[cpu_thread_id]);
  cudaFree(d_trainb[cpu_thread_id]);
  cudaFree(d_trainc[cpu_thread_id]);
  cudaFree(d_factora[cpu_thread_id]);
  cudaFree(d_factorb[cpu_thread_id]);
  cudaFree(d_factorc[cpu_thread_id]);
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
  free(d_dims_a); 
  free(d_dims_b); 
  free(d_dims_c);
  free(d_entries_a); 
  free(d_entries_b); 
  free(d_entries_c); 
  free(d_nominbuffer); 
  free(d_denominbuffer); 
  free(d_value_ha); 
  free(d_value_hb); 
  free(d_value_hc); 

  free(d_factora); 
  free(d_factorb); 
  free(d_factorc); 
  free(d_value_a); 
  free(d_value_b); 
  free(d_value_c); 
  free(maxdlength);
  
}


}
