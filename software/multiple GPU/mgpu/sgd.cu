
extern "C"
{
#include "completion.h"
#include "ciss.h"
#include "base.h"
#include "matrixprocess.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
}

#include "sgd.cuh"
#include "loss.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>


/**
* @brief Update a three-mode model based on a given observation.
*
* @param DEFAULT_NFACTORS The rank.
* @param train The training data.
* @param nnz_index The index of the observation to update from.
* @param mats The model to update.

static inline void p_update_sgd(
    sptensor_t * train,
    idx_t nnz_index,
    ordi_matrix ** mats,
    double learning_rate,
    double regularization_index
    )
{
  idx_t const x = nnz_index;

  assert(train->nmodes == 3);

  idx_t ** ind = train->ind;
  double * arow = mats[0] + (ind[0][x] * DEFAULT_NFACTORS);
  double * brow = mats[1] + (ind[1][x] * DEFAULT_NFACTORS);
  double * crow = mats[2] + (ind[2][x] * DEFAULT_NFACTORS);

  /* predict value 
  double predicted = 0;
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    predicted += arow[f] * brow[f] * crow[f];
  }
  double const loss = train->vals[x] - predicted;
  double const rate = learning_rate;
  double reg = regularization_index;

  /* update rows 
  for(idx_t f=0; f < DEFAULT_NFACTORS; ++f) {
    double const moda = (loss * brow[f] * crow[f]) - (reg[0] * arow[f]);
    double const modb = (loss * arow[f] * crow[f]) - (reg[1] * brow[f]);
    double const modc = (loss * arow[f] * brow[f]) - (reg[2] * crow[f]);
    arow[f] += rate * moda;
    brow[f] += rate * modb;
    crow[f] += rate * modc;
  }
}*/

//the gpu kernel
__global__ void p_update_sgd_gpu(cissbasic_t * d_traina, 
                                 ordi_matrix * d_factora,
                                 ordi_matrix * d_factorb, 
                                 ordi_matrix * d_factorc, 
                                 double * d_value_ha,
                                 double * d_value_hb,
                                 double * d_value_hc,
                                 double learning_rate, 
                                 double regularization_index,
                                 idx_t tilebegin,
                                 idx_t tileend)
{
  //get thread and block index
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid + tilebegin;
  //idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double * entries = d_traina->entries;
  idx_t localtile = tileid*((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  //buffer for matrices
  double __align__(256) mabuffer[DEFAULT_NFACTORS];
  double __align__(256) mbbuffer[DEFAULT_NFACTORS];
  double __align__(256) mcbuffer[DEFAULT_NFACTORS];
  double __align__(256) localtbuffer[6];
  idx_t a,b,c, localcounter;
  double localvalue;

  #ifdef SGD_DEBUG
  if(bid ==0 && tid == 0) {printf("my tilebegin is %ld, tileend is %ld\n", tilebegin, tileend);}
  __syncthreads();
  #endif

  if(tileid < tileend)
  {
    #ifdef SGD_DEBUG
    printf("now mytileid is %ld, localtile is %ld, the first element in entries is %lf\n", tileid, localtile, entries[localtile]);
    #endif
    //get the indices and value
    idx_t f_id = (idx_t)(entries[localtile] * (-1));
    #ifdef SGD_DEBUG
    //printf("now in thread %ld my fid is %ld\n", tid, f_id);
    #endif
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1));
    idx_t bitmap = (idx_t)(entries[localtile+2]);
    if(bitmap != 0)
    {
      bitmap = __brevll(bitmap);
      while((bitmap & 1) == 0) {bitmap = (bitmap >> 1);}
      bitmap = (bitmap >> 1);
      localtile += DEFAULT_T_TILE_WIDTH;

      for(idx_t j = 0; j < DEFAULT_T_TILE_LENGTH/2; j++)
      {
        //unroll loop and load
        localtbuffer[0] = entries[localtile];
        localtbuffer[1] = entries[localtile + 1];
        localtbuffer[2] = entries[localtile + 2];
        

        if(localtbuffer[0] == -1 && localtbuffer[1] == -1) break;
        //for the first
        f_id += (!(bitmap & 1));
        #ifdef SGD_DEBUG
        if(f_id >= 260208)printf("now in thread %ld my fid is %ld\n", tid, f_id);
        #endif
        bitmap = bitmap >> 1;
        a = d_traina->directory[f_id] - 1;
        localcounter = d_traina->dcounter[f_id + 1] - d_traina->dcounter[f_id];
        //dcounter = d_traina->dcounter[f_id];
        b = (idx_t)localtbuffer[0] - 1;
        c = (idx_t)localtbuffer[1] - 1;
        #ifdef SGD_DEBUG
        if(localtbuffer[1] == 0 or localtbuffer[2] == 0)printf("now in thread %ld are zero indices\n", tid);
        #endif
        localvalue = localtbuffer[2];
        #ifdef SGD_DEBUG
        printf("now a b c in tile %ld are %ld %ld %ld\n", tileid, a, b, c);
        #endif
        //if(localtbuffer[0] == -1 && localtbuffer[1] == -1) break;
        for(idx_t i = 0; i< DEFAULT_NFACTORS; i++)
        {
          mabuffer[i] = (d_factora->values)[a * DEFAULT_NFACTORS + i] + d_value_ha[a * DEFAULT_NFACTORS + i];
          mbbuffer[i] = (d_factorb->values)[b * DEFAULT_NFACTORS + i] + d_value_hb[b * DEFAULT_NFACTORS + i];
          mcbuffer[i] = (d_factorc->values)[c * DEFAULT_NFACTORS + i] + d_value_hc[c * DEFAULT_NFACTORS + i];

        }
        /* predict value */
        double predicted = 0;
        for(idx_t f=0; f < DEFAULT_NFACTORS; f++) {
          predicted += mabuffer[f] * mbbuffer[f] * mcbuffer[f];
        }
        predicted = localvalue - predicted;
        /* update rows */
        for(idx_t f=0; f < DEFAULT_NFACTORS; f++) {
          double moda = (predicted * mbbuffer[f] * mcbuffer[f]) - (regularization_index * mabuffer[f]);
          double modb = (predicted * mabuffer[f] * mcbuffer[f]) - (regularization_index * mbbuffer[f]);
          double modc = (predicted * mbbuffer[f] * mabuffer[f]) - (regularization_index * mcbuffer[f]);
          atomicAdd(&(d_factora->values[a * DEFAULT_NFACTORS + f]), learning_rate*moda * (double)(SGD_MODIFICATIONA));
          atomicAdd(&(d_factorb->values[b * DEFAULT_NFACTORS + f]), learning_rate*modb * (double)(SGD_MODIFICATIONB));
          atomicAdd(&(d_factorc->values[c * DEFAULT_NFACTORS + f]), learning_rate*modc * (double)(SGD_MODIFICATIONC));
       }

       //for the second
       localtbuffer[3] = entries[localtile + 3];
       localtbuffer[4] = entries[localtile + 4];
       localtbuffer[5] = entries[localtile + 5];
       f_id += (!(bitmap & 1));
       #ifdef SGD_DEBUG
       if(f_id >= 260208)printf("now in thread %ld my fid is %ld\n", tid, f_id);
       #endif
       bitmap = bitmap >> 1;
       a = d_traina->directory[f_id] - 1;
       localcounter = d_traina->dcounter[f_id + 1] - d_traina->dcounter[f_id];
       b = (idx_t)localtbuffer[3] - 1;
       c = (idx_t)localtbuffer[4] - 1;
       #ifdef SGD_DEBUG
       if(localtbuffer[3] == 0 or localtbuffer[4] == 0)printf("now in thread %ld are zero indices\n", tid);
       #endif
       #ifdef SGD_DEBUG
       printf("now a b c in tile %ld are %ld %ld %ld\n", tileid, a, b, c);
       #endif
       localvalue = localtbuffer[5];
       if(localtbuffer[3] == -1 && localtbuffer[4] == -1) break;
       for(idx_t i = 0; i< DEFAULT_NFACTORS; i++)
       {
        mabuffer[i] = (d_factora->values)[a * DEFAULT_NFACTORS + i] + d_value_ha[a * DEFAULT_NFACTORS + i];
        mbbuffer[i] = (d_factorb->values)[b * DEFAULT_NFACTORS + i] + d_value_hb[b * DEFAULT_NFACTORS + i];
        mcbuffer[i] = (d_factorc->values)[c * DEFAULT_NFACTORS + i] + d_value_hc[c * DEFAULT_NFACTORS + i];
       }
       /* predict value */
       predicted = 0;
       for(idx_t f=0; f < DEFAULT_NFACTORS; f++) {
         predicted += mabuffer[f] * mbbuffer[f] * mcbuffer[f];
       }
       predicted = localvalue - predicted;
       /* update rows */
       for(idx_t f=0; f < DEFAULT_NFACTORS; f++) {
         double moda = (predicted * mbbuffer[f] * mcbuffer[f]) - (regularization_index * mabuffer[f]);
         double modb = (predicted * mabuffer[f] * mcbuffer[f]) - (regularization_index * mbbuffer[f]);
         double modc = (predicted * mbbuffer[f] * mabuffer[f]) - (regularization_index * mcbuffer[f]);
         atomicAdd(&(d_factora->values[a * DEFAULT_NFACTORS + f]), learning_rate*moda * (double)(SGD_MODIFICATIONA));
         atomicAdd(&(d_factorb->values[b * DEFAULT_NFACTORS + f]), learning_rate*modb * (double)(SGD_MODIFICATIONB));
         atomicAdd(&(d_factorc->values[c * DEFAULT_NFACTORS + f]), learning_rate*modc * (double)(SGD_MODIFICATIONC));
      }
    localtile +=  2 * DEFAULT_T_TILE_WIDTH;
}
  }
}
  __syncthreads();

}



/**
 * @brief The main function for tensor completion in sgd
 * @param train The tensor for generating factor matrices
 * @param validation The tensor for validation(RMSE)
 * @param test The tensor for testing the quality
 * @param regularization_index Lambda
*/
extern "C"{
void tc_sgd(sptensor_t * traina, 
            sptensor_t * trainb,
            sptensor_t * trainc,
            sptensor_t * validation,
            sptensor_t * test,
            ordi_matrix ** mats, 
            ordi_matrix ** best_mats,
            ordi_matrix ** aux_mats,
            int algorithm_index,
            double regularization_index, 
            double learning_rate,
            double * best_rmse, 
            double * tolerance, 
            idx_t * nbadepochs, 
            idx_t * bestepochs, 
            idx_t * max_badepochs)
{
    //only in sgd
    idx_t steps_size = 1000;
    idx_t nmodes = traina->nmodes;

    //initialize the devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int n;
    //print the GPU status
    for(n = 0; n < deviceCount; n++)
    {
      cudaDeviceProp dprop;
      cudaGetDeviceProperties(&dprop, n);
      printf("   %d: %s\n", n, dprop.name);
    }
    omp_set_num_threads(deviceCount);
    //prepare the tensor in TB-COO
    cissbasic_t * h_traina = cissbasic_alloc(traina, 0, traina->ind[0][0], (traina->ind[0][traina->nnz - 1] + 1));
    #ifdef MCISS_DEBUG
    fprintf(stdout, "the newtensor\n");
    fprintf(stdout, "the lasti is %ld\n",traina->ind[0][traina->nnz - 1] + 1);
    cissbasic_display(h_traina);
    #endif
    struct timeval start;
    struct timeval end;
    idx_t diff;
    
    
    //copy the real and auxiliary factor matrices
    cissbasic_t ** d_traina = (cissbasic_t**)malloc(deviceCount * sizeof(cissbasic_t*));
    idx_t ** d_directory_a = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    idx_t ** d_counter_a = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    
    idx_t ** d_dims_a = (idx_t**)malloc(deviceCount * sizeof(idx_t*));
    
    double ** d_entries_a = (double**)malloc(deviceCount * sizeof(double*));
    
    double ** d_value_ha = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_hb = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_hc = (double**)malloc(deviceCount * sizeof(double*));

    ordi_matrix ** d_factora = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    ordi_matrix ** d_factorb = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    ordi_matrix ** d_factorc = (ordi_matrix**)malloc(deviceCount * sizeof(ordi_matrix*));
    double ** d_value_a = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_b = (double**)malloc(deviceCount * sizeof(double*));
    double ** d_value_c = (double**)malloc(deviceCount * sizeof(double*));
    
    
  #pragma omp parallel
  {
    //prepare the threads
    //
    unsigned int cpu_thread_id = omp_get_thread_num();
    //unsigned int num_cpu_threads = omp_get_num_threads();

    //set gpus
    cudaSetDevice(cpu_thread_id % deviceCount);  // "% num_gpus" allows more CPU threads than GPU devices

    idx_t * d_itemp1, *d_itemp2, * d_itemp3;
    double * d_ftemp;

    cissbasic_t * myh_traina = (cissbasic_t*)malloc(sizeof(cissbasic_t));
    myh_traina->directory = h_traina->directory;
    myh_traina->dcounter = h_traina->dcounter;
    myh_traina->entries = h_traina->entries;
    myh_traina->dims = h_traina->dims;
    myh_traina->nnz = h_traina->nnz;
    myh_traina->nmodes = h_traina->nmodes;
    myh_traina->size = h_traina->size;
    myh_traina->dlength = h_traina->dlength;
 
    #ifdef MCISS_DEBUG
    fprintf(stdout, "in cpu_thread %d, my tensor\n", cpu_thread_id);
    cissbasic_display(myh_traina);
    #endif
             
    //copy tensor for mode-1
    HANDLE_ERROR(cudaMalloc((void**)&(d_traina[cpu_thread_id]), sizeof(cissbasic_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_directory_a[cpu_thread_id]), myh_traina->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_counter_a[cpu_thread_id]), (myh_traina->dlength + 1) * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_entries_a[cpu_thread_id]), myh_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_dims_a[cpu_thread_id]), nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_a[cpu_thread_id], myh_traina->directory, myh_traina->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_counter_a[cpu_thread_id], myh_traina->dcounter, (myh_traina->dlength + 1)*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_a[cpu_thread_id], myh_traina->entries, myh_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_a[cpu_thread_id], myh_traina->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = myh_traina->directory;
    d_itemp2 = myh_traina->dims;
    d_itemp3 = myh_traina->dcounter;
    d_ftemp = myh_traina->entries;
    myh_traina->directory = d_directory_a[cpu_thread_id];
    myh_traina->dcounter = d_counter_a[cpu_thread_id];
    myh_traina->dims = d_dims_a[cpu_thread_id];
    myh_traina->entries = d_entries_a[cpu_thread_id];
    HANDLE_ERROR(cudaMemcpy(d_traina[cpu_thread_id], myh_traina, sizeof(cissbasic_t), cudaMemcpyHostToDevice));
    myh_traina->directory = d_itemp1;
    myh_traina->dims = d_itemp2;
    myh_traina->dcounter = d_itemp3;
    myh_traina->entries = d_ftemp;
    
    
    HANDLE_ERROR(cudaMalloc((void**)&(d_factora[cpu_thread_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_a[cpu_thread_id]), mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_a[cpu_thread_id], mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    #pragma omp critical
    {d_ftemp = mats[0]->values;
    mats[0]->values = d_value_a[cpu_thread_id];
    HANDLE_ERROR(cudaMemcpy(d_factora[cpu_thread_id], mats[0], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[0]->values = d_ftemp;
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
    #pragma omp barrier

    HANDLE_ERROR(cudaMalloc((void**)&(d_factorb[cpu_thread_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_b[cpu_thread_id]), mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_b[cpu_thread_id], mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
  
    #pragma omp critical
    {
    d_ftemp = mats[1]->values;
    mats[1]->values = d_value_b[cpu_thread_id];
    HANDLE_ERROR(cudaMemcpy(d_factorb[cpu_thread_id], mats[1], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[1]->values = d_ftemp;
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
    #pragma omp barrier

    HANDLE_ERROR(cudaMalloc((void**)&(d_factorc[cpu_thread_id]), sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_c[cpu_thread_id]), mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_c[cpu_thread_id], mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    #pragma omp critical
    {d_ftemp = mats[2]->values;
    mats[2]->values = d_value_c[cpu_thread_id];
    HANDLE_ERROR(cudaMemcpy(d_factorc[cpu_thread_id], mats[2], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[2]->values = d_ftemp;
    }

    #pragma omp barrier

    //for auxiliary factor matrices
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_ha[cpu_thread_id]), mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_hb[cpu_thread_id]), mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_value_hc[cpu_thread_id]), mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaMemset(d_value_ha[cpu_thread_id], 0, mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemset(d_value_hb[cpu_thread_id], 0, mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemset(d_value_hc[cpu_thread_id], 0, mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
   
    
    HANDLE_ERROR(cudaDeviceSynchronize());
    free(myh_traina);
  }

   

    #ifdef CUDA_LOSS
    //to be done
    #else
    double loss = tc_loss_sq(traina, mats, algorithm_index);
    double frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    tc_converge(traina, validation, mats, best_mats, algorithm_index, loss, frobsq, 0, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);
    #endif

    /* for bold driver */
    double obj = loss + frobsq;
    double prev_obj = obj;

    //step into the kernel
    idx_t nnz = traina->nnz;
    idx_t tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
    idx_t* tileptr = (idx_t*)malloc((deviceCount + 1) * sizeof(idx_t));
    tileptr[0] = 0;
    tileptr[deviceCount] = tilenum;
    for(n = 1; n < deviceCount; n++)
    {
      tileptr[n] = tilenum / deviceCount * (n);
    } 

    #ifdef SGD_DEBUG
    for(n = 0; n < deviceCount + 1; n++)
    printf("now the tileptr[%ld] is %ld\n", n, tileptr[n]);
    #endif
    
    #ifdef SGD_DEBUG
    printf("nnz %d tilenum %d\n", nnz, tilenum);
    #endif

    /* foreach epoch */
    for(idx_t e=1; e < DEFAULT_MAX_ITERATE; ++e) {

    /* update model from all training observations */
    gettimeofday(&start,NULL);
  #pragma omp parallel
  {
    //prepare the threads
    unsigned int cpu_thread_id = omp_get_thread_num();
    idx_t blocknum_m = (tileptr[cpu_thread_id + 1] - tileptr[cpu_thread_id] - 1)/DEFAULT_BLOCKSIZE + 1;
    //idx_t blocknum_m = tilenum/DEFAULT_BLOCKSIZE + 1;
    //set gpus
    cudaSetDevice(cpu_thread_id % deviceCount);  // "% num_gpus" allows more CPU threads than GPU devices
    #ifdef SGD_DEBUG
    printf("now in thread %d, the sgd starts, tilebegin at %ld, tileend at %ld, blocknum is %ld\n", cpu_thread_id, tileptr[cpu_thread_id], tileptr[cpu_thread_id + 1], blocknum_m);
    #endif
    p_update_sgd_gpu<<<blocknum_m, DEFAULT_BLOCKSIZE, 0>>>(d_traina[cpu_thread_id], d_factora[cpu_thread_id], d_factorb[cpu_thread_id], d_factorc[cpu_thread_id], d_value_ha[cpu_thread_id], d_value_hb[cpu_thread_id], d_value_hc[cpu_thread_id], learning_rate, regularization_index, tileptr[cpu_thread_id], tileptr[cpu_thread_id + 1]);      

    HANDLE_ERROR(cudaDeviceSynchronize());
    #ifdef SGD_DEBUG
    printf("now in thread %d, the sgd ends, tilebegin at %ld, tileend at %ld\n", cpu_thread_id, tileptr[cpu_thread_id], tileptr[cpu_thread_id + 1]);
    #endif

    #pragma omp barrier    
    if(!cpu_thread_id)
    {
      HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a[cpu_thread_id], mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_b[cpu_thread_id], mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_c[cpu_thread_id], mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
    }
    else
    {
      HANDLE_ERROR(cudaMemcpy(aux_mats[0]->values, d_value_a[cpu_thread_id], mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(aux_mats[1]->values, d_value_b[cpu_thread_id], mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(aux_mats[2]->values, d_value_c[cpu_thread_id], mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaDeviceSynchronize()); 
    #pragma omp barrier
    if(!cpu_thread_id)
    {
      HANDLE_ERROR(cudaMemcpy(d_value_ha[cpu_thread_id], aux_mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(d_value_hb[cpu_thread_id], aux_mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(d_value_hc[cpu_thread_id], aux_mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    }
    else
    {
      HANDLE_ERROR(cudaMemcpy(d_value_ha[cpu_thread_id], mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(d_value_hb[cpu_thread_id], mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(d_value_hc[cpu_thread_id], mats[2]->values,  mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    }
    HANDLE_ERROR(cudaDeviceSynchronize()); 
  }  
    gettimeofday(&end,NULL);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("this time cost %ld\n",diff);
    /* compute RMSE and adjust learning rate */
    loss = tc_loss_sq_sgd(traina, mats, aux_mats, algorithm_index);
    frobsq = tc_frob_sq_sgd(nmodes, regularization_index, mats, aux_mats);
    obj = loss + frobsq;
    if(tc_converge_sgd(traina, validation, mats, best_mats, aux_mats, algorithm_index, loss, frobsq, e, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs)) {
      break;
    }

    /* bold driver */
    if(e > 1) {
      if(obj < prev_obj) {
        learning_rate *= 1.05;
      } else {
        learning_rate *= 0.50;
      }
    }

    prev_obj = obj;
  }
  //free the cudabuffer

#pragma omp parallel
{
  unsigned int cpu_thread_id = omp_get_thread_num();
  
  //set gpus
  cudaSetDevice(cpu_thread_id % deviceCount);  // "% num_gpus" allows more CPU threads than GPU devices
  cudaFree(d_directory_a[cpu_thread_id]);
  cudaFree(d_dims_a[cpu_thread_id]);
  cudaFree(d_entries_a[cpu_thread_id]);
  cudaFree(d_value_ha[cpu_thread_id]);
  cudaFree(d_value_hb[cpu_thread_id]);
  cudaFree(d_value_hc[cpu_thread_id]);
  cudaFree(d_value_a[cpu_thread_id]);
  cudaFree(d_value_b[cpu_thread_id]);
  cudaFree(d_value_c[cpu_thread_id]);
  cudaFree(d_traina[cpu_thread_id]);
  cudaFree(d_factora[cpu_thread_id]);
  cudaFree(d_factorb[cpu_thread_id]);
  cudaFree(d_factorc[cpu_thread_id]);

  cudaDeviceReset();  
}
  cissbasic_free(h_traina); 
  free(d_traina); 
   
  free(d_directory_a); 
  free(d_counter_a);
  free(d_dims_a); 
  
  free(d_entries_a); 
  
  free(d_value_ha); 
  free(d_value_hb); 
  free(d_value_hc); 

  free(d_factora); 
  free(d_factorb); 
  free(d_factorc); 
  free(d_value_a); 
  free(d_value_b); 
  free(d_value_c); 
  
  free(tileptr);
}
}