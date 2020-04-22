
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
#include <cuda.h>
#include <cuda_runtime.h>
//#include "loss.h"


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
__global__ void p_update_sgd_gpu(ciss_t * d_traina, 
                                 ordi_matrix * d_factora,
                                 ordi_matrix * d_factorb, 
                                 ordi_matrix * d_factorc, 
                                 double learning_rate, 
                                 double regularization_index,
                                 idx_t tilenum)
{
  //get thread and block index
  idx_t bid = blockIdx.x;
  idx_t tid = threadIdx.x;
  idx_t tileid = bid * DEFAULT_BLOCKSIZE + tid;
  double * entries = d_traina->entries;
  idx_t localtile = tileid*((DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH);
  //buffer for matrices
  double __align__(256) mabuffer[DEFAULT_NFACTORS];
  double __align__(256) mbbuffer[DEFAULT_NFACTORS];
  double __align__(256) mcbuffer[DEFAULT_NFACTORS];
  double __align__(256) localtbuffer[6];
  idx_t a,b,c, localcounter;
  double localvalue;

  if(tileid < tilenum)
  {
    //get the indices and value
    idx_t f_id = (idx_t)(entries[localtile] * (-1));
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1));
    idx_t bitmap = (idx_t)(entries[localtile+2]);
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
      bitmap = bitmap >> 1;
      a = d_traina->directory[f_id] - 1;
      localcounter = d_traina->dcounter[f_id + 1] - d_traina->dcounter[f_id];
      b = (idx_t)localtbuffer[0] - 1;
      c = (idx_t)localtbuffer[1] - 1;
      localvalue = localtbuffer[2];
      #ifdef SGD_DEBUG
      printf("now a b c in tile %ld are %ld %ld %ld\n", tileid, a, b, c);
      #endif
      //if(localtbuffer[0] == -1 && localtbuffer[1] == -1) break;
      for(idx_t i = 0; i< DEFAULT_NFACTORS; i++)
      {
        //((double2*)mabuffer)[i] = ((double2*)d_factora->values)[a * DEFAULT_NFACTORS/2 + i];
        //((double2*)mbbuffer)[i] = ((double2*)d_factorb->values)[b * DEFAULT_NFACTORS/2 + i];
        //((double2*)mcbuffer)[i] = ((double2*)d_factorc->values)[c * DEFAULT_NFACTORS/2 + i];
        mabuffer[i] = (d_factora->values)[a * DEFAULT_NFACTORS + i];
        mbbuffer[i] = (d_factorb->values)[b * DEFAULT_NFACTORS + i];
        mcbuffer[i] = (d_factorc->values)[c * DEFAULT_NFACTORS + i];

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
        atomicAdd(&(d_factora->values[a * DEFAULT_NFACTORS + f]), learning_rate*moda * (double)SGD_MODIFICATIONA);
        atomicAdd(&(d_factorb->values[b * DEFAULT_NFACTORS + f]), learning_rate*modb * (double)SGD_MODIFICATIONB);
        atomicAdd(&(d_factorc->values[c * DEFAULT_NFACTORS + f]), learning_rate*modc * (double)SGD_MODIFICATIONC);
     }

     //for the second
     localtbuffer[3] = entries[localtile + 3];
     localtbuffer[4] = entries[localtile + 4];
     localtbuffer[5] = entries[localtile + 5];
     f_id += (!(bitmap & 1));
     bitmap = bitmap >> 1;
     a = d_traina->directory[f_id] - 1;
     localcounter = d_traina->dcounter[f_id + 1] - d_traina->dcounter[f_id];
     b = (idx_t)localtbuffer[3] - 1;
     c = (idx_t)localtbuffer[4] - 1;
     #ifdef SGD_DEBUG
     printf("now a b c in tile %ld are %ld %ld %ld\n", tileid, a, b, c);
     #endif
     localvalue = localtbuffer[5];
     if(localtbuffer[3] == -1 && localtbuffer[4] == -1) break;
     for(idx_t i = 0; i< DEFAULT_NFACTORS; i++)
     {
      mabuffer[i] = (d_factora->values)[a * DEFAULT_NFACTORS + i];
      mbbuffer[i] = (d_factorb->values)[b * DEFAULT_NFACTORS + i];
      mcbuffer[i] = (d_factorc->values)[c * DEFAULT_NFACTORS + i];
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
       atomicAdd(&(d_factora->values[a * DEFAULT_NFACTORS + f]), learning_rate*moda * (double)SGD_MODIFICATIONA);
       atomicAdd(&(d_factorb->values[b * DEFAULT_NFACTORS + f]), learning_rate*modb * (double)SGD_MODIFICATIONA);
       atomicAdd(&(d_factorc->values[c * DEFAULT_NFACTORS + f]), learning_rate*modc * (double)SGD_MODIFICATIONA);
    }
    localtile +=  2 * DEFAULT_T_TILE_WIDTH;
}

}


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
    cudaSetDevice(0);
    //prepare the tensor in TB-COO
    ciss_t * h_traina = ciss_alloc(traina, 1);
    #ifdef CISS_DEBUG
    ciss_display(h_traina);
    #endif
    //ciss_t * h_trainb = ciss_alloc(train, 1);
    //ciss_t * h_trainc = ciss_alloc(train, 2);
    struct timeval start;
    struct timeval end;
    idx_t diff;
    
    //malloc and copy the tensors + matrices to gpu
    ciss_t * d_traina;
    idx_t * d_directory_a, * d_counter_a;
    idx_t * d_dims_a; 
    idx_t * d_itemp1, *d_itemp2, *d_itemp3;
    double * d_entries_a; 
    double * d_ftemp;
    //copy tensor for mode-1
    HANDLE_ERROR(cudaMalloc((void**)&d_traina, sizeof(ciss_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_directory_a, h_traina->dlength * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&(d_counter_a), (h_traina->dlength + 1) * sizeof(idx_t)));
    HANDLE_ERROR(cudaMalloc((void**)&d_entries_a, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void**)&d_dims_a, nmodes * sizeof(idx_t)));
    HANDLE_ERROR(cudaMemcpy(d_directory_a, h_traina->directory, h_traina->dlength*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_counter_a, h_traina->dcounter, (h_traina->dlength + 1)*sizeof(idx_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_entries_a, h_traina->entries, h_traina->size * DEFAULT_T_TILE_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_dims_a, h_traina->dims, nmodes*sizeof(idx_t), cudaMemcpyHostToDevice));
    d_itemp1 = h_traina->directory;
    d_itemp2 = h_traina->dims;
    d_itemp3 = h_traina->dcounter;
    d_ftemp = h_traina->entries;
    h_traina->directory = d_directory_a;
    h_traina->dcounter = d_counter_a;
    h_traina->dims = d_dims_a;
    h_traina->entries = d_entries_a;
    HANDLE_ERROR(cudaMemcpy(d_traina, h_traina, sizeof(ciss_t), cudaMemcpyHostToDevice));
    h_traina->directory = d_itemp1;
    h_traina->dims = d_itemp2;
    h_traina->dcounter = d_itemp3;
    h_traina->entries = d_ftemp;
    
    //buffer for HTH
    //idx_t maxdlength = SS_MAX(SS_MAX(h_traina->dlength, h_trainb->dlength),h_trainc->dlength);
    //double * h_hbuffer = (double *)malloc(DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double));
    //double * h_invbuffer = (double *)malloc(DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double));
    //HANDLE_ERROR(cudaMalloc((void**)&d_hbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)));
    //double* d_invbuffer; //for inverse
    //HANDLE_ERROR(cudaMalloc((void**)&d_invbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)));

    //copy the factor matrices
    ordi_matrix * d_factora, * d_factorb, * d_factorc;
    double * d_value_a, * d_value_b, * d_value_c;
    HANDLE_ERROR(cudaMalloc((void**)&d_factora, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[0]->values;
    mats[0]->values = d_value_a;
    HANDLE_ERROR(cudaMemcpy(d_factora, mats[0], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[0]->values = d_ftemp;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorb, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[1]->values;
    mats[1]->values = d_value_b;
    HANDLE_ERROR(cudaMemcpy(d_factorb, mats[1], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[1]->values = d_ftemp;

    HANDLE_ERROR(cudaMalloc((void**)&d_factorc, sizeof(ordi_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    d_ftemp = mats[2]->values;
    mats[2]->values = d_value_c;
    HANDLE_ERROR(cudaMemcpy(d_factorc, mats[2], sizeof(ordi_matrix), cudaMemcpyHostToDevice));
    mats[2]->values = d_ftemp;

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
    idx_t blocknum_m = tilenum/DEFAULT_BLOCKSIZE + 1;
    #ifdef SGD_DEBUG
    printf("nnz %d tilenum %d\n", nnz, tilenum);
    #endif

    /* foreach epoch */
    for(idx_t e=1; e < DEFAULT_MAX_ITERATE; ++e) {

    /* update model from all training observations */
    gettimeofday(&start,NULL);
    HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaDeviceSynchronize()); 

    p_update_sgd_gpu<<<blocknum_m, DEFAULT_BLOCKSIZE, 0>>>(d_traina, d_factora, d_factorb, d_factorc, learning_rate, regularization_index, tilenum);
    HANDLE_ERROR(cudaDeviceSynchronize());

    gettimeofday(&end,NULL);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("this time cost %ld\n",diff);
    
    HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaDeviceSynchronize()); 
    #ifdef SGD_DEBUG
    printf("start display matrices\n");
    matrix_display(mats[0]);
    matrix_display(mats[1]);
    matrix_display(mats[2]);
    #endif
    gettimeofday(&end,NULL);
    diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("this time cost %ld\n",diff);
    /* compute RMSE and adjust learning rate */
    loss = tc_loss_sq(traina, mats, algorithm_index);
    frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    obj = loss + frobsq;
    if(tc_converge(traina, validation, mats, best_mats, algorithm_index, loss, frobsq, e, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs)) {
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
  cudaFree(d_directory_a);
  cudaFree(d_dims_a);
  cudaFree(d_entries_a);
  //cudaFree(d_hbuffer);
  cudaFree(d_value_a);
  cudaFree(d_value_b);
  cudaFree(d_value_c);
  cudaFree(d_traina);
  cudaFree(d_factora);
  cudaFree(d_factorb);
  cudaFree(d_factorc);

  ciss_free(h_traina);
  cudaDeviceReset();

  
}

}