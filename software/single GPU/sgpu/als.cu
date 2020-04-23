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
//#include "loss.h"



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




// gpu global function
/**
 * @brief For computing the mttkrp in als
 * @version Now only contains the atomic operation
*/
__global__ void p_mttkrp_gpu(ciss_t* d_traina, 
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
__global__ void p_mttkrp_gpu_as(ciss_t* d_traina, 
                                ordi_matrix * d_factora, 
                                ordi_matrix * d_factorb, 
                                ordi_matrix * d_factorc, 
                                double * d_hbuffer, 
                                idx_t tilenum)
{
  //set masks
  __shared__ uint32_t warpmask[((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)];
  //get block, warp and thread index
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
  printf("now the mymask and mynnz id in thread %ld are %x and %ld\n", tid, mymask, (tileid * DEFAULT_T_TILE_LENGTH + laneid));
  #endif

  if((tileid < tilenum) && ((tileid * DEFAULT_T_TILE_LENGTH + laneid)<d_traina->nnz))
  {
    //initialize the information for tile and local entry
    idx_t f_id = (idx_t)(entries[localtile] * (-1)) ;
    idx_t l_id = (idx_t)(entries[localtile+1] * (-1)) ;
    idx_t bitmap = (idx_t)(entries[localtile+2]);

    bitmap = __brevll(bitmap);
    while((bitmap & 1) == 0) {bitmap = bitmap >> 1;}
    bitmap = bitmap >> 1;
    idx_t itercounter = __popcll(bitmap) - (bitmap & 1);
    #ifdef ALSAS_DEBUG
    if(laneid == 0)
    printf("now the itercounter is %ld\n", itercounter);
    #endif
            
    idx_t myfid = f_id + laneid - __popcll((bitmap << (63-laneid))) + 1;
    #ifdef ALSAS_DEBUG
    printf("now the myfid in thread %ld is %ld\n", tid, myfid);
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
    for(int m = 0; m < DEFAULT_NFACTORS; m++)
    {
      localmbuffer[m] = d_factorb->values[b * DEFAULT_NFACTORS + m] * d_factorc->values[c * DEFAULT_NFACTORS + m];        
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
  __syncthreads();
}



/**
 * @brief Compute the inverse and finish the final update
 * @version Now only with coarse grain
*/
 __global__ void p_update_als_gpu(ciss_t * d_traina,
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
__global__ void p_update_matrice(ciss_t * d_traina,
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
    cudaSetDevice(0);
    //prepare the tensor in TB-COO
    ciss_t * h_traina = ciss_alloc(traina, 1);
    #ifdef CISS_DEBUG
    ciss_display(h_traina);
    #endif
    ciss_t * h_trainb = ciss_alloc(trainb, 2);
    ciss_t * h_trainc = ciss_alloc(trainc, 3);
    struct timeval start;
    struct timeval end;
    idx_t diff;

    //initialize the cusolver
    cusolverDnHandle_t handle;
    HANDLE_SOLVERERR(cusolverDnCreate((&handle)));

    //malloc and copy the tensors + matrices to gpu
    ciss_t * d_traina, * d_trainb, * d_trainc;
    idx_t * d_directory_a, * d_directory_b, * d_directory_c;
    idx_t * d_dims_a, * d_dims_b, * d_dims_c; 
    idx_t * d_itemp1, *d_itemp2;
    double * d_entries_a , * d_entries_b, * d_entries_c; 
    double * d_ftemp, * d_hbuffer;
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

    //buffer for HTH
    idx_t maxdlength = SS_MAX(SS_MAX(h_traina->dlength, h_trainb->dlength),h_trainc->dlength);
    //double * h_hbuffer = (double *)malloc(DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double));
    //double * h_invbuffer = (double *)malloc(DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double));
    HANDLE_ERROR(cudaMalloc((void**)&d_hbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)));
    double ** d_hbufptr, ** d_factptr; //for inverse
    int * d_infoarray;
    //HANDLE_ERROR(cudaMalloc((void**)&d_invbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)));
    //buffer for inversion
    HANDLE_ERROR(cudaMalloc((void**)&d_hbufptr, maxdlength * sizeof(double*)));
    HANDLE_ERROR(cudaMalloc((void**)&d_factptr, maxdlength * sizeof(double*)));
    HANDLE_ERROR(cudaMalloc((void**)&d_infoarray, maxdlength * sizeof(int)));
    

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


    #ifdef CUDA_LOSS //to be done
    sptensor_gpu_t * d_test, * d_validate;    
    #else
    double loss = tc_loss_sq(traina, mats, algorithm_index);
    double frobsq = tc_frob_sq(nmodes, regularization_index, mats);
    #endif
    tc_converge(traina, validation, mats, best_mats, algorithm_index, loss, frobsq, 0, nmodes, best_rmse, tolerance, nbadepochs, bestepochs, max_badepochs);

    //step into the kernel
    idx_t nnz = traina->nnz;
    idx_t tilenum = nnz/DEFAULT_T_TILE_LENGTH + 1;
    idx_t blocknum_m = tilenum/(((idx_t)DEFAULT_BLOCKSIZE)/((idx_t)ALS_WARPSIZE)) + 1;
    #ifdef ALSAS_DEBUG
    printf("the blocknum_m is %ld\n", blocknum_m);
    #endif
    idx_t blocknum_u;
    #ifdef DEBUG
    HANDLE_ERROR(cudaMemcpy(h_traina->entries, d_entries_a, h_traina->size* DEFAULT_T_TILE_WIDTH* sizeof(double), cudaMemcpyDeviceToHost));
    ciss_display(h_traina);
    #endif
    
    idx_t mode_i, mode_n, m;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    for(idx_t e=1; e < DEFAULT_MAX_ITERATE+1; ++e) {
      //HANDLE_ERROR(cudaMemcpy(d_value_a, mats[0]->values, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
      //HANDLE_ERROR(cudaMemcpy(d_value_b, mats[1]->values, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
      //HANDLE_ERROR(cudaMemcpy(d_value_c, mats[2]->values, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyHostToDevice));
      gettimeofday(&start,NULL);
      //can set random variables  
      srand(time(0));
      mode_i = rand()%3;
      for(m=0; m < nmodes; m++) {
          mode_n = (mode_i + m)%3;
          
          printf("mode_n %d nmodes %d m %d\n",mode_n, nmodes, m);
          
          HANDLE_ERROR(cudaMemset(d_hbuffer, 0, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)));
          //HANDLE_ERROR(cudaMemcpy(d_invbuffer, h_invbuffer, DEFAULT_NFACTORS * DEFAULT_NFACTORS * maxdlength * sizeof(double)),cudaMemcpyHostToDevice);
          switch (mode_n)
          {
            case 0:
              {  
                
                printf("now mode 0\n");               
                HANDLE_ERROR(cudaMemset(d_value_a, 0, mats[0]->I * DEFAULT_NFACTORS * sizeof(double)));
                blocknum_u = h_traina->dlength / DEFAULT_BLOCKSIZE + 1;
                p_mttkrp_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_traina, d_factora, d_factorb, d_factorc, d_hbuffer, tilenum);
                HANDLE_ERROR(cudaDeviceSynchronize());
                #ifdef DEBUG
                printf("mttkrp of mode 0 finishes\n");
                HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                matrix_display(mats[0]);
                #endif
                gettimeofday(&end,NULL);
                diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
                printf("this time cost %ld\n",diff);
                p_update_matrice<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(d_traina, d_value_a, d_hbuffer, d_hbufptr, d_factptr, h_traina->dlength, regularization_index); 
                #ifdef ALS_DEBUG
                p_cholecheck(d_value_a, d_hbuffer, d_hbufptr, d_factptr, h_traina->dlength);
                #endif
                HANDLE_SOLVERERR(cusolverDnDpotrfBatched(handle, uplo, DEFAULT_NFACTORS, d_hbufptr, DEFAULT_NFACTORS, d_infoarray, (int)h_traina->dlength));
                HANDLE_SOLVERERR(cusolverDnDpotrsBatched(handle, uplo, DEFAULT_NFACTORS, 1, d_hbufptr, DEFAULT_NFACTORS, d_factptr, DEFAULT_NFACTORS, d_infoarray, (int)h_traina->dlength)); 
                HANDLE_ERROR(cudaDeviceSynchronize());
                #ifdef ALS_DEBUG
                printf("mode 0 finishes\n");
                HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                matrix_display(mats[0]);
                #endif
                break;
              }
            case 1:
              {  
                
                printf("now mode 1\n");
                
                HANDLE_ERROR(cudaMemset(d_value_b, 0, mats[1]->I * DEFAULT_NFACTORS * sizeof(double)));
                blocknum_u = h_trainb->dlength / DEFAULT_BLOCKSIZE + 1;
                p_mttkrp_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainb, d_factorb, d_factorc, d_factora, d_hbuffer, tilenum);
                HANDLE_ERROR(cudaDeviceSynchronize());
                #ifdef DEBUG
                printf("mttkrp of mode 1 finishes\n");
                HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_a, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                matrix_display(mats[1]);
                #endif
                gettimeofday(&end,NULL);
                diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
                printf("this time cost %ld\n",diff);
                p_update_matrice<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(d_trainb, d_value_b, d_hbuffer, d_hbufptr, d_factptr, h_trainb->dlength, regularization_index); 
                HANDLE_SOLVERERR(cusolverDnDpotrfBatched(handle, uplo, DEFAULT_NFACTORS, d_hbufptr, DEFAULT_NFACTORS, d_infoarray, (int)h_trainb->dlength));
                HANDLE_SOLVERERR(cusolverDnDpotrsBatched(handle, uplo, DEFAULT_NFACTORS, 1, d_hbufptr, DEFAULT_NFACTORS, d_factptr,DEFAULT_NFACTORS, d_infoarray, (int)h_trainb->dlength)); 
                HANDLE_ERROR(cudaDeviceSynchronize());
                #ifdef ALS_DEBUG
                printf("mode 1 finishes\n");
                HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_a, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                matrix_display(mats[1]);
                #endif
                break;
              }
            default:
              {
                printf("now mode 2\n");
                
                HANDLE_ERROR(cudaMemset(d_value_c, 0, mats[2]->I * DEFAULT_NFACTORS * sizeof(double)));
                blocknum_u = h_trainc->dlength / DEFAULT_BLOCKSIZE + 1;
                p_mttkrp_gpu_as<<<blocknum_m,DEFAULT_BLOCKSIZE,0>>>(d_trainc, d_factorc, d_factora, d_factorb, d_hbuffer, tilenum);
                HANDLE_ERROR(cudaDeviceSynchronize());
                #ifdef DEBUG
                printf("mttkrp of mode 2 finishes\n");
                HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_a, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                matrix_display(mats[2]);
                #endif
                gettimeofday(&end,NULL);
                diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
                printf("this time cost %ld\n",diff);
                p_update_matrice<<<blocknum_u, DEFAULT_BLOCKSIZE, 0>>>(d_trainc, d_value_c, d_hbuffer, d_hbufptr, d_factptr, h_trainc->dlength, regularization_index); 
                HANDLE_SOLVERERR(cusolverDnDpotrfBatched(handle, uplo, DEFAULT_NFACTORS, d_hbufptr, DEFAULT_NFACTORS, d_infoarray, (int)h_trainc->dlength));
                HANDLE_SOLVERERR(cusolverDnDpotrsBatched(handle, uplo, DEFAULT_NFACTORS, 1, d_hbufptr, DEFAULT_NFACTORS, d_factptr,DEFAULT_NFACTORS, d_infoarray, (int)h_trainc->dlength));
                HANDLE_ERROR(cudaDeviceSynchronize()); 
                #ifdef ALS_DEBUG
                printf("mode 2 finishes\n");
                HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_a, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
                matrix_display(mats[2]);
                #endif
                break;
              }
          //p_update_als(train, mats, m, DEFAULT_NFACTORS, regularization_index);
          
          }
        }
        gettimeofday(&end,NULL);
        diff = 1000000*(end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
        printf("this time cost %ld\n",diff);
        //copy the matrices back
        HANDLE_ERROR(cudaMemcpy(mats[0]->values, d_value_a, mats[0]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(mats[1]->values, d_value_b, mats[1]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(mats[2]->values, d_value_c, mats[2]->I * DEFAULT_NFACTORS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaDeviceSynchronize()); 
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
 

    //end the cusolver
    HANDLE_SOLVERERR(cusolverDnDestroy(handle));
    //free the cudabuffer
    cudaFree(d_directory_a);
    cudaFree(d_dims_a);
    cudaFree(d_entries_a);
    cudaFree(d_directory_b);
    cudaFree(d_dims_b);
    cudaFree(d_entries_b);
    cudaFree(d_directory_c);
    cudaFree(d_dims_c);
    cudaFree(d_entries_c);
    cudaFree(d_hbuffer);
    cudaFree(d_hbufptr);
    cudaFree(d_factptr);
    cudaFree(d_infoarray);
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
