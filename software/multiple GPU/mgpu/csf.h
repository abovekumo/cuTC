#ifndef CSF_H
#define CSF_H

//include
#include "base.h"



//data structures
typedef struct ktensors  //CPD output
{
  idx_t rank;   //rank of the decomposition
  double* factors[MAX_NMODES];  //row-major matrix factors for each mode
  double* lambda;
  idx_t nmodes;  //number of modes in the tensor
  idx_t dims[MAX_NMODES];  //number of rows in each factor
  double fit;  //the error of CPD
  
  }ktensors;

typedef struct
{
  idx_t nfibs[MAX_NMODES]; //the size of each fptr and fids array
  idx_t* fptr[MAX_NMODES-1]; //the start of a fid sub-tree.Like the k-pointer in documentation fptr[nmode-1]may not need
  idx_t* fids[MAX_NMODES]; //the index of node. Like the j_index in documentation
  double* values; //the nnz values.
}csf_sparsity;

typedef struct
{
  idx_t nnz; //number of nonzeros
  idx_t nmodes; //number of modes
  idx_t dims[MAX_NMODES]; //dimension of each mode
  //idx_t ntiles; how many tiles are there
  //idx_t ntiled_modes;
  //idx_t tile_dims[MAX_NMODES];
  csf_sparsity* pt; //sparsity structures(need to be pointer when paralleled)
  
}csf_sptensor;

//functions
void csf_display(csf_sptensor* ct, idx_t* blockref);
int csf_tensor_load(char const* const fname,
                    idx_t* nmodes,
                    csf_sptensor* atensora,
                    csf_sptensor* atensorb,
                    csf_sptensor* atensorc,
                    idx_t** blockref);

int csf_tensor_free(csf_sptensor* atensor);

size_t csf_storage(csf_sptensor const* const tensors, 
                   double const* const opts);

double csf_frobsq(csf_sptensor const * atensor);
void ktensor_free(ktensors* aktensor);
void ktensor_display(ktensors* aktensor);

#endif


