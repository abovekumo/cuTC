#ifndef SPTENSOR_H
#define SPTENSOR_H


#include "base.h"
#include "matrixprocess.h"

//structure

/**
* @brief The main data structure for representing sparse tensors in
*        coordinate format.
*/
typedef struct
{
  idx_t nmodes;   /** The number of modes in the tensor, denoted 'm'. */
  idx_t nnz;      /** The number of nonzeros in the tensor. */
  idx_t* dims;   /** An array containing the dimension of each mode. */  
  idx_t** ind;   /** An m x nnz matrix containing the coordinates of each
                      nonzero. The nth nnz is accessed via ind[0][n], ind[1][n],
                      ..., ind[m][n]. */
  double* vals;   /** An array containing the values of each nonzero. */
  // int tiled;      /** Whether sptensor_t has been tiled. Used by ftensor_t. */

  //double* indmap[MAX_NMODES]; /** Maps local -> global indices. */
} sptensor_t;

//for gpu
typedef struct
{
  idx_t nmodes;   /** The number of modes in the tensor, denoted 'm'. */
  idx_t nnz;      /** The number of nonzeros in the tensor. */
  idx_t* dims;   /** An array containing the dimension of each mode. */  
  idx_t* find;   /** An m x nnz matrix containing the coordinates of each
                      nonzero. The nth nnz is accessed via ind[0][n], ind[1][n],
                      ..., ind[m][n]. */
  idx_t* sind;
  idx_t* tind;
  double* vals;   /** An array containing the values of each nonzero. */
  // int tiled;      /** Whether sptensor_t has been tiled. Used by ftensor_t. */

  //double* indmap[MAX_NMODES]; /** Maps local -> global indices. */
} sptensor_gpu_t;






//public functions
sptensor_t * tt_read(
  char const * const ifname);


sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes);



void tt_fill(
  sptensor_t * const tt,
  idx_t const nnz,
  idx_t const nmodes,
  idx_t** const inds,
  double* const vals);

sptensor_t * tt_copy(
  sptensor_t* oldtensor
);



/**
* @brief Return a histogram counting nonzeros appearing in indices of a given
*        mode.
*
* @param tt The sparse tensor to make a histogram from.
* @param mode Which mode we are counting.
*
* @return An array of length tt->dims[m].
*/
idx_t* tt_get_hist(
  sptensor_t const * const tt,
  idx_t const mode);



void tt_free(
  sptensor_t * tt);


/**
* @brief Compute the density of a sparse tensor, defined by nnz/(I*J*K).
*
* @param tt The sparse tensor.
*
* @return The density of tt.
*/
double tt_density(
  sptensor_t const * const tt);

/**
* @brief Calculate the Frobenius norm of tt, squared. This is the
*        sum-of-squares for all nonzero values.
*
* @param tv The tensor values to operate on.
*
* @return  The squared Frobenius norm.
*/
double tt_normsq(
  sptensor_t const * const tt);

#endif



