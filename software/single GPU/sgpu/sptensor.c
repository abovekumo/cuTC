#include <stdio.h>
#include "sptensor.h"
#include "matrixprocess.h"
#include "io.h"
//#include "timer.h"
#include <math.h>
#include <stdlib.h>


//private functions
static inline int p_same_coord(
  sptensor_t const * const tt,
  idx_t const i,
  idx_t const j)
{
  idx_t m = 0;
  idx_t const nmodes = tt->nmodes;
  if(nmodes == 3) {
    return (tt->ind[0][i] == tt->ind[0][j]) &&
           (tt->ind[1][i] == tt->ind[1][j]) &&
           (tt->ind[2][i] == tt->ind[2][j]);
  } else {
    for(m=0; m < nmodes; ++m) {
      if(tt->ind[m][i] != tt->ind[m][j]) {
        return 0;
      }
    }
    return 1;
  }
}




//public functions

double tt_normsq(sptensor_t const * const tt)
{
  double norm = 0.0;
  double const * const tv = tt->vals;
  idx_t n = 0;
  for(n=0; n < tt->nnz; ++n) {
    norm += tv[n] * tv[n];
  }
  return norm;
}


double tt_density(
  sptensor_t const * const tt)
{
  double root = pow((double)tt->nnz, 1./(double)tt->nmodes);
  double density = 1.0;
  idx_t m = 0;
  for(m=0; m < tt->nmodes; ++m) {
    density *= root / (double)tt->dims[m];
  }

  return density;
}


//public function
sptensor_t * tt_read(
  char const * const ifname)
{
  static sptensor_t* tt;
  tt  = tt_read_file(ifname);
  return tt;
}


sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes)
{
  idx_t m = 0;
  sptensor_t * tt = (sptensor_t*) malloc(sizeof(sptensor_t));
  //tt->tiled = SPLATT_NOTILE;
  printf("sptensor malloc ok!\n");
  tt->nnz = nnz;
  tt->vals = (double*)malloc(nnz * sizeof(double));
  printf("value malloc ok!\n");
  tt->nmodes = nmodes;
  tt->dims = (idx_t*)malloc(nmodes * sizeof(idx_t));
  printf("dims malloc ok!\n");
  tt->ind  = (idx_t**)malloc(nmodes * sizeof(idx_t*));
  printf("ind first malloc ok!\n");
  for(m=0; m < nmodes; ++m) {
    tt->ind[m] = (idx_t*)malloc(nnz * sizeof(idx_t));
    printf("ind second malloc ok!\n");
  }

  return tt;
}

sptensor_t * tt_copy(
             sptensor_t * oldtensor)
{
  sptensor_t * newtensor = tt_alloc(oldtensor->nnz, oldtensor->nmodes);
  idx_t m,i,j,k;
  for(m=0;m<oldtensor->nmodes;m++)
  {
    newtensor->dims[m] = oldtensor->dims[m];
  }
  for(m=0;m<oldtensor->nmodes;m++)
  {
    for(i=0;i<oldtensor->nnz;i++)
    {
      newtensor->ind[m][i] = oldtensor->ind[m][i];
      newtensor->vals[i] = oldtensor->vals[i];
    }
  }
  return newtensor;
}


void tt_fill(
  sptensor_t * const tt,
  idx_t const nnz,
  idx_t const nmodes,
  idx_t ** const inds,
  double* const vals)

{
  tt->nnz = nnz;
  tt->vals = vals;
  tt->ind = inds;

  tt->nmodes = nmodes;
  tt->dims = (idx_t*)malloc(nmodes * sizeof(*tt->dims));
  idx_t m = 0;
  idx_t i = 0;

  for(m=0; m < nmodes; ++m) {
    tt->dims[m] = 1 + inds[m][0];
    for(i=1; i < nnz; ++i) {
      tt->dims[m] = SS_MAX(tt->dims[m], 1 + inds[m][i]);
    }
  }
}



void tt_free(
  sptensor_t * tt)
{
  tt->nnz = 0;
  idx_t m = 0;
  for( m=0; m < tt->nmodes; ++m) {
    free(tt->ind[m]);
    
  }
  tt->nmodes = 0;
  free(tt->dims);
  free(tt->ind);
  free(tt->vals);
  free(tt);
}





