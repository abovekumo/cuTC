/*This is for the serial MTTKRP computation*/

//include
#include "mttkrp.h"
#include "base.h"
#include "csf.h"
#include "matrixprocess.h"
//#include "timer.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


//public functions
void mttkrp_csf(
  csf_sptensor* tensors,  //tensor:the tensor needed to be factorized; mats:factor matrix
  ordi_matrix** mats,               //mode: the processing mode
  idx_t  mode,
  idx_t  DEFAULT_NFACTORS
  
  )
{  
  #ifdef DEBUG
  printf("DEFAULT_NFACTORS, mode %d %d\n", DEFAULT_NFACTORS, mode); 
  printf("the rank is %d\n",DEFAULT_NFACTORS);
  #endif
  //intialziation
  idx_t firstrank, secondrank,thirdrank;
  if(mode==0) 
    {
      firstrank = 0;
      secondrank = 1;
      thirdrank = 2;
    }
    else
    {
      if(mode==1)
      {
        firstrank = 1;
        secondrank = 0;
        thirdrank = 2;
      }
      else
      {
        firstrank = 2;
        secondrank = 0;
        thirdrank = 1;
      }
    }
  ordi_matrix * M = mats[4];
  double* sum = (double*)malloc(DEFAULT_NFACTORS*sizeof(double));
  csf_sparsity* pt = tensors->pt;
  idx_t i_counter = pt->nfibs[0];
  idx_t j_counter = pt->nfibs[1];
  idx_t nnz = tensors->nnz;
  double* tvalues = pt->values;
  double* bvalues = mats[secondrank]->values;
  double* cvalues = mats[thirdrank]->values;
  double* avalues = M->values;

  idx_t i,j,k,r;
  idx_t i_id,j_id,k_id;
  idx_t j_hedge,j_ledge,k_ledge,k_hedge;

  for(i=0;i<i_counter;i++)
  {
    i_id = pt->fids[0][i]-1;
    j_ledge = pt->fptr[0][i];
    j_hedge = pt->fptr[0][i+1];
    for(j=j_ledge;j<j_hedge;j++)
    {
      memset(sum,0,DEFAULT_NFACTORS*sizeof(double));
      k_ledge = pt->fptr[1][j];
      k_hedge = pt->fptr[1][j+1];
      for(k=k_ledge;k<k_hedge;k++)
      {
        j_id = pt->fids[1][j]-1;
        k_id = pt->fids[2][k]-1;
        for(r=0;r<DEFAULT_NFACTORS;r++)
        {
          sum[r]+=tvalues[k]*cvalues[k_id*DEFAULT_NFACTORS+r];
        }
      }
      for(r=0;r<DEFAULT_NFACTORS;r++)
      {
        avalues[i_id*DEFAULT_NFACTORS+r] += sum[r]*bvalues[j_id*DEFAULT_NFACTORS+r];
      }
    }
  }
  free(sumbuffer);
     
}




