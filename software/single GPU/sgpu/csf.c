//include
#include "base.h"
#include "csf.h"
#include "io.h"
#include "sptensor.h"
#include "util.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
//private functions
static void p_modecounter(csf_sptensor* ct,
                           sptensor_t* tt,
                           idx_t mode,
                           idx_t * blockref)
{ csf_sparsity*  pt = ct->pt;
  idx_t nnz = tt->nnz;
  idx_t tmp0_length;
  tmp0_length = tt->dims[mode-1];
  idx_t* tmparray0 = (idx_t*)malloc((tmp0_length+1) * sizeof(idx_t));   //workspace
  idx_t* tmparray1 = (idx_t*)malloc((nnz+1) * sizeof(idx_t));
  idx_t* tmparray2 = (idx_t*)malloc((nnz+1) * sizeof(idx_t));
  idx_t tmp0=0;
  idx_t tmp1=0;
  idx_t i = 0;
  idx_t i_counter = 1;
  idx_t k_counter = 0;
  idx_t firstrank = mode - 1;
  idx_t secondrank,thirdrank;
  idx_t refnum = nnz / (CORE_GROUPS);
  #ifdef DEBUG
  printf("refnum %d\n",refnum);
  printf("mode %d\n", mode);
  #endif
  idx_t refcount = 0;
  if(mode == 1) {secondrank = 1; thirdrank = 2;}
  else {
      if(mode == 2){secondrank = 0; thirdrank = 2;}
      else {secondrank = 0; thirdrank = 1;}}
  memset(tmparray0, 0, (tmp0_length+1)*sizeof(idx_t));
  memset(tmparray1, 0, (nnz+1)*sizeof(idx_t));
  memset(tmparray2, 0, (nnz+1)*sizeof(idx_t));
  tmp0=tt->ind[firstrank][0];
  tmparray0[tmp0-1]=1;
  for(i = 1;i<nnz;i++)    //first mode
  { tmp1 = tt->ind[firstrank][i];                  //can be paralleled
    if(tmp0!=tmp1) i_counter++;
    tmp0=tmp1;
    tmparray0[tmp0-1]++;
    }
  pt->nfibs[0]=i_counter;
  pt->fids[0]=(idx_t*)malloc(i_counter*sizeof(idx_t));
  pt->fptr[0]=(idx_t*)malloc((i_counter+1)*sizeof(idx_t));
  
  idx_t* anewptr1 = pt->fids[0];
  idx_t* anewptr2 = tmparray0;
  for(i=0; i<tmp0_length+1;i++) 
  { if(tmparray0[i]!=0)
    { *anewptr1 = i+1;
       anewptr1++;
      *anewptr2 = tmparray0[i];
       anewptr2++; 
       }
       }
       
  //for splatt:there are two situations:if the i_counter<64, there is no need to divide.
  //now only consider evenly divide i
  if(i_counter%64==0) refnum = i_counter/64;
  else refnum = i_counter / 64 +1;
  if(i_counter<64)
  {memset(blockref,0,65*sizeof(idx_t));
    for(i=0;i<i_counter;i++)
    blockref[i] = i;
    blockref[i] = i_counter;
  } //initialize the blockreference
  else
  {
    memset(blockref,0,65*sizeof(idx_t));
    for(i=1;i<65;i++)
    {if(refnum*i<i_counter)    blockref[i] = refnum*i;
     else {blockref[i]=i_counter;break;}
    }
    
  }
  tmp1 = 0;
  tmp0 = 0; //to store the former data
  anewptr2 = pt->fptr[0];
  anewptr1 = tmparray2;    //to store k_index(fids[1])
 
  idx_t* anewptr3 = tmparray1; //to store k_ptr(fptr[1]),the frequency of each k_index(i not change)
  idx_t sum = 0; //add sum to avoid buffer overload
  idx_t j = 0;
  idx_t ncounter = 0; //range from 1 to nnz
  idx_t layercounter = 0;//for every i, how many kinds of k are there
  idx_t oldlayercounter = 0; //store the counter in last layer
  for(i=0;i<i_counter;i++) 
  { oldlayercounter = layercounter;
    layercounter = 0;
    ncounter = sum;
    tmp0 = tt->ind[secondrank][ncounter];//as the i change,the tmp0 need to initialize again
    k_counter++;
    /*if(ncounter % refnum == 0) {blockref[refcount] = i;refcount++;
                                if(refcount>1 && blockref[refcount-1] == blockref[refcount-2]) blockref[refcount-2]=i-1;}*/
    ncounter++;
    layercounter++;
    *anewptr1 = tmp0;
    anewptr1++;
    (*anewptr3)++;
     for(j=1;j<tmparray0[i];j++)                //how many kinds of k are there totally
    { tmp1 = tt->ind[secondrank][ncounter];
      if(tmp0!=tmp1)
      {
        k_counter++;
        layercounter++;
        *anewptr1 = tmp1;
         anewptr1++;
        anewptr3++;
       }
     (*anewptr3)++;
     tmp0 = tmp1;
     /*if(ncounter % refnum == 0) {blockref[refcount] = i;refcount++;
                                 if(refcount>1 && blockref[refcount-1] == blockref[refcount-2] )blockref[refcount-2]= i-1;}*/
     ncounter++;
       } 
    sum += tmparray0[i];

    anewptr3++;
     if(i == 0){*anewptr2 = 0;anewptr2++;}
    else{*(anewptr2) = *(anewptr2-1)+oldlayercounter;anewptr2++;}
               
    }
  pt->fptr[1] = (idx_t*)malloc((k_counter+1)*sizeof(idx_t));  //allocate
  pt->fids[1] = (idx_t*)malloc(k_counter*sizeof(idx_t));
  pt->nfibs[1] = k_counter; 
  memcpy(pt->fids[1],tmparray2,k_counter*sizeof(idx_t));//copy the k_index
  anewptr1 = pt->fptr[1];
  pt->fptr[0][i_counter] = pt->nfibs[1];
  pt->fptr[1][pt->nfibs[1]] = pt->nfibs[2];

  for(i = 0; i<k_counter;i++)  //copy the k_ptr
  {if(i==0) *anewptr1 = 0;
   else *anewptr1 = tmparray1[i-1]+*(anewptr1-1);
   anewptr1++;}
  free(tmparray0);
  free(tmparray1);
  free(tmparray2);
 
 }

static void p_rearrange(sptensor_t* tt, idx_t mode)
{ idx_t nnz = tt->nnz;  //mode1: i,j,k; mode2:j,i,k; mode3:k,i,j
  idx_t tmp0_length = tt->dims[mode-1];
  idx_t * temparray0 =(idx_t*)malloc(tmp0_length * sizeof(idx_t));
  idx_t * temparray1 =(idx_t*)malloc(tmp0_length * sizeof(idx_t));
  memset(temparray0, 0, tmp0_length * sizeof(idx_t));
  memset(temparray1, 0, tmp0_length * sizeof(idx_t));
  idx_t sortnum = mode - 1;
  idx_t i;

  if(mode == 1) {free(temparray0);free(temparray1);return;}
  else
  { if(mode == 2)
    {
     quicksort(tt->ind[1], tt->ind[0], tt->ind[2], tt->vals, 0, nnz-1);
     }
   else quicksort(tt->ind[2], tt->ind[0], tt->ind[1], tt->vals, 0, nnz-1);
   }
  for(i=0;i<nnz;i++)
  {
    temparray0[(tt->ind)[sortnum][i]-1]++;
    
  }
  idx_t count = 0;
  temparray1[count] = 0;
  count++;
  idx_t j;
  for(i=0;i<tmp0_length;i++)
  {
    if(temparray0[i]!=0) {temparray1[count] = temparray0[i]+temparray1[count-1];count++;/*printf("temparray1[%d] %d\n",count-1,temparray1[count-1]);*/}

   }
  free(temparray0);
  
  for(i=0;i<count-1;i++)
  {
    quicksort(tt->ind[0],tt->ind[1],tt->ind[2],tt->vals,temparray1[i],temparray1[i+1]-1);
    }
  free(temparray1);  
  }


static void p_mk_fptr(
  csf_sptensor *  ct,
  sptensor_t* tt,
  idx_t  mode,
  idx_t* blockref)
{idx_t  nnz = tt->nnz;  //need to write again(need to add tile)
 idx_t i=0; 
 for(i=0;i<3;i++)
 { ct->dims[i] = tt->dims[i];
   }
 csf_sparsity* pt = ct->pt;
 p_rearrange(tt,mode);  //mode1:i,j,k;mode2: j,i,k; mode3:k, i, j
 size_t num = nnz;
 memcpy(pt->values,tt->vals,num*sizeof(double));
 if(mode == 1) memcpy(pt->fids[2], tt->ind[2] , nnz*sizeof(idx_t));
 else{
   if(mode == 2) memcpy(pt->fids[2], tt->ind[2], nnz*sizeof(idx_t));
   else memcpy(pt->fids[2], tt->ind[1], nnz*sizeof(idx_t));
}
 p_modecounter(ct,tt,mode,blockref);
}

void csf_display(csf_sptensor* ct,idx_t* blockref)
{
  idx_t nnz = ct->nnz;
  idx_t nmodes = ct->nmodes;
  idx_t i = 0;
  for(i=0;i<nmodes;i++)  printf("dims[%d]:%d\n",i,ct->dims[i]);
  csf_sparsity* pt = ct->pt;
  for(i=0;i<nmodes;i++) printf("nfibs[%d]:%d\n",i,pt->nfibs[i]);
  for(i=0;i<pt->nfibs[0];i++)
  {printf("i_ptr[%d]:%d\n",i,pt->fptr[0][i]);
   printf("i_ind[%d]:%d\n",i,pt->fids[0][i]);}
   printf("i_ptr[pt->nfibs[0]] %d", pt->fptr[0][pt->nfibs[0]]);
  for(i=0;i<pt->nfibs[1];i++)
  {printf("j_ptr[%d]:%d\n",i,pt->fptr[1][i]);
   printf("j_ind[%d]:%d\n",i,pt->fids[1][i]);}
   printf("j_ptr[pt->nfibs[1]] %d", pt->fptr[1][pt->nfibs[1]]);
  for(i=0;i<pt->nfibs[2];i++)
  {printf("k_ind[%d]:%d\n",i,pt->fids[2][i]);
   printf("the value[%d]:%lf\n",i,pt->values[i]);}
   for(i=0;i<65;i++)
  {
   printf("blockref[%d]:%d\n",i,blockref[i]);
  }
   }



static void csf_alloc(
  csf_sptensor* ct,
  sptensor_t* tt,
  idx_t mode,
  idx_t* blockref)
{ idx_t nmodes = tt->nmodes;
  ct->nmodes = nmodes;
  ct->nnz = tt->nnz;
  ct->pt = (csf_sparsity*)malloc(sizeof(*(ct->pt)));
  csf_sparsity * pt = ct->pt;
  /* last row of fptr is just nonzero inds */
  pt->nfibs[nmodes-1] = ct->nnz;
  pt->fids[nmodes-1] = (idx_t*)malloc(ct->nnz * sizeof(idx_t));
  pt->values = (double*)malloc(ct->nnz * sizeof(double));
  //process the j_index and values first
  p_mk_fptr(ct, tt, mode, blockref);
  
 }





//public functions
//first the display function
void ktensor_display(ktensors* aktensor)
{
  printf("the rank is %d\n",aktensor->rank);
  printf("the nmodes is %d\n",aktensor->nmodes);
  printf("the fit is %lf\n",aktensor->fit);
}

int csf_tensor_load(char const* const fname,
                    idx_t* nmodes,
                    csf_sptensor* atensora,
                    csf_sptensor* atensorb,
                    csf_sptensor* atensorc,
                    idx_t** blockref)
{
  sptensor_t * tt = tt_read(fname);
  if(tt == NULL) {
    return -1;
  }
  csf_alloc(atensora, tt, 1, blockref[0]);
  csf_alloc(atensorb, tt, 2, blockref[1]);
  csf_alloc(atensorc, tt, 3, blockref[2]);
  *nmodes = tt->nmodes;
  tt_free(tt);
  return 0;
 }


int csf_tensor_free(csf_sptensor* atensor)
{  idx_t i=0;
   free((atensor->pt)->values);
   for(i=0;i<2;i++)
   { free((atensor->pt)->fptr[i]); }
   for(i=0;i<3;i++)
   { free((atensor->pt)->fids[i]);}
   free(atensor);
   return 0;
}


double csf_frobsq(
    csf_sptensor const * atensor)
{
  /* accumulate into double to help with some precision loss */
  double norm = 0;
  idx_t n=0;
   //can be paralleled
  double const * const vals = (atensor->pt)->values;
  idx_t const nnz = (atensor->pt)->nfibs[atensor->nmodes-1];
  //can be paralleled
  for(n=0; n < nnz; ++n) {
        norm += vals[n] * vals[n];
      }
    


  return norm;
}

void ktensor_free(ktensors* aktensor)
{ idx_t i = 0;
  free(aktensor->lambda);
  idx_t nmodes = aktensor->nmodes;
  for(i=0;i<nmodes;i++)
  {free(aktensor->factors[i]);
   }
  
}
