#include "base.h"



idx_t argmax_elem(idx_t const* const arr,
                     idx_t const N)
{
   idx_t mkr=0;
   idx_t i=1;
   for(i=1;i<N;i++)
   {
     if(arr[i] > arr[mkr])  mkr=i;
      }
   return mkr;
}

