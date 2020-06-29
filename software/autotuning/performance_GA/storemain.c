//include
#include "base.h"
#include "csf.h"
#include "matrixprocess.h"
#include "sptensor.h"
#include "io.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

int main(int argc, char** argv)
{
   int flag;
   idx_t nfactors = 16;
   idx_t nmodes = 3;
   int j = 0;
  static csf_sptensor* newtensora;
  static csf_sptensor* newtensorb;
  static csf_sptensor* newtensorc;
   newtensora = (csf_sptensor*)malloc(sizeof(csf_sptensor));
   newtensorb = (csf_sptensor*)malloc(sizeof(csf_sptensor));
   newtensorc = (csf_sptensor*)malloc(sizeof(csf_sptensor));
   //idx_t blockref[3][9];
  static idx_t** blockref;
   blockref = (idx_t**)malloc(3*sizeof(idx_t*));
   for(j=0;j<3;j++)
   blockref[j]=(idx_t*)malloc(9*sizeof(idx_t));
   printf("%s\n",argv[1]);
   flag = csf_tensor_load(argv[1], &nmodes, newtensora, newtensorb,newtensorc,blockref);
   if(flag ==0) csf_display(newtensora, blockref[0]);
   else printf("something is wrong in the load process");
   flag = csf_tensor_free(newtensora);
   flag = csf_tensor_free(newtensorb);
   flag = csf_tensor_free(newtensorc);
   if(flag==0) printf("free the tensor successfully!");
   else printf("something is wrong in the free process");
   for(j=0;j<3;j++)
   free(blockref[j]);
   free(blockref);
  return 0;
}

