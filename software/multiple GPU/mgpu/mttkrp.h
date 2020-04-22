#ifndef MTTKRP_H
#define MTTKRP_H

/*This is for the MTTKRP-serial version*/
#include "base.h"
#include "matrixprocess.h"
#include "csf.h"
#include "sptensor.h"

//structures
/*typedef struct
{
  ordi_matrix** mats;
  idx_t DEFAULT_NFACTORS;
  int mode;
  csf_sptensor* tensors;
  }mttkrp_data;*/

void mttkrp_csf(
  csf_sptensor* tensors,
  ordi_matrix** mats,
  idx_t mode,
  idx_t DEFAULT_NFACTORS

  );




#endif
