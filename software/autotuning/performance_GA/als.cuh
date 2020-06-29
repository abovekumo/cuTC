#ifndef TC_ALS_H
#define TC_ALS_H

#ifdef __cplusplus
extern "C"
{
#endif 
#include "matrixprocess.h"
#include "sptensor.h"
#ifdef __cplusplus
}
#endif

//public functions
//This is for tensor completion in ALS
#ifdef __cplusplus
extern "C"
{
#endif 
idx_t tc_als(sptensor_t * traina,
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
	    int SGD_DEFAULT_BLOCKSIZE,
	    int SGD_DEFAULT_T_TILE_LENGTH,
            idx_t * nbadepochs, 
            idx_t * bestepochs, 
            idx_t * max_badepochs);

#ifdef __cplusplus
}
#endif
#endif
