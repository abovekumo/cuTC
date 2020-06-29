#ifndef COMPLETION_H
#define COMPLETION_H

#include <stdbool.h>
#include "sptensor.h"
#include "matrixprocess.h"

/*
This is for the main function in completion and CPU version in LOSS
*/

double tc_rmse(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    );

double tc_loss_sq(
    sptensor_t * test,
    ordi_matrix ** mats,
    int algorithm_index
    );

double tc_frob_sq(
    idx_t nmodes,
    double regularization_index,
    ordi_matrix ** mats);

double tc_predict_val(
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index
    );

double tc_predict_val_col(
    sptensor_t * test,
    ordi_matrix ** mats,
    idx_t const index
    );

bool tc_converge(
    sptensor_t * train,
    sptensor_t * validate,
    ordi_matrix ** mats,
    ordi_matrix ** best_mats,
    int algorithm_index,
    double const loss,
    double const frobsq,
    idx_t const epoch,
    idx_t nmodes,
    double * best_rmse,
    double * tolerance,
    idx_t * nbadepochs,
    idx_t * bestepochs,
    idx_t * max_badepochs
    );

//for sgd
double tc_rmse_sgd(
    sptensor_t * test,
    ordi_matrix ** mats,
    ordi_matrix ** aux_mats,
    int algorithm_index
    );

double tc_loss_sq_sgd(
    sptensor_t * test,
    ordi_matrix ** mats,
    ordi_matrix ** aux_mats,
    int algorithm_index
    );

double tc_frob_sq_sgd(
    idx_t nmodes,
    double regularization_index,
    ordi_matrix ** mats,
    ordi_matrix ** aux_mats);

double tc_predict_val_sgd(
    sptensor_t * test,
    ordi_matrix ** mats,
    ordi_matrix ** aux_mats,
    idx_t const index
    );

double tc_predict_val_col_sgd(
    sptensor_t * test,
    ordi_matrix ** mats,
    ordi_matrix ** aux_mats,
    idx_t const index
    );

bool tc_converge_sgd(
    sptensor_t * train,
    sptensor_t * validate,
    ordi_matrix ** mats,
    ordi_matrix ** best_mats,
    ordi_matrix ** aux_mats,
    int algorithm_index,
    double const loss,
    double const frobsq,
    idx_t const epoch,
    idx_t nmodes,
    double * best_rmse,
    double * tolerance,
    idx_t * nbadepochs,
    idx_t * bestepochs,
    idx_t * max_badepochs
    );

/**
 * @brief The main function for tensor completion under TB-COO(in ciss.h)
*/
idx_t tc_main_ciss(sptensor_t* traina, 
             sptensor_t* trainb,
             sptensor_t* trainc,
             sptensor_t* validation, 
             sptensor_t* test,
             int algorithm_index,
	     int SGD_DEFAULT_BLOCKSIZE,
	     int SGD_DEFAULT_T_TILE_LENGTH
             );

#endif
