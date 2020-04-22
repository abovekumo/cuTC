#ifndef MATRIXPROCESS_H
#define MATRIXPROCESS_H

//include
#include "base.h"

//options 
//turn the math_norm_type into flags

//struct
typedef struct
{
   idx_t I;
   idx_t J;
   double* values;
   int row_col_flag;//default:row_first(1)
   
}ordi_matrix;

typedef struct
{
   idx_t I;
   idx_t J;
   idx_t nnz;
   idx_t* rowptr;
   idx_t* colid;
   double* values;
} spordi_matrix;

//functions
void matrix_display(ordi_matrix * A);

void matrix_cholesky(  ordi_matrix * A,  //cholesky factorization
                    ordi_matrix * L);

void matrix_multiply(ordi_matrix * A,    //C = AB + C(has not paralled) 
                     ordi_matrix * B,
                     ordi_matrix* C);

void matrix_syminverse(ordi_matrix* const A);  //inverse for symmetric matrix

void matrix_ata_hada(ordi_matrix** mats,    // compute hadamard product between ATA,BTB
                     idx_t const start, 
                     idx_t const end,
                     idx_t const nmats,
                     ordi_matrix* const buf,
                     ordi_matrix* const ret);
void matrix_ata(ordi_matrix * A,  //compute ATA(has not paralled)
                ordi_matrix* ret
                );

void matrix_graminverse(idx_t const mode, //(BTB * CTC)^(-1),* is hadamard product
                        idx_t const nmodes,
                        ordi_matrix** ata);

/*void matrix_normals(idx_t const mode,
                    idx_t const nmodes,
                    ordi_matrix** ata,
                    ordi_matrix * rhs,
                   double const reg);*/

void matrix_normalize(ordi_matrix* const A,  //normalize a matrix(has not paralled)
                      double *lambda,
                      idx_t mat_norm_flag ); //pay attention to which
ordi_matrix* matrix_randomize(idx_t nrows, idx_t ncols); //return a matrix with random value

ordi_matrix* matrix_zeros(idx_t nrows, idx_t ncols); //return a matrix with zero value

ordi_matrix* matrix_alloc(idx_t nrows, idx_t ncols); //return a dense matrix,need to use free

void matrix_free(ordi_matrix* mat);

void matrix_copy(ordi_matrix* omat, ordi_matrix* cmat);

spordi_matrix* spmatrix_alloc(idx_t nrows, idx_t ncols, idx_t nnz); //return a new sparse matrix,need to use free

void spmatrix_free(spordi_matrix* mat); 

ordi_matrix* matrix_change_pattern(ordi_matrix const* const mat); //copy a matrix and inverse the row/column pattern


#endif
