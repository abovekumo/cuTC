//include 

#include "matrixprocess.h"
#include <math.h>
#include "base.h"
//#include "timer.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//every public function need to add a timer
//special display function
void matrix_display(ordi_matrix * A)
{
   printf("The matrix\n");
   printf("the I is %d\n", A->I);
   printf("the J is %d\n", A->J);
   printf("the row_col_flag is %d\n", A->row_col_flag);
   idx_t I = A->I;
   idx_t J = A->J;
   idx_t i = 0;
   idx_t j = 0;
   for(i=0;i<I;i++)
 { for(j=0;j<J;j++)
   printf("the (%d,%d)th value is : %lf\n",i,j,A->values[i*J+j]);}
}


//private functions
static void p_mat_forwardsolve(
  ordi_matrix const * const L,
  ordi_matrix * const B)
{
  idx_t const N = L->I;

  double const * const lv = L->values;
  double * const  bv = B->values;
  idx_t i = 1; //define counters outside the loop
  idx_t j = 0;
  idx_t f = 0;
  for(j=0; j < N; ++j) {
    bv[j] /= lv[0];
  }

  for(i=1; i < N; ++i) {
/* X(i,f) = B(i,f) - \sum_{j=0}^{i-1} L(i,j)X(i,j) */   
    for(j=0; j < i; ++j) {
      for(f=0; f < N; ++f) {
        bv[f+(i*N)] -= lv[j+(i*N)] * bv[f+(j*N)];
      }
    }
    for(f=0; f < N; ++f) {
      bv[f+(i*N)] /= lv[i+(i*N)];
    }
  }
}

static void p_mat_backwardsolve(
  ordi_matrix const * const U,
  ordi_matrix * const B)
{
  idx_t const N = U->I;
  idx_t f = 0;  //set counters
  idx_t j = 0;
  idx_t row = 2;
  
  double const * const rv = U->values;
  double * const bv = B->values;

  /* last row of X is easy */
  for(f=0; f < N; ++f) {
    idx_t const i = N-1;
    bv[f+(i*N)] /= rv[i+(i*N)];
  }

  /* now do backward substitution */
  for(row=2; row <= N; ++row) {
    
    idx_t const i = N - row;

    /* X(i,f) = B(i,f) - \sum_{j=0}^{i-1} R(i,j)X(i,j) */
    for( j=i+1; j < N; ++j) {
      for( f=0; f < N; ++f) {
        bv[f+(i*N)] -= rv[j+(i*N)] * bv[f+(j*N)];
      }
    }
    for(f=0; f < N; ++f) {
      bv[f+(i*N)] /= rv[i+(i*N)];
    }
  }
}


static void p_mat_2norm(  ordi_matrix * const A,
                          double * const lambda
  )
{
  idx_t const I = A->I;
  idx_t const J = A->J;
  double* const  vals = A->values;
  idx_t j = 0;  //set counters
  idx_t i = 0;
 
  for(j=0; j < J; ++j) {
      lambda[j] = 0;
    }
  for(i=0; i < I; ++i) {
      for(j=0; j < J; ++j) {
      lambda[j] += vals[j + (i*J)] * vals[j + (i*J)];
      }
    }

    for(j=0; j < J; ++j) {
      lambda[j] = sqrt(lambda[j]);
    }

    for(i=0; i < I; ++i) {
      for(j=0; j < J; ++j) {
        vals[j+(i*J)] /= lambda[j];
      }
    }
  } 



static void p_mat_maxnorm(  ordi_matrix * const A,
                            double* const lambda
  )
{
  idx_t const I = A->I;
  idx_t const J = A->J;
  double* const vals = A->values;
  idx_t i = 0; //set counters
  idx_t j = 0;

  for( j=0; j < J; ++j) {
      lambda[j] = 0;
    }
  for(i=0; i < I; ++i) {
      for(j=0; j < J; ++j) {
        lambda[j] = SS_MAX(lambda[j], vals[j+(i*J)]);
      }
    }

  for(j=0; j < J; ++j) {
      lambda[j] = SS_MAX(lambda[j], 1.);
    }
  for(i=0; i < I; ++i) {
      for(j=0; j < J; ++j) {
        vals[j+(i*J)] /= lambda[j];
      }
    }
  }

static void fill_rand(double* valptr, idx_t number)
{  idx_t i = 0;
   
   for(i=0; i<number;i++)
   { valptr[i]= 1.0 * ((double) rand() / (double) RAND_MAX);
     if(rand() % 2 == 0) {
      valptr[i] *= -1;
      }
    }
  }

/**
* @brief Form the Gram matrix from A^T * A.
*
* @param[out] neq_matrix The matrix to fill.
* @param aTa The individual Gram matrices.
* @param mode Which mode we are computing for.
* @param nmodes How many total modes.
* @param reg Regularization parameter (to add to the diagonal).
*/
static void p_form_gram(
    ordi_matrix* neq_matrix,
    ordi_matrix** aTa,
    idx_t const mode,
    idx_t const nmodes,
    double const reg)
{   
    int N = aTa[0]->J;
  /* form upper-triangual normal equations */
  double* const neqs = neq_matrix->values;
      /* first initialize with 1s */
  idx_t i = 0;
  idx_t j = 0;
  idx_t m = 0;
    for(i=0; i < N; ++i) {    //can be paralleled
      neqs[i+(i*N)] = 1.0 + reg;
      for(j=0; j < N; ++j) {
        neqs[j+(i*N)] = 1.0;
      }
    }

    /* now Hadamard product all (A^T * A) matrices */
    for(m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }

      double const * const mat = aTa[m]->values;
      for(i=0; i < N; ++i) {
        /* 
         * `mat` is symmetric but stored upper right triangular, so be careful
         * to only access that.
         */

        /* copy upper triangle */
        for(j=i; j < N; ++j) {
          neqs[j+(i*N)] *= mat[j+(i*N)];
        }
      }
    } /* foreach mode */

    

    /* now copy lower triangular */
    
    for( i=0; i < N; ++i) {
      for(j=0; j < i; ++j) {
        neqs[j+(i*N)] = neqs[i+(j*N)];
      }
    }
  } 



//public functions

ordi_matrix* matrix_alloc(  idx_t const nrows,
                          idx_t const ncols)
 {
   static ordi_matrix* mat= NULL;
   mat = (ordi_matrix *) malloc(sizeof(ordi_matrix));
   mat->I = nrows;
   mat->J = ncols;
   mat->values = (double *) malloc(nrows * ncols * sizeof(double));
   mat->row_col_flag = 1;
   return mat;
 }


void matrix_cholesky(  ordi_matrix * A,
                    ordi_matrix * L)
{
  assert(A->I == A->J);
  assert(A->I == L->J);
  assert(L->I == L->J);
  idx_t const N = A->I;
  double *av = A->values;
  double *lv = L->values;
  #ifdef CHOTEST
  matrix_display(A);
  #endif
  idx_t i = 0;
  idx_t j = 0;
  idx_t k = 0;
  memset(lv, 0, N*N*sizeof(double));
  for (i = 0; i < N; ++i) {
    for (j = 0; j <= i; ++j) {
      double inner = 0;
      for (k = 0; k < j; ++k) {
        inner += lv[k+(i*N)] * lv[k+(j*N)];
          }

      if(i == j) {
        #ifdef CHOTEST
        printf("(i,j):(%d %d), av %lf\n",i,j,av[i+(i*N)]);
        #endif
        lv[j+(i*N)] = sqrt(av[i+(i*N)] - inner+0.000001);
         } else {  
           #ifdef CHOTEST
         printf("(i,j):(%d %d), av %lf, lv %lf\n",i,j,av[i+(i*N)],lv[j+j*N]);
           #endif
        lv[j+(i*N)] = 1.0 / lv[j+(j*N)] * (av[j+(i*N)] - inner);
      }
    }
  }
}


void matrix_multiply(ordi_matrix * A,    //C = AB + C 
                     ordi_matrix * B,
                     ordi_matrix* C)
{
   assert(A->J == B->I);
   assert(C->I * C->J <= A->I * B->J);
   
   C->I = A->I; //set result dimension
   C->J = B->J;
   #ifdef DEBUG
   printf("the A is\n");
   matrix_display(A);
   #endif

   double * av = A->values;
   double * bv = B->values;
   double *cv = C->values;

   idx_t  M = A->I;
   idx_t  N = B->J;
   idx_t  Na = A->J;
   
   idx_t i = 0;
   idx_t j = 0;
   idx_t k = 0;
   double lsum=0;
   for(i = 0; i<M;i++)
   {for(j=0;j<N;j++)
     { lsum = 0;
       for(k=0;k<Na;k++)
        { lsum += av[k+(i*Na)]*bv[j+(k*N)];
           }
       cv[j+(i*N)]+=lsum;
       }
     }
}

void matrix_syminverse(ordi_matrix* A)
{
   assert(A->I == A->J);
   
   idx_t const N = A->I;
   ordi_matrix * L = matrix_alloc(N,N); 
   matrix_cholesky(A,L);
   #ifdef CHOTEST
   printf("cholesky ends\n");
   #endif
   memset(A->values, 0, N*N*sizeof(double));
   idx_t n = 0;
   idx_t i = 0;
   idx_t j = 0;
   for(n=0; n<N; n++) //get identity matrix
   {A->values[n+(n*N)] = 1.0;}
   
   p_mat_forwardsolve(L,A);
 
   for(i=0; i < N; ++i) {
    for(j=i+1; j < N; ++j) {
      L->values[j+(i*N)] = L->values[i+(j*N)];
      L->values[i+(j*N)] = 0.0;
    }
  }
   p_mat_backwardsolve(L,A);
   matrix_free(L);
}

void matrix_ata_hada( ordi_matrix** mats,
                      idx_t const start,
                      idx_t const end,
                      idx_t const nmats,
                      ordi_matrix *  buf,
                      ordi_matrix *  ret)
{
  idx_t const F = mats[0]->J;

  /* check matrix dimensions */
  assert(ret->I == ret->J);
  assert(ret->I == F);
  assert(buf->I == F);
  assert(buf->J == F);
  assert(ret->values != NULL);
  assert(mats[0]->row_col_flag);
  assert(ret->row_col_flag);

  double* rv   = ret->values;
  double* bufv = buf->values;
  idx_t i = 0;    //set counter
  idx_t j = 0;
  idx_t mode = 0;
  idx_t mi = 0;
  idx_t mj = 0;
  for(i=0; i < F; ++i) {
    for(j=i; j < F; ++j) {
      rv[j+(i*F)] = 1.0;
    }
  }

  for(mode=0; mode < end; ++mode) {
    idx_t const m = (start+mode) % nmats;
    idx_t const I  = mats[m]->I;
    double const * const Av = mats[m]->values;
    memset(bufv, 0, F * F * sizeof(double));

    /* compute upper triangular matrix */
    for( i=0; i < I; ++i) {
      for(mi=0; mi < F; ++mi) {
        for(mj=mi; mj < F; ++mj) {
          bufv[mj + (mi*F)] += Av[mi + (i*F)] * Av[mj + (i*F)];
        }
      }
    }

    /* hadamard product */
    for(mi=0; mi < F; ++mi) {
      for(mj=mi; mj < F; ++mj) {
        rv[mj + (mi*F)] *= bufv[mj + (mi*F)];
      }
    }
  }

  /* copy to lower triangular matrix */
  for(i=1; i < F; ++i) {
    for(j=0; j < i; ++j) {
      rv[j + (i*F)] = rv[i + (j*F)];
    }
  }
}

void matrix_ata(  ordi_matrix * A,  //have to be paralleled
               ordi_matrix * ret
              )
{
  assert(ret->I == ret->J);
  assert(ret->I == A->J);
  assert(ret->values!=NULL);
  assert(A->row_col_flag);
  assert(ret->row_col_flag);
 
  idx_t const I = A->I;
  idx_t const F = A->J;
  idx_t i = 0;
  idx_t mi = 0;
  idx_t mj = 0;
  idx_t j = 0;
  double * Av = A->values;

  double* accum = (double*)malloc(F*F*sizeof(double));
  memset(accum,0,F*F*sizeof(double));
  for(i=0;i<I;i++)
  {
   for(mi=0;mi<F;mi++)
    {
     for(mj = mi;mj<F;mj++)
     {accum[mj+mi*F] += Av[mi + (i*F)]*Av[mj+(i*F)];}
     }
   }
  for(i=1;i<F;i++)
  {
   for(j = 0; j<i;j++)
   {accum[j+i*F] = accum[i+j*F];}
   }
  memcpy(ret->values,accum,F*F*sizeof(double));
  free(accum);
}

void matrix_graminverse(  idx_t const mode,
                          idx_t const nmodes,
                          ordi_matrix ** aTa)
{
  idx_t const rank = aTa[0]->J;
  double * av = aTa[MAX_NMODES]->values;

  idx_t x = 0;
  idx_t m = 1;
  idx_t madjust;
  double* vals;
  for(x=0; x < rank*rank; ++x) {
    av[x] = 1.0;
  }
  for(m=1; m < nmodes; ++m) {
     madjust = (mode + m) % nmodes;
     vals = aTa[madjust]->values;
    for(x=0; x < rank*rank; ++x) {
      av[x] *= vals[x];
    }
  }
  /* M2 = M2^-1 */
  matrix_syminverse(aTa[MAX_NMODES]);
  
}

/*void matrix_normals(  idx_t const mode,
                      idx_t const nmodes,
	              ordi_matrix** aTa,
                      ordi_matrix* rhs,
                      double const reg)
{
  // DEFAULT_NFACTORS 
  

  p_form_gram(aTa[MAX_NMODES], aTa, mode, nmodes, reg);

  idx_t info;
  char uplo = 'L';
  idx_t lda = N;
  idx_t ldb = N;
  idx_t order = N;
  idx_t nrhs = rhs->I;

  double * const neqs = aTa[MAX_NMODES]->values;

  // Cholesky factorization 
  bool is_spd = true;
  dpotrf_(&uplo, &order, neqs, &lda, &info);
  if(info)
     is_spd = false;
  }

  // Continue with Cholesky 
  if(is_spd) {
    // Solve against rhs 
    dpotrs_(&uplo, &order, &nrhs, neqs, &lda, rhs->values, &ldb, &info);
    if(info) {
      fprintf(stderr, "SPLATT: DPOTRS returned %d\n", info);
    }
  } else {
    // restore gram matrix 
    p_form_gram(aTa[MAX_NMODES], aTa, mode, nmodes, reg);

    idx_t effective_rank;
    double * conditions = malloc(N * sizeof(*conditions));

    // query worksize 
    idx_t lwork = -1;

    double rcond = -1.0f;

    double work_query;
    dgelss_(&N, &N, &nrhs,
        neqs, &lda,
        rhs->values, &ldb,
        conditions, &rcond, &effective_rank,
        &work_query, &lwork, &info);
    lwork = (idx_t) work_query;

    // setup workspace 
    double* work = malloc(lwork * sizeof(*work));

    // Use an SVD solver 
    dgelss_(&N, &N, &nrhs,
        neqs, &lda,
        rhs->values, &ldb,
        conditions, &rcond, &effective_rank,
        work, &lwork, &info);
    if(info) {
      printf("SPLATT: DGELSS returned %d\n", info);
    }
    printf("SPLATT:   DGELSS effective rank: %d\n", effective_rank);

    free(conditions);
    free(work);
  }

} */

void matrix_normalize(  ordi_matrix* const A,
                     double  *lambda,
                     idx_t mat_norm_flag
  )
{
  switch(mat_norm_flag) {
  case 1:
    p_mat_2norm(A, lambda);
    break;
  case 0:
    p_mat_maxnorm(A, lambda);
    break;
  default:
    fprintf(stderr,"only support 2 and max .\n");
    abort();
  }
  
}

ordi_matrix* matrix_randomize(  idx_t const nrows,
                                idx_t const ncols)
{
  static ordi_matrix * mat = NULL;
  mat = matrix_alloc(nrows, ncols);
  double* vals = mat->values;

  fill_rand(vals, nrows * ncols);
  return mat;
}

void matrix_free( ordi_matrix* mat)
{
  free(mat->values);
  free(mat);
}

void matrix_copy(ordi_matrix* omat, 
                 ordi_matrix* cmat)
{
  cmat->I = omat->I;
  cmat->J = omat->J;
  idx_t i;
  for(i = 0; i < (cmat->I) * (cmat->J); i++)
  {
    cmat->values[i] = omat->values[i];
  }
}

ordi_matrix* matrix_zeros(idx_t nrows, 
                          idx_t ncols)
{
  static ordi_matrix * mat = NULL;
  mat = matrix_alloc(nrows, ncols);
  double * vals = mat->values;

  memset(vals, 0, nrows * ncols * sizeof(double));
  return mat;
}

spordi_matrix* spmatrix_alloc(  idx_t const nrows,
                             idx_t const ncols,
                             idx_t const nnz)
{
  static spordi_matrix* mat = NULL;
  mat = (spordi_matrix*) malloc(sizeof(spordi_matrix));
  mat->I = nrows;
  mat->J = ncols;
  mat->nnz = nnz;
  mat->rowptr = (idx_t*) malloc((nrows+1) * sizeof(idx_t));
  mat->colid = (idx_t*) malloc(nnz * sizeof(idx_t));
  mat->values   = (double*) malloc(nnz * sizeof(double));
  return mat;
}

void spmatrix_free(spordi_matrix* mat)
{
  free(mat->rowptr);
  free(mat->colid);
  free(mat->values);
  free(mat);
}

ordi_matrix* matrix_change_pattern(ordi_matrix const* const mat)
{ idx_t i = 0;
  idx_t j = 0;  //set counters 
  idx_t const I = mat->I;
  idx_t const J = mat->J;

  static ordi_matrix* nmat = NULL;
  nmat = matrix_alloc(I,J);
  if(!mat->row_col_flag)
    {
      double* const rowv = nmat->values;
      double const * const colv = mat->values;

      for( i=0; i < I; ++i) {
         for(j=0; j < J; ++j) {
           rowv[j + (i*J)] = colv[i + (j*I)];
    }
  }
}
    else
    {
      double* const colv = nmat->values;
      double const * const rowv = mat->values;

      for( i=0; i < I; ++i) {
         for(j=0; j < J; ++j) {
           colv[i + (j*I)] = rowv[j + (i*J)];
    }
  }
      nmat->row_col_flag=0;
       }
    return nmat;
     }
 
