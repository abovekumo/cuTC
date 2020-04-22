#ifndef CISS_H
#define CISS_H

//include
#include "io.h"
#include "sptensor.h"
#include "csf.h"

/*
This is used for loading tensors in TB-COO
*/


/**
 * @brief the struct for ciss format
 * @param entries: the entries for index and tile indication + value
 * @param directories: guidance for another index
*/
typedef struct 
{
    idx_t nmodes;
    idx_t nnz;
    idx_t size; //the total size for entries
    idx_t dlength; //the length for directory
    idx_t * dims; //the actual dimension
    idx_t * directory; //list for the first dimension(mode-1)
    idx_t * dcounter; //for SGD
    double * entries; //actual elements(including indices and values)
}ciss_t;

//public function
//now for single gpu
ciss_t* ciss_alloc(
    sptensor_t * newtensor,
    idx_t mode
);

ciss_t* ciss_copy(
    ciss_t * oldtensor
);

void ciss_display(ciss_t* newtensor);

void ciss_free(ciss_t* newtensor);


#endif