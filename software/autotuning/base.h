#ifndef BASE_H
#define BASE_H

//includes
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


//define
#define SS_MIN(x,y) ((x) < (y) ? (x) : (y))
#define SS_MAX(x,y) ((x) > (y) ? (x) : (y))
#define MAX_NMODES  4
#define MODES 3
#define CORE_COLUMN_NUM 8
#define CALCULATE_CORE_NUM 7
#define CORE_NUM (CORE_ROW_NUM * CORE_COLUMN_NUM)
#define PHISICAL_CORE_NUM 64
#define PHISICAL_CORE_ROW_NUM 8
#define PHISICAL_CORE_COLUMN_NUM 8
#define RESULT_CORE_COLUMN_ID 7
#define RESULT_CORE_NUM 1
#define RESULT_MEM_REF 6400
#define DEBUG_FLAG 0
#define CORE_GROUPS 8

//constants
typedef uint64_t idx_t;
static double const DEFAULT_ERROR = 1e-5;
static idx_t const DEFAULT_RANK = 16;
//static int const  DEFAULT_THREAD = 4;
//static int const  DEFAULT_MPI_PROCESS = 
static idx_t const DEFAULT_ITERATE = 50;


//functions
 
idx_t argmax_elem(idx_t const* const arr,
                     idx_t const N);           //scan a list and return the index of the maximum valued element.


#endif
