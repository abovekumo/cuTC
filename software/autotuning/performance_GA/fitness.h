#ifndef FITNESS_H_
#define FITNESS_H_
#include "types.h"
#include "base.h"

void 	fitness(deme*);
void	fitness_simple(deme*);
void	fitness_shpath(deme*);
double	pt_dist(point*, point*);
int		collision(point*, point*, object*);
int		valid_loc(point*);
void	pt_copy(point*, point*);
point   **make_path(char*, point*, point*);
void    free_path(point**);
int 	binToDecimal(char*, unsigned int, unsigned int);

idx_t tc_main(char* input_traina, char* input_trainb, char* input_trainc, char* input_validation, char *input_test, int SGD_DEFAULT_BLOCKSIZE, int SGD_DEFAULT_T_TILE_LENGTH);

#endif

