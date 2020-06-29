# cuTC auto-tuning

This directory contains the code that uses genetic search to tune the performance of the SGD algorithm. Convergence tuning can be achieved by changing the objective function (input and return parameters), which is not described here. The genetic algorithms can also be extended to automatic tuning in other research areas.

## Details of the genetic algorithm

### Import parameters in `config.h`:

`DEFAULT_POP_SIZE`: the size of sub-population;

`DIM`: the number of tuning parameters;

`SEARCH_RANGE`: the search range of parameters (2^n);

`MAX_COST`: related to the return value (fitness) of the objective function, generally set to the median of the return value, too small may endless loop, too large may fall into the local optimal;

`MUTATION_RATE`: the mutation rate;

`CROSSOVER_RATE`: the crossover rate;

`BUFFER_SIZE`: buffer size of the sub-population;

`DEFAULT_END_TYPE`: the end condition of the genetic algorithm;

`DEFAULT_END_GENERATION`: the maximum number of iterations.

### The interface of the objective function in `fitness.c`:

`fitness_simple()`: modify the input and return parameters of the objective function;

`binToDecimal()`: define the data type and search range of the input parameters.

### Usage of `Makefile`:

After the objective function is completed, the `Makefile` needs to be modified to compile the genetic algorithm and the objective function collaboratively.

## Build

`make`

## Run

To run the cuTC autotuning with mutiple processes (number of sub-populations):

`./run.sh`
