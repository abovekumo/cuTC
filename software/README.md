# cuTC

## Build

### 1. Prepare the train, validation and test datasets

The input tensor need to be in .tns format. To generate the train, validation and test dataset from input, please run the following scripts in /script:

`./tvtensor.py`

`./stensor.py`

Please do not forget to change the directory and name of the input tensor in those two scripts.

### 2. Build the cuTC for single GPU

To build the cuTC for single GPU in /single GPU/sgpu:

`make`

### 3. Build the cuTC for multiple GPUs

To build the cuTC for multiple GPUs in /multiple GPU/mgpu:

`make`

### 4. Build the auto-tuning

## Run

### 1. Run the cuTC for single GPU

To run the cuTC for single GPU in /single GPU/sgpu:

`./run.sh`

Both the information for execution time and RMSE-vl is in /log, and the unit of execution time is microseconds.

Please do not forget to change the name of algorithm type and tensor name in run.sh

### 2. Run the cuTC for multiple GPUs

To run the cuTC for multiple GPUs in /multiple GPU/mgpu:

`./run.sh`

Both the information for execution time and RMSE-vl is in /log, and the unit of execution time is microseconds.

Please do not forget to change the name of algorithm type and tensor name in run.sh

### 3. Run the auto-tuning



## Change the parameters

### 1. Rank

To change rank, please change the DEFAULT_NFACTORS in base.h

### 2. Algorithm Type

To change Algorithm Type, please change the algorithm_index in tc.c (0:ALS; 1: SGD; 2: CCD+)