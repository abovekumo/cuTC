#########################################################################
# File Name: run.sh
# Author: THU Code Farmer
# mail: thu@thu.thu
# Created Time: 2019年11月28日 星期四 01时48分56秒
#########################################################################
#!/bin/bash

PROCESS=(2 4 8 16)

for((i=1;i<2;i++))
do
    mpirun -np ${PROCESS[i]} ./perf-ga
done
