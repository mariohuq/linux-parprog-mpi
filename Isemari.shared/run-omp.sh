#!/bin/sh
gcc --version
echo |cpp -fopenmp -dM |grep -i open
echo ""
gcc -o hello -fopenmp hello-omp.c
export OMP_NUM_THREADS=5
./hello