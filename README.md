# MPI_SUM
Parallel Computing via MPI. Sum the matrix.

## Requirements

- Environment 
  - Linux

- Preparation 

  - MPI

  You can get a free MPI implementation at 
  http://www.open-mpi.org/ (Links to an external site.) 
  
  - Shared, Linux-based high-performance computing (HPC) cluster

## How to use

1. If you are running on regular computer
  * Compile MPI_SUM.cpp `mpic++ -std=c++11 -O2 MPI_SUM.cpp -o t`
  * `mpirun -np 36 t 1000 4000` Run in parallel
  * 36 is the # proc you want to use
  * 1000 4000 are the size of matrix to sum. 


