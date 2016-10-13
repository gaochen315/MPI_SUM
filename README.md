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
  * Compile MPI_SUM.cpp `mpic++ -std=c++11 -O3 MPI_SUM.cpp -o t`
  * `mpirun -np 36 t 1000 4000` Run in parallel
  * 36 is the # proc you want to use
  * 1000 4000 are the size of matrix to sum. 
  
2. If you are running on HPC cluster(e.g. Flux@Umich)
  * Compile MPI_SUM.cpp `mpic++ -std=c++11 -O3 MPI_SUM.cpp -o t`
  * Modify MPI_SUM.PBS 
  * line 4, specify cluster
  * line 6, specify # proc required and running time limit
  * line 16, specify # proc to use and matrix size
  * `qsub MPI_SUM.PBS`
  * When finish, you will find a result domument in current folder.

