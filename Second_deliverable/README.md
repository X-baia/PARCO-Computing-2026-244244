This project implements distributed Sparse Matrix–Vector Multiplication (SpMV) using MPI, with two algorithms: a required 1D cyclic (modulo) row partitioning and a bonus 2D Cartesian block partitioning. Both implementations support optional hybrid MPI+OpenMP parallelism for the local SpMV computation. Matrices are read in Matrix Market format.

To run the full project automatically, you only need to submit the provided PBS file:

qsub test_spmv.pbs

The PBS script is already configured to run on the short_cpuQ queue and requests 2 nodes with 64 CPU cores per node (128 total cores), 64 GB of memory, and a walltime of 2 hour and 30 minutes. Once submitted, the job will execute entirely on the cluster without any manual intervention.

The PBS script performs the following steps automatically: it prepares the matrix directories, compiles the 1D and 2D SpMV codes, launches the execution scripts for strong and weak scaling experiments, and finally generates plots from the collected results. The execution scripts are testing_1d_spmv.sh and testing_2d_spmv.sh, which run a comprehensive set of MPI-only and MPI+OpenMP configurations.

At the end of the job, four CSV files are produced in the project directory. These files contain the timing and performance results of all experiments:

1d_spmv_results_mpi.csv (1D MPI-only)

1d_spmv_results_hybrid.csv (1D MPI+OpenMP)

2d_spmv_results_mpi.csv (2D MPI-only)

2d_spmv_results_hybrid.csv (2D MPI+OpenMP)

In addition, a directory named "plots" is created. This directory contains all the generated performance plots, including execution time, scalability trends, and other useful performance metrics derived from the CSV files.

If you want to run a single experiment manually instead of using the PBS workflow, you can compile the programs directly. First load the required modules, then compile:

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

mpicc -O3 -std=c11 -fopenmp spmv_1d.c mmio.c -o spmv_1d -lm
mpicc -O3 -std=c11 -fopenmp spmv_2d.c mmio.c -o spmv_2d -lm

The 1D SpMV program implements the required cyclic row ownership rule owner(i) = i mod P and supports multiple matrix reading modes. Its general usage is:

mpiexec -n MPI_RANKS ./spmv_1d matrix.mtx --read=MODE [--no-omp] [--threads N]

The available read modes for the 1D implementation are:

--read=root: baseline mode where rank 0 reads the entire Matrix Market file and distributes the nonzeros to all MPI ranks (required by the assignment)

--read=parallel: parallel file reading where each rank reads a chunk of the file

--read=mpiio: MPI-IO collective reading mode

To run the 1D code in MPI-only mode, disable OpenMP using:

mpiexec -n 32 ./spmv_1d matrices/strong/webbase-1M.mtx --read=root --no-omp

To run the 1D code in hybrid MPI+OpenMP mode, either specify the number of threads explicitly:

mpiexec -n 16 ./spmv_1d matrices/strong/webbase-1M.mtx --read=parallel --threads 4

or use the OpenMP environment variable:

export OMP_NUM_THREADS=4
mpiexec -n 16 ./spmv_1d matrices/strong/webbase-1M.mtx --read=parallel

The 2D SpMV program implements a 2D Cartesian partitioning of the matrix using MPI_Dims_create and MPI_Cart_create. The process grid is created automatically based on the total number of MPI ranks. The general usage is:

mpiexec -n MPI_RANKS ./spmv_2d matrix.mtx --read=MODE [--no-omp] [--threads N]

The supported read modes for the 2D implementation are:

--read=parallel: parallel chunk-based file reading

--read=mpiio: MPI-IO collective reading

The baseline rank-0 reading mode is not implemented for the 2D algorithm; the baseline requirement is satisfied by the 1D implementation.

To run the 2D code in MPI-only mode:

mpiexec -n 64 ./spmv_2d matrices/strong/af_shell9.mtx --read=mpiio --no-omp

To run the 2D code in hybrid MPI+OpenMP mode:

mpiexec -n 16 ./spmv_2d matrices/strong/af_shell9.mtx --read=mpiio --threads 8

or equivalently:

export OMP_NUM_THREADS=8
mpiexec -n 16 ./spmv_2d matrices/strong/af_shell9.mtx --read=mpiio

When analyzing the results, note that sudden changes in execution time may occur when increasing the number of MPI ranks at fixed OpenMP thread counts. These typically correspond to transitions from single-node execution to multi-node execution, where inter-node communication becomes dominant.