The project was tested on the university cluster, in the shortCPU queue, in order to reproduce it you need to clone the repository, upload it on the cluster and then run the .pbs file, it will automatically compile the code and run all the possible test with the bash file i created, as output two .csv file will be produced, and from them the three python scripts will generate the graphs and save them in the plots folder.

to run a single time the parallel_omp_spmv_complete.c you need to compile it with the command
gcc -O3 -fopenmp parallel_omp_spmv_complete.c mmio.c -o parallel_omp_spmv_complete
then to run it
- Loop mode : ./parallel_omp_spmv_complete matrix.mtx [threads] loop [static|dynamic|guided] [chunk]
- Task mode : ./parallel_omp_spmv_complete matrix.mtx [threads] task [chunk_rows] (to use it you need to decomment it)
- Inner mode: ./parallel_omp_spmv_complete matrix.mtx [threads] inner [chunk] (to use it you need to decomment it)
- Collapse : ./parallel_omp_spmv_complete matrix.mtx [threads] collapse (to use it you need to decomment it)

Explanation:
- The "inner" function parallelizes the inner (nonzero) loop using a reduction per row. It is mainly useful for matrices with very few rows but many nonzeros per row.
- The "collapse" functions builds a rectangular iteration space using the maximum row length and uses collapse(2) + atomic updates to y. This is intentionally heavyweight in order to test it; it typically performs worse due to atomics and wasted loop iterations. 


to run a single time the csr_blocked.c you need to comopile it with the command
gcc -O3 -fopenmp csr_blocked.c mmio.c -o csr_blocked
then to run it 
./csr_blocked matrix.mtx [threads] [block_rows] [schedule]

to run a single time the sequential_spmv.c you need to compile it with the command
gcc -O3 -fopenmp sequential_spmv.c mmio.c -o sequential_spmv
then to run it
./sequential_spmv matrix.mtx
