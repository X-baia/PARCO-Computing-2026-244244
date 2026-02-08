#!/bin/bash

############################################
# EXECUTABLE AND OUTPUTS
############################################
EXEC=./spmv_2d
CSV_MPI=2d_spmv_results_mpi.csv
CSV_HYB=2d_spmv_results_hybrid.csv

############################################
# MATRICES
############################################
STRONG_MATRICES=(
  matrices/strong/af_shell9.mtx
  matrices/strong/thermal2.mtx
  matrices/strong/poisson.mtx
  matrices/strong/amazon0312.mtx
  matrices/strong/ecology2.mtx
  matrices/strong/webbase-1M.mtx
)

WEAK_DIR=matrices/weak
WEAK_PREFIX=poisson_weak_

############################################
# PARAMETERS
############################################
MPI_RANKS=(1 2 4 8 16 32 64 128)

# Hybrid OMP values (exclude 1; MPI-only handled separately via --no-omp)
OMP_THREADS_HYB=(2 4 8)

WEAK_OMP=4

# 2D code uses parallel reading only; keep column for consistency
READ_MODE_FIXED="mpiio"

############################################
# DETECT ALLOCATION (PBS / SLURM / NONE)
############################################
CORES_ON_MACHINE=$(nproc)

if [[ -n "$PBS_NODEFILE" && -f "$PBS_NODEFILE" ]]; then
    TOTAL_CORES=$(wc -l < "$PBS_NODEFILE")
    NODES_ALLOC=$(sort "$PBS_NODEFILE" | uniq | wc -l)
    CORES_PER_NODE=$(sort "$PBS_NODEFILE" | uniq -c | head -n 1 | awk '{print $1}')
    HOSTS_UNIQ=($(sort "$PBS_NODEFILE" | uniq))
elif [[ -n "$SLURM_CPUS_ON_NODE" ]]; then
    TOTAL_CORES=$SLURM_CPUS_ON_NODE
    NODES_ALLOC=${SLURM_JOB_NUM_NODES:-1}
    CORES_PER_NODE=$SLURM_CPUS_ON_NODE
    HOSTS_UNIQ=()
else
    TOTAL_CORES=$CORES_ON_MACHINE
    NODES_ALLOC=1
    CORES_PER_NODE=$CORES_ON_MACHINE
    HOSTS_UNIQ=()
fi

if (( TOTAL_CORES < CORES_ON_MACHINE )); then
    TOTAL_CORES=$CORES_ON_MACHINE
    NODES_ALLOC=1
    CORES_PER_NODE=$CORES_ON_MACHINE
fi

echo "Detected total cores allocated: $TOTAL_CORES"
echo "Detected nodes allocated      : $NODES_ALLOC"
echo "Detected cores per node       : $CORES_PER_NODE"

############################################
# CSV HEADERS (same fields + read_mode)
############################################
CSV_HEADER="matrix,scaling_type,read_mode,mpi_ranks,omp_threads,status,\
spmv_total_p90_ms,spmv_comm_p90_ms,spmv_comp_p90_ms,gflops,\
dist_sum_mb,iter_sum_mb,\
nnz_min,nnz_avg,nnz_max"

echo "$CSV_HEADER" > "$CSV_MPI"
echo "$CSV_HEADER" > "$CSV_HYB"

############################################
# FIELD EXTRACTOR (ROBUST)
############################################
extract_kv () {
  echo "$1" | awk -v key="$2" '
    {
      for(i=1;i<=NF;i++){
        if(index($i,key)==1){
          val=substr($i,length(key)+1);
          gsub(/[,;]/,"",val);
          print val;
          exit;
        }
      }
    }'
}

############################################
# CHOOSE USED NODES + PPN
############################################
choose_nodes_and_ppn () {
    local ranks=$1
    local threads=$2

    local total_threads=$((ranks * threads))

    local used_nodes=1
    if (( total_threads > CORES_PER_NODE )); then
        used_nodes=$NODES_ALLOC
        if (( used_nodes < 1 )); then used_nodes=1; fi
    fi

    local need_ppn=$(( (ranks + used_nodes - 1) / used_nodes ))
    if (( need_ppn < 1 )); then need_ppn=1; fi

    local max_ppn=$(( CORES_PER_NODE / threads ))
    if (( max_ppn < 1 )); then max_ppn=1; fi
    if (( need_ppn > max_ppn )); then need_ppn=$max_ppn; fi

    if (( need_ppn * used_nodes < ranks )); then
        echo "0 0"
        return
    fi

    echo "$used_nodes $need_ppn"
}

build_hostlist () {
    local used_nodes=$1

    if [[ -n "$PBS_NODEFILE" && -f "$PBS_NODEFILE" ]]; then
        local hs=()
        for ((i=0; i<used_nodes && i<${#HOSTS_UNIQ[@]}; i++)); do
            hs+=("${HOSTS_UNIQ[$i]}")
        done
        local IFS=,
        echo "${hs[*]}"
    else
        echo ""
    fi
}

############################################
# RUN SINGLE TEST
# mode: mpi|hybrid
############################################
run_test () {
    local matrix=$1
    local scaling=$2
    local ranks=$3
    local threads=$4
    local mode=$5  # mpi or hybrid

    local csv_out
    if [[ "$mode" == "mpi" ]]; then
        csv_out="$CSV_MPI"
    else
        csv_out="$CSV_HYB"
    fi

    if [[ ! -f "$matrix" ]]; then
        echo "Matrix not found: $matrix (skipping)"
        return
    fi

    local effective_threads=$threads
    local extra_args=""
    if [[ "$mode" == "mpi" ]]; then
        effective_threads=1
        extra_args="--no-omp"
    else
        extra_args="--threads $threads"
    fi

    local total_threads=$((ranks * effective_threads))
    if (( total_threads > TOTAL_CORES )); then
        echo "Skipping oversubscription vs allocation: MODE=$mode MPI=$ranks OMP=$effective_threads (need $total_threads, have $TOTAL_CORES)"
        return
    fi

    export OMP_NUM_THREADS=$effective_threads
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores
    export MPICH_CPU_BINDING=none

    read USED_NODES PPN < <(choose_nodes_and_ppn "$ranks" "$effective_threads")
    if [[ "$USED_NODES" == "0" || "$PPN" == "0" ]]; then
        echo "Skipping: cannot place ranks. MPI=$ranks OMP=$effective_threads cores/node=$CORES_PER_NODE nodes_alloc=$NODES_ALLOC"
        return
    fi

    HOSTLIST=$(build_hostlist "$USED_NODES")

    echo "Running: $mode | $scaling | read=$READ_MODE_FIXED | $matrix | MPI=$ranks | OMP=$effective_threads | USED_NODES=$USED_NODES | PPN=$PPN"

    if [[ -n "$HOSTLIST" ]]; then
        output=$(mpiexec -hosts "$HOSTLIST" -np "$ranks" -ppn "$PPN" -bind-to core \
            "$EXEC" "$matrix" --read="$READ_MODE_FIXED" $extra_args 2>&1)
    else
        output=$(mpiexec -np "$ranks" -ppn "$PPN" -bind-to core \
            "$EXEC" "$matrix" --read="$READ_MODE_FIXED" $extra_args 2>&1)
    fi
    status=$?

    if [[ $status -ne 0 ]]; then
        echo "MPI run failed (MODE=$mode MPI=$ranks OMP=$effective_threads)"
        echo "$matrix,$scaling,$READ_MODE_FIXED,$ranks,$effective_threads,ERROR,,,,,,,,," >> "$csv_out"
        return
    fi

    line_spmv=$(echo "$output" | grep -m1 "^SPMV_P90_MS:")
    line_comm=$(echo "$output" | grep -m1 "^COMM_MB:")
    line_nnz=$(echo "$output" | grep -m1 "^NNZ_STATS:")
    line_gfl=$(echo "$output" | grep -m1 "GFLOPS=")

    spmv_comm=$(extract_kv "$line_spmv" "COMM=")
    spmv_comp=$(extract_kv "$line_spmv" "COMP=")
    spmv_total=$(extract_kv "$line_spmv" "TOTAL=")

    dist_mb=$(extract_kv "$line_comm" "DIST_SUM_MB=")
    iter_mb=$(extract_kv "$line_comm" "ITER_SUM_MB=")

    nnz_min=$(extract_kv "$line_nnz" "MIN=")
    nnz_avg=$(extract_kv "$line_nnz" "AVG=")
    nnz_max=$(extract_kv "$line_nnz" "MAX=")

    gflops=$(echo "$line_gfl" | awk '{
      for(i=1;i<=NF;i++){
        if(index($i,"GFLOPS=")==1){
          print substr($i,8); exit
        }
      }
    }')

    echo "$matrix,$scaling,$READ_MODE_FIXED,$ranks,$effective_threads,OK,\
$spmv_total,$spmv_comm,$spmv_comp,$gflops,\
$dist_mb,$iter_mb,\
$nnz_min,$nnz_avg,$nnz_max" >> "$csv_out"
}

############################################
# STRONG SCALING
############################################
echo "===== STRONG SCALING (2D) ====="
for matrix in "${STRONG_MATRICES[@]}"; do
  for ranks in "${MPI_RANKS[@]}"; do
    run_test "$matrix" "strong" "$ranks" 1 "mpi"
    for threads in "${OMP_THREADS_HYB[@]}"; do
      run_test "$matrix" "strong" "$ranks" "$threads" "hybrid"
    done
  done
done

############################################
# WEAK SCALING
############################################
echo "===== WEAK SCALING (2D) ====="
for ranks in "${MPI_RANKS[@]}"; do
  matrix="$WEAK_DIR/${WEAK_PREFIX}${ranks}.mtx"

  run_test "$matrix" "weak" "$ranks" 1 "mpi"
  run_test "$matrix" "weak" "$ranks" "$WEAK_OMP" "hybrid"
done

echo "===== ALL TESTS COMPLETED (2D) ====="
echo "Results written to:"
echo "  $CSV_MPI"
echo "  $CSV_HYB"
