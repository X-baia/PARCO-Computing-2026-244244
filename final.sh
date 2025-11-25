#!/bin/bash

MATRICES=("1138_bus.mtx" "bcsstk13.mtx" "west0479.mtx" "orani678.mtx" "smt.mtx" "TEM27623_M.mtx" "TSOPF_FS_b39_c7.mtx" "bratu3d.mtx")
THREADS=(1 2 4 8 16 32)
SCHEDULES=("static" "dynamic" "guided")
CHUNKS=(0 1 4 8 16 32)
BLOCKS=(32 64 128 256)
MODE=("loop") #"task" "inner" "collapse" including these in mode will run test with these types of parallelization

SEQ_EXE="./sequential_spmv"
OMP_EXE="./parallel_omp_spmv_complete"
BLK_EXE="./csr_blocked"

ALL="final_results_all.csv"
BEST="final_results_best.csv"

echo "matrix;approach;variant;threads;schedule;chunk;block_rows;p90_ms;gflops;bandwidth" > $ALL


echo "Running SEQUENTIAL"
for M in "${MATRICES[@]}"; do
    $SEQ_EXE "$M" >> $ALL
done

echo "Running OMP CSR"
for M in "${MATRICES[@]}"; do
    for MO in "$MODE"; do
        for T in "${THREADS[@]}"; do
            for SCH in "${SCHEDULES[@]}"; do
                for CH in "${CHUNKS[@]}"; do
                    $OMP_EXE "$M" "$T" "$MO" "$SCH" "$CH" >> $ALL
                done
            done
        done
    done
done

echo "Running BLOCKED CSR"
for M in "${MATRICES[@]}"; do
    for T in "${THREADS[@]}"; do
        for SCH in "${SCHEDULES[@]}"; do
            for CH in "${CHUNKS[@]}"; do
                for B in "${BLOCKS[@]}"; do
                    $BLK_EXE "$M" "$T" "$B" "$SCH" "$CH" >> $ALL
                done
            done
        done
    done
done

echo "All runs complete."
echo "Extracting best results for each matrix and each approach..."


echo "matrix;approach;variant;threads;schedule;chunk;block_rows;p90_ms;gflops;bandwidth" > $BEST

for M in "${MATRICES[@]}"; do
    echo "Processing $M ..."

    awk -F";" -v mat="$M" '$1==mat && $2=="sequential"{print}' $ALL \
        | sort -t";" -k8,8n | head -n 1 >> $BEST

    awk -F";" -v mat="$M" '$1==mat && $2=="parallel(omp+csr)"{print}' $ALL \
        | sort -t";" -k8,8n | head -n 1 >> $BEST

    awk -F";" -v mat="$M" '$1==mat && $2=="blocked"{print}' $ALL \
        | sort -t";" -k8,8n | head -n 1 >> $BEST
done

echo "Done!"
echo "Generated:"
echo " - $ALL   (all configurations)"
echo " - $BEST  (best seq + best omp + best blocked per matrix)"

python3 plot_best_results.py
python3 plot_best_results_3.py
python3 plot_comparison.py
echo "All plots created in 'plots/' folder."