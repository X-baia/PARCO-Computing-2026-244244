#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mmio.h"

typedef struct {
    int rows, cols, nnz;
    int *row_ptr;
    int *col_idx;
    double *values;
} CSR;

typedef struct {
    int row;
    int col;
    double val;
} Triplet;

int compareTriplets(const void *a, const void *b) {
    const Triplet *t1 = (const Triplet *)a;
    const Triplet *t2 = (const Triplet *)b;
    if (t1->row != t2->row)
        return t1->row - t2->row;
    else
        return t1->col - t2->col;
}

CSR read_matrix_market_to_csr(const char *filename) {
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;

    if ((f = fopen(filename, "r")) == NULL) {
        printf("Cannot open %s\n", filename);
        exit(1);
    }
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        printf("Could not read matrix size.\n");
        exit(1);
    }

    Triplet *triplets = malloc(nz * sizeof(Triplet));
    if (!triplets) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (int i = 0; i < nz; i++) {
        if (fscanf(f, "%d %d %lf\n", &triplets[i].row, &triplets[i].col, &triplets[i].val) != 3) {
            fprintf(stderr, "Error reading triplet %d\n", i);
            exit(1);
        }
        triplets[i].row--;
        triplets[i].col--;
    }
    fclose(f);
    qsort(triplets, nz, sizeof(Triplet), compareTriplets);

    CSR A;
    A.rows = M;
    A.cols = N;
    A.nnz = nz;
    A.row_ptr = calloc(M + 1, sizeof(int));
    A.col_idx = malloc(nz * sizeof(int));
    A.values = malloc(nz * sizeof(double));
    if (!A.row_ptr || !A.col_idx || !A.values) { 
        fprintf(stderr, "malloc failed\n"); exit(1); 
    }

    for (int i = 0; i < nz; i++)
        A.row_ptr[triplets[i].row + 1]++;

    for (int i = 0; i < M; i++)
        A.row_ptr[i + 1] += A.row_ptr[i];

    int *counter = calloc(M, sizeof(int));
    for (int i = 0; i < nz; i++) {
        int r = triplets[i].row;
        int dest = A.row_ptr[r] + counter[r];
        A.col_idx[dest] = triplets[i].col;
        A.values[dest] = triplets[i].val;
        counter[r]++;
    }

    free(counter);
    free(triplets);
    return A;
}


void spmv_csr_parallel_loop(int rows, const int *row_ptr, const int *col_idx, const double *values, const double *x, double *y, const char *schedule, int chunk_size) {

    omp_sched_t kind;
    if (strcmp(schedule, "static") == 0) kind = omp_sched_static;
    else if (strcmp(schedule, "dynamic") == 0) kind = omp_sched_dynamic;
    else if (strcmp(schedule, "guided") == 0) kind = omp_sched_guided;
    else {
        printf("Unknown schedule: %s (use static, dynamic, or guided)\n", schedule);
        exit(1);
    }

    omp_set_schedule(kind, chunk_size);

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            sum += values[j] * x[col_idx[j]];
        y[i] = sum;
    }
}


void spmv_csr_parallel_task(int rows, const int *row_ptr, const int *col_idx, const double *values, const double *x, double *y, int chunk_size) {
    if (chunk_size <= 0) chunk_size = 1; 
    #pragma omp parallel
    {
        #pragma omp single
        for (int i = 0; i < rows; i += chunk_size) {
            int i_end = (i + chunk_size < rows) ? (i + chunk_size) : rows;
            #pragma omp task firstprivate(i, i_end)
            for (int r = i; r < i_end; r++) {
                double sum = 0.0;
                for (int j = row_ptr[r]; j < row_ptr[r + 1]; j++)
                    sum += values[j] * x[col_idx[j]];
                y[r] = sum;
            }
        }
    }
}


void spmv_csr_parallel_inner(int rows, const int *row_ptr, const int *col_idx,
const double *values, const double *x, double *y,
const char *schedule, int chunk_size) {
    // Choose OpenMP schedule
    omp_sched_t kind;
    if (strcmp(schedule, "static") == 0) kind = omp_sched_static;
    else if (strcmp(schedule, "dynamic") == 0) kind = omp_sched_dynamic;
    else if (strcmp(schedule, "guided") == 0) kind = omp_sched_guided;
    else { printf("Unknown schedule: %s\n", schedule); exit(1); }


    omp_set_schedule(kind, chunk_size);


    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(runtime)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            sum += values[j] * x[col_idx[j]];
        y[i] = sum;
}
}


void spmv_csr_parallel_collapse(int rows, const int *row_ptr, const int *col_idx,
                                const double *values, const double *x, double *y) {
    int maxlen = 0;
    for (int i = 0; i < rows; i++) {
        int len = row_ptr[i + 1] - row_ptr[i];
        if (len > maxlen) maxlen = len;
    }

    for (int i = 0; i < rows; i++) y[i] = 0.0;

    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int i = 0; i < rows; i++) {
        for (int t = 0; t < maxlen; t++) {
            int idx = row_ptr[i] + t;
            if (idx < row_ptr[i + 1]) {
                double v = values[idx] * x[col_idx[idx]];
                #pragma omp atomic
                y[i] += v;
            }
        }
    }
}

int cmp_double(const void *a, const void *b) {
    double diff = *(double*)a - *(double*)b;
    return (diff > 0) - (diff < 0);
}

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Hoe to run:\n");
        printf("Loop mode : %s matrix.mtx [threads] loop [static|dynamic|guided] [chunk]\n", argv[0]);
        printf("Task mode : %s matrix.mtx [threads] task [chunk_rows]\n", argv[0]);
        printf("Inner mode: %s matrix.mtx [threads] inner [chunk]\n", argv[0]);
        printf("Collapse : %s matrix.mtx [threads] collapse\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int threads = atoi(argv[2]);
    const char *mode = argv[3];
    omp_set_num_threads(threads);

    CSR A = read_matrix_market_to_csr(filename);
    double *x = malloc(A.cols * sizeof(double));
    double *y = malloc(A.rows * sizeof(double));
    if (!x || !y) { 
        fprintf(stderr, "malloc failed\n"); return 1; 
    }

    srand(12345);
    for (int i = 0; i < A.cols; i++) x[i] = (double)rand() / RAND_MAX;

    double times[15];
    if (strcmp(mode, "loop") == 0) {
        if (argc < 5) { printf("Missing schedule type\n"); return 1; }
        const char *schedule = argv[4];
        int chunk_size = (argc >= 6) ? atoi(argv[5]) : 0;
        for (int run = 0; run<15; run++){
            double start = get_time_sec();
            spmv_csr_parallel_loop(A.rows, A.row_ptr, A.col_idx, A.values, x, y, schedule, chunk_size);
            double end = get_time_sec();
            times[run] = (end - start) * 1000.0;
            //printf("Run %d: %.6f ms\n", run, times[run]);
        }
        qsort(times, 15, sizeof(double), cmp_double);
        double p90 = times[13];
        double flops = 2.0 * A.nnz;
        double gflops = flops / (p90 * 1e6);
        double bytes = A.nnz * 20.0;
        double bandwidth = bytes / (p90 * 1e6); // GB/s
        printf("%s;%s;%s;%d;%s;%d;%d;%.6f;%.6f;%.6f\n",
            argv[1], "parallel(omp+csr)", mode, threads, schedule, chunk_size, 0, 
                p90, gflops, bandwidth);
    } else if (strcmp(mode, "task") == 0) {
        int chunk_rows = (argc >= 5) ? atoi(argv[4]) : 1;
        for (int run = 0; run<15; run++){
            double start = get_time_sec();
            spmv_csr_parallel_task(A.rows, A.row_ptr, A.col_idx, A.values, x, y, chunk_rows);
            double end = get_time_sec();
            times[run] = (end - start) * 1000.0;
            //printf("Run %d: %.6f ms\n", run, times[run]);
        }
        qsort(times, 15, sizeof(double), cmp_double);
        double p90 = times[13];
        double flops = 2.0 * A.nnz;
        double gflops = flops / (p90 * 1e6);
        double bytes = A.nnz * 20.0;
        double bandwidth = bytes / (p90 * 1e6); // GB/s
        printf("%s;%s;%s;%d;%s;%d;%d;%.6f;%.6f;%.6f\n",
            argv[1], "parallel(omp+csr)", mode, threads, "", 0, 0, 
                p90, gflops, bandwidth);
    } else if (strcmp(mode, "inner") == 0) {
        if (argc < 5) { printf("Missing schedule type\n"); return 1; }
        const char *schedule = argv[4];
        int chunk_size = (argc >= 6) ? atoi(argv[5]) : 0;
        for (int run = 0; run<15; run++){
            double start = get_time_sec();
            spmv_csr_parallel_inner(A.rows, A.row_ptr, A.col_idx, A.values, x, y, schedule, chunk_size);
            double end = get_time_sec();
            times[run] = (end - start) * 1000.0;
            //printf("Run %d: %.6f ms\n", run, times[run]);
        }  
        qsort(times, 15, sizeof(double), cmp_double);
        double p90 = times[13];
        double flops = 2.0 * A.nnz;
        double gflops = flops / (p90 * 1e6);
        double bytes = A.nnz * 20.0;
        double bandwidth = bytes / (p90 * 1e6); // GB/s
        printf("%s;%s;%s;%d;%s;%d;%d;%.6f;%.6f;%.6f\n",
            argv[1], "parallel(omp+csr)", mode, threads, schedule, chunk_size, 0, 
                p90, gflops, bandwidth);  
    } else if (strcmp(mode, "collapse") == 0) {
        for (int run = 0; run<15; run++){
            double start = get_time_sec();
            spmv_csr_parallel_collapse(A.rows, A.row_ptr, A.col_idx, A.values, x, y);
            double end = get_time_sec();
            times[run] = (end - start) * 1000.0;
            //printf("Run %d: %.6f ms\n", run, times[run]);
        }  
        qsort(times, 15, sizeof(double), cmp_double);
        double p90 = times[13];
        double flops = 2.0 * A.nnz;
        double gflops = flops / (p90 * 1e6);
        double bytes = A.nnz * 20.0;
        double bandwidth = bytes / (p90 * 1e6); // GB/s
        printf("%s;%s;%s;%d;%s;%d;%d;%.6f;%.6f;%.6f\n",
            argv[1], "parallel(omp+csr)", mode, threads, "", 0, 0, 
                p90, gflops, bandwidth);
    } else {
        printf("Unknown mode: %s (use 'loop','task','inner','collapse')\n", mode);
        return 1;
    }

    /*qsort(times, 15, sizeof(double), cmp_double);
    double p90 = times[13];
    double flops = 2.0 * A.nnz;
    double gflops = flops / (p90 * 1e6);
    double bytes = A.nnz * 20.0;
    double bandwidth = bytes / (p90 * 1e6); // GB/s
    printf("%s;%s;%s;%d;%d;%d;%d;%.6f;%.6f;%.6f\n",
       argv[1], "parallel(omp+csr)", threads, mode, 0, 0, 0, 
       p90, gflops, bandwidth);*/
    //printf("[%s mode | %d threads] 90th percentile: %.6f ms\n", mode, threads, p90);

    free(A.row_ptr); free(A.col_idx); free(A.values);
    free(x); free(y);
    return 0;
}
