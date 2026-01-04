#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include<string.h>

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
    if (!triplets) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lf\n", &triplets[i].row, &triplets[i].col, &triplets[i].val);
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
        fprintf(stderr, "Memory allocation failed for CSR.\n");
        exit(1);
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

void spmv_csr(int rows, const int *row_ptr, const int *col_idx, const double *values, const double *x, double *y) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            sum += values[j] * x[col_idx[j]];
        y[i] = sum;
    }
}

int cmp_double(const void *a, const void *b) {
    double diff = *(double*)a - *(double*)b;
    return (diff > 0) - (diff < 0);
}

double get_time_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime failed");
        exit(EXIT_FAILURE);
    }
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s matrix.mtx\n", argv[0]);
        return 1;
    }

    CSR A = read_matrix_market_to_csr(argv[1]);

    double *x = malloc(A.cols * sizeof(double));
    double *y = malloc(A.rows * sizeof(double));

    srand(12345);
    for (int i = 0; i < A.cols; i++) x[i] = (double)rand() / RAND_MAX;

    double times[15];
    for (int run = 0; run < 15; run++) {
        double start = get_time_ms();
        spmv_csr(A.rows, A.row_ptr, A.col_idx, A.values, x, y);
        double end = get_time_ms();
        times[run] = (end - start) * 1000.0; 
    }

    qsort(times, 15, sizeof(double), cmp_double);
    double p90 = times[13]; 

    double flops = 2.0 * A.nnz;
    double gflops = flops / (p90 * 1e6);

    double bytes = A.nnz * 20.0;
    double bandwidth = bytes / (p90 * 1e6); 

    printf("%s;%s;%s;%d;%s;%d;%d;%.6f;%.6f;%.6f\n",
       argv[1], "sequential", "",0, "", 0, 0, 
       p90, gflops, bandwidth);


    free(A.row_ptr); free(A.col_idx); free(A.values);
    free(x); free(y);
    return 0;
}