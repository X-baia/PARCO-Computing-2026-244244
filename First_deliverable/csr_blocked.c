#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>
#include "mmio.h"

typedef struct {
    int rows, cols, nnz;
    int *row_ptr;
    int *col_idx;
    double *vals;
} CSR;

CSR read_matrix_market_to_csr(const char *fn) {
    FILE *f; MM_typecode matcode;
    int M,N,nz;
    if ((f = fopen(fn,"r"))==NULL){perror("fopen"); exit(1);}
    if (mm_read_banner(f,&matcode)!=0){fprintf(stderr,"MM banner error\n"); exit(1);}
    if (mm_read_mtx_crd_size(f,&M,&N,&nz)!=0){fprintf(stderr,"MM size error\n"); exit(1);}
    
    int *I = malloc(nz*sizeof(int));
    int *J = malloc(nz*sizeof(int));
    double *V = malloc(nz*sizeof(double));
    if (!I||!J||!V){fprintf(stderr,"malloc failed\n"); exit(1);}
    
    for(int i=0;i<nz;i++){
        if(fscanf(f,"%d %d %lf",&I[i],&J[i],&V[i])!=3){fprintf(stderr,"read triplet failed\n"); exit(1);}
        I[i]--; J[i]--; 
    }
    fclose(f);

    CSR A;
    A.rows = M; A.cols = N; A.nnz = nz;
    A.row_ptr = calloc(M+1,sizeof(int));
    A.col_idx = malloc(nz*sizeof(int));
    A.vals = malloc(nz*sizeof(double));
    if(!A.row_ptr || !A.col_idx || !A.vals){fprintf(stderr,"malloc failed\n"); exit(1);}

    for(int i=0;i<nz;i++) A.row_ptr[I[i]+1]++;
    for(int i=0;i<M;i++) A.row_ptr[i+1] += A.row_ptr[i];

    int *counter = calloc(M,sizeof(int));
    for(int i=0;i<nz;i++){
        int r=I[i];
        int pos = A.row_ptr[r] + counter[r];
        A.col_idx[pos] = J[i];
        A.vals[pos] = V[i];
        counter[r]++;
    }
    free(I); free(J); free(V); free(counter);
    return A;
}

void spmv_blocked_sched(const CSR *A, const double *x, double *y, int block_rows, const char *schedule_type, int chunk_size) {
    int rows = A->rows;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<rows;i++) y[i]=0.0;

    int num_blocks = (rows + block_rows - 1)/block_rows;

    omp_sched_t schedule_kind;

    if (strcmp(schedule_type, "static") == 0) schedule_kind = omp_sched_static;
    else if (strcmp(schedule_type, "dynamic") == 0) schedule_kind = omp_sched_dynamic;
    else if (strcmp(schedule_type, "guided") == 0) schedule_kind = omp_sched_guided;
    else { printf("Unknown schedule: %s\n", schedule_type); exit(1); }

    omp_set_schedule(schedule_kind, chunk_size);

    #pragma omp parallel for schedule(runtime)
    for(int b=0;b<num_blocks;b++){
        int start_row = b*block_rows;
        int end_row = start_row + block_rows;
        if(end_row>rows) end_row=rows;

        for(int r=start_row;r<end_row;r++){
            double sum=0.0;
            for(int j=A->row_ptr[r]; j<A->row_ptr[r+1]; j++){
                sum += A->vals[j]*x[A->col_idx[j]];
            }
            y[r] = sum;
        }
    }
}

double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

int cmp_double(const void *a,const void *b){
    double x=*(double*)a, y=*(double*)b;
    if(x<y) return -1; if(x>y) return 1; return 0;
}

int main(int argc,char *argv[]){
    if(argc<2){
        fprintf(stderr,"Usage: %s matrix.mtx [threads] [block_rows] [schedule] [chunks]\n",argv[0]);
        return 1;
    }
    const char *fn = argv[1];
    int threads = (argc>=3)?atoi(argv[2]):1;
    int block_rows = (argc>=4)?atoi(argv[3]):128;
    const char *schedule_type = (argc>=5)?(argv[4]):"static";
    int chunk_size = (argc>=6)?atoi(argv[5]):0;
    if(threads<=0) threads=1;
    if(block_rows<=0) block_rows=128;

    omp_set_num_threads(threads);
    CSR A = read_matrix_market_to_csr(fn);

    double *x = malloc(A.cols*sizeof(double));
    double *y = malloc(A.rows*sizeof(double));
    for(int i=0;i<A.cols;i++) x[i] = (double)rand()/RAND_MAX;

    const int runs=15;
    double times[runs];

    for(int r=0;r<runs;r++){
        double t0=now_sec();
        spmv_blocked_sched(&A,x,y,block_rows,schedule_type,chunk_size);
        double t1=now_sec();
        times[r] = (t1-t0)*1000.0;
    }

    qsort(times,runs,sizeof(double),cmp_double);
    double p90=times[13];
    double flops = 2.0 * A.nnz;
    double gflops = flops / (p90 * 1e6);
    
    
    double bytes = A.nnz * 20.0;
    double bandwidth = bytes / (p90 * 1e6); 

    printf("%s;%s;%s;%d;%s;%d;%d;%.6f;%.6f;%.6f\n",
          argv[1], "blocked", "", threads, schedule_type, chunk_size, block_rows, 
           p90, gflops, bandwidth);
    
    
    //printf("%.6f\n", times[13]); // 90th percentile

    free(A.row_ptr); free(A.col_idx); free(A.vals);
    free(x); free(y);
    return 0;
}
