#define _GNU_SOURCE
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <math.h>
#include <limits.h>

#include "mmio.h"

enum { ITERS = 15 };

typedef struct { int32_t r,c; double v; } Triplet;

typedef struct {
    int nrows_local;
    int ncols_local;
    int64_t nnz;
    int *rowptr;
    int *colind;
    double *vals;
} CSR;

static inline double to_ms(double s){ return 1000.0*s; }

static void die(int rank, const char *msg){
    fprintf(stderr,"[rank %d] ERROR: %s\n", rank, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}
static void* xmalloc(size_t n){
    void *p=malloc(n);
    if(!p){fprintf(stderr,"malloc failed\n"); exit(1);}
    return p;
}
static void* xrealloc(void* p,size_t n){
    void *q=realloc(p,n);
    if(!q){fprintf(stderr,"realloc failed\n"); exit(1);}
    return q;
}

static int cmp_triplet_rowcol(const void *a,const void *b){
    const Triplet *x=(const Triplet*)a, *y=(const Triplet*)b;
    if(x->r!=y->r) return (x->r<y->r)?-1:1;
    if(x->c!=y->c) return (x->c<y->c)?-1:1;
    return 0;
}

typedef struct { double v; int idx; } ValIdx;
static int cmp_validx(const void *a,const void *b){
    const ValIdx *x=(const ValIdx*)a, *y=(const ValIdx*)b;
    if(x->v < y->v) return -1;
    if(x->v > y->v) return 1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}
static int p90_index_of_array(const double *arr, int n){
    ValIdx *tmp=(ValIdx*)xmalloc((size_t)n*sizeof(ValIdx));
    for(int i=0;i<n;i++){ tmp[i].v=arr[i]; tmp[i].idx=i; }
    qsort(tmp,(size_t)n,sizeof(ValIdx),cmp_validx);
    int k=(int)ceil(0.9*(double)n)-1;
    if(k<0) k=0;
    if(k>=n) k=n-1;
    int idx=tmp[k].idx;
    free(tmp);
    return idx;
}

static void block_range_1d(int64_t n, int p, int coord, int64_t *start, int64_t *end){
    int64_t base=n/p, rem=n%p;
    if(coord<rem){ *start=coord*(base+1); *end=*start+(base+1); }
    else { *start=rem*(base+1)+(coord-rem)*base; *end=*start+base; }
}
static int find_block(int64_t x, int p, const int64_t *st, const int64_t *en){
    for(int i=0;i<p;i++) if(x>=st[i] && x<en[i]) return i;
    return p-1;
}

/* header rank0 */
static void read_header_rank0(const char *fname, int rank,
                              int64_t *M,int64_t *N,int64_t *NNZfile,
                              int *is_sym,int *is_pattern,
                              int64_t *data_start,int64_t *fsize)
{
    struct stat st;
    if(stat(fname,&st)!=0) die(rank,"stat() failed");
    *fsize=(int64_t)st.st_size;

    FILE *f=fopen(fname,"r");
    if(!f) die(rank,"fopen() failed");

    MM_typecode matcode;
    if(mm_read_banner(f,&matcode)!=0) die(rank,"mm_read_banner failed");
    if(!mm_is_matrix(matcode) || !mm_is_sparse(matcode) || !mm_is_coordinate(matcode))
        die(rank,"Only sparse coordinate MatrixMarket supported");
    if(!(mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)))
        die(rank,"Only real/integer/pattern supported");

    int m,n,nnz;
    if(mm_read_mtx_crd_size(f,&m,&n,&nnz)!=0) die(rank,"mm_read_mtx_crd_size failed");

    *M=m; *N=n; *NNZfile=nnz;
    *is_sym = mm_is_symmetric(matcode) ? 1 : 0;
    *is_pattern = mm_is_pattern(matcode) ? 1 : 0;

    long long off=ftello(f);
    if(off<0) die(rank,"ftello failed");
    *data_start=(int64_t)off;

    fclose(f);
}

/* parallel chunk read + symmetric expansion (alignment-aware) */
static Triplet* parallel_read_chunk(const char *fname, int rank, int P,
                                    int64_t data_start, int64_t fsize,
                                    int is_sym, int is_pattern,
                                    int64_t *out_n)
{
    int64_t data_bytes=fsize-data_start;
    if(data_bytes<0) data_bytes=0;

    int64_t base=data_bytes/P, rem=data_bytes%P;
    int64_t my_len,my_off;
    if(rank<rem){ my_len=base+1; my_off=rank*my_len; }
    else { my_len=base; my_off=rem*(base+1)+(rank-rem)*base; }

    int64_t my_start=data_start+my_off;
    int64_t my_end=my_start+my_len;

    FILE *f=fopen(fname,"r");
    if(!f) die(rank,"fopen failed (parallel read)");
    if(fseeko(f,(off_t)my_start,SEEK_SET)!=0) die(rank,"fseeko failed");

    if(rank!=0 && my_start > data_start){
        if(fseeko(f,(off_t)(my_start-1),SEEK_SET)!=0) die(rank,"fseeko prev failed");
        int prev = fgetc(f);
        if(fseeko(f,(off_t)my_start,SEEK_SET)!=0) die(rank,"fseeko start failed");

        if(prev != '\n' && prev != EOF){
            int c;
            while((c=fgetc(f))!=EOF){
                if(c=='\n') break;
            }
        }
    }

    char line[512];
    int64_t cap=1024, n=0;
    Triplet *tr=(Triplet*)xmalloc((size_t)cap*sizeof(Triplet));

    while(1){
        long long pos=ftello(f);
        if(pos<0) break;
        if(rank!=P-1 && pos>=my_end) break;

        if(!fgets(line,(int)sizeof(line),f)) break;

        char *s=line;
        while(*s && isspace((unsigned char)*s)) s++;
        if(*s=='\0' || *s=='%' || *s=='\n') continue;

        int64_t i,j; double v=1.0;
        int cnt;
        if(is_pattern){
            cnt=sscanf(s,"%" SCNd64 " %" SCNd64,&i,&j);
            if(cnt<2) continue;
            v=1.0;
        } else {
            cnt=sscanf(s,"%" SCNd64 " %" SCNd64 " %lf",&i,&j,&v);
            if(cnt<2) continue;
            if(cnt==2) v=1.0;
        }

        int32_t r=(int32_t)(i-1), c=(int32_t)(j-1);

        if(n==cap){ cap*=2; tr=(Triplet*)xrealloc(tr,(size_t)cap*sizeof(Triplet)); }
        tr[n++] = (Triplet){r,c,v};

        if(is_sym && r!=c){
            if(n==cap){ cap*=2; tr=(Triplet*)xrealloc(tr,(size_t)cap*sizeof(Triplet)); }
            tr[n++] = (Triplet){c,r,v};
        }
    }

    fclose(f);
    *out_n=n;
    return tr;
}


/*  MPI-IO parallel chunk read (MPI_File_read_at_all, newline correct)  */
static Triplet* mpiio_read_chunk(const char *fname, int rank, int P,
                                 int64_t data_start, int64_t fsize,
                                 int is_sym, int is_pattern,
                                 int64_t *out_n)
{
    int64_t data_bytes = fsize - data_start;
    if(data_bytes < 0) data_bytes = 0;

    int64_t base = data_bytes / P;
    int64_t rem  = data_bytes % P;

    int64_t my_len, my_off;
    if(rank < rem){ my_len = base + 1; my_off = rank * my_len; }
    else          { my_len = base;     my_off = rem * (base + 1) + (rank - rem) * base; }

    int64_t off  = data_start + my_off;
    int64_t end  = off + my_len;

    const int64_t pre  = (rank == 0)   ? 0 : 1;
    const int64_t post = (rank == P-1) ? 0 : 4096;

    int64_t read_off = off - pre;
    if(read_off < data_start) {
        read_off = data_start;
    }

    int64_t want = my_len + (off - read_off) + post;
    if(read_off + want > fsize) want = fsize - read_off;
    if(want < 0) want = 0;

    MPI_File fh;
    int rc = MPI_File_open(MPI_COMM_WORLD, (char*)fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(rc != MPI_SUCCESS) die(rank, "MPI_File_open failed");

    char *buf = (char*)xmalloc((size_t)(want > 0 ? want : 1) * sizeof(char));
    if(want > 0){
        MPI_Status st;
        rc = MPI_File_read_at_all(fh, (MPI_Offset)read_off, buf, (int)want, MPI_BYTE, &st);
        if(rc != MPI_SUCCESS) die(rank, "MPI_File_read_at_all failed");
    }
    MPI_File_close(&fh);

    int64_t ps = 0;
    int64_t boundary = (off - read_off) + my_len;
    if(boundary < 0) boundary = 0;
    if(boundary > want) boundary = want;

    if(rank != 0){
        int64_t start_idx = off - read_off;
        if(start_idx > 0 && start_idx <= want){
            if(buf[start_idx - 1] != '\n'){
                int64_t k = start_idx;
                while(k < want && buf[k] != '\n') k++;
                if(k < want) k++;
                ps = k;
            } else {
                ps = start_idx;
            }
        }
    } else {
        ps = 0;
    }

    int64_t pe = want;
    if(rank != P-1){
        int64_t k = boundary;
        if(k > want) k = want;
        while(k > ps && buf[k-1] != '\n') k--;
        pe = k;
    }

    int64_t cap = 1024, n = 0;
    Triplet *tr = (Triplet*)xmalloc((size_t)cap * sizeof(Triplet));

    int64_t i = ps;
    while(i < pe){
        int64_t j = i;
        while(j < pe && buf[j] != '\n') j++;

        int64_t linelen = j - i;
        if(linelen > 0){
            int64_t k = i;
            while(k < j && isspace((unsigned char)buf[k])) k++;
            if(k < j && buf[k] != '%' ){
                char line[512];
                if(linelen >= (int64_t)sizeof(line)) linelen = (int64_t)sizeof(line) - 1;
                memcpy(line, buf + i, (size_t)linelen);
                line[linelen] = '\0';

                char *s = line;
                while(*s && isspace((unsigned char)*s)) s++;
                if(*s && *s!='%' && *s!='\n'){
                    int64_t r,c; double v = 1.0;
                    int cnt;
                    if(is_pattern){
                        cnt = sscanf(s, "%" SCNd64 " %" SCNd64, &r, &c);
                        if(cnt >= 2){
                            v = 1.0;
                            r--; c--;
                            if(n >= cap){ cap *= 2; tr = (Triplet*)xrealloc(tr, (size_t)cap*sizeof(Triplet)); }
                            tr[n++] = (Triplet){ (int32_t)r, (int32_t)c, v };
                            if(is_sym && r != c){
                                if(n >= cap){ cap *= 2; tr = (Triplet*)xrealloc(tr, (size_t)cap*sizeof(Triplet)); }
                                tr[n++] = (Triplet){ (int32_t)c, (int32_t)r, v };
                            }
                        }
                    } else {
                        cnt = sscanf(s, "%" SCNd64 " %" SCNd64 " %lf", &r, &c, &v);
                        if(cnt >= 2){
                            if(cnt == 2) v = 1.0;
                            r--; c--;
                            if(n >= cap){ cap *= 2; tr = (Triplet*)xrealloc(tr, (size_t)cap*sizeof(Triplet)); }
                            tr[n++] = (Triplet){ (int32_t)r, (int32_t)c, v };
                            if(is_sym && r != c){
                                if(n >= cap){ cap *= 2; tr = (Triplet*)xrealloc(tr, (size_t)cap*sizeof(Triplet)); }
                                tr[n++] = (Triplet){ (int32_t)c, (int32_t)r, v };
                            }
                        }
                    }
                }
            }
        }

        i = (j < pe) ? (j + 1) : pe;
    }

    free(buf);
    *out_n = n;
    return tr;
}

/* Create MPI datatype for Triplet (portable, avoids MPI_BYTE tricks) */
static MPI_Datatype make_mpi_triplet_type(void){
    MPI_Datatype t;
    int blocklen[3] = {1,1,1};
    MPI_Aint disp[3];
    MPI_Datatype types[3] = {MPI_INT32_T, MPI_INT32_T, MPI_DOUBLE};

    Triplet tmp;
    MPI_Aint base;
    MPI_Get_address(&tmp, &base);
    MPI_Get_address(&tmp.r, &disp[0]); disp[0] -= base;
    MPI_Get_address(&tmp.c, &disp[1]); disp[1] -= base;
    MPI_Get_address(&tmp.v, &disp[2]); disp[2] -= base;

    MPI_Type_create_struct(3, blocklen, disp, types, &t);
    MPI_Type_commit(&t);
    return t;
}

/* redistribute to 2D owner (SAFE Alltoallv with element counts + overflow checks) */
static Triplet* redistribute_2d(const Triplet *in, int64_t nin,
                               MPI_Comm cart, int world_size,
                               int Pr,int Pc,
                               const int64_t *rstart,const int64_t *rend,
                               const int64_t *cstart,const int64_t *cend,
                               int64_t *out_nlocal,
                               int64_t *out_bytes,
                               MPI_Datatype triplet_type)
{
    int64_t *sendcnt64=(int64_t*)calloc((size_t)world_size,sizeof(int64_t));
    if(!sendcnt64) die(0, "calloc sendcnt64 failed");

    for(int64_t k=0;k<nin;k++){
        int pr=find_block((int64_t)in[k].r, Pr, rstart, rend);
        int pc=find_block((int64_t)in[k].c, Pc, cstart, cend);
        int coords[2]={pr,pc};
        int dest; MPI_Cart_rank(cart, coords, &dest);
        sendcnt64[dest]++;
    }

    int64_t total_send=0;
    int64_t *sdisp64=(int64_t*)xmalloc((size_t)world_size*sizeof(int64_t));
    for(int p=0;p<world_size;p++){ sdisp64[p]=total_send; total_send+=sendcnt64[p]; }

    Triplet *sendbuf=(Triplet*)xmalloc((size_t)(total_send>0?total_send:1)*sizeof(Triplet));
    int64_t *pos=(int64_t*)xmalloc((size_t)world_size*sizeof(int64_t));
    memcpy(pos,sdisp64,(size_t)world_size*sizeof(int64_t));

    for(int64_t k=0;k<nin;k++){
        int pr=find_block((int64_t)in[k].r, Pr, rstart, rend);
        int pc=find_block((int64_t)in[k].c, Pc, cstart, cend);
        int coords[2]={pr,pc};
        int dest; MPI_Cart_rank(cart, coords, &dest);
        sendbuf[pos[dest]++] = in[k];
    }
    free(pos);

    int64_t *recvcnt64=(int64_t*)xmalloc((size_t)world_size*sizeof(int64_t));
    MPI_Alltoall(sendcnt64,1,MPI_INT64_T, recvcnt64,1,MPI_INT64_T, cart);

    int64_t total_recv=0;
    int64_t *rdisp64=(int64_t*)xmalloc((size_t)world_size*sizeof(int64_t));
    for(int p=0;p<world_size;p++){ rdisp64[p]=total_recv; total_recv+=recvcnt64[p]; }

    Triplet *recvbuf=(Triplet*)xmalloc((size_t)(total_recv>0?total_recv:1)*sizeof(Triplet));

    /* MPI_Alltoallv wants int counts/displs: check fit */
    int *sc=(int*)xmalloc((size_t)world_size*sizeof(int));
    int *rc=(int*)xmalloc((size_t)world_size*sizeof(int));
    int *sd=(int*)xmalloc((size_t)world_size*sizeof(int));
    int *rd=(int*)xmalloc((size_t)world_size*sizeof(int));

    for(int p=0;p<world_size;p++){
        if(sendcnt64[p] > INT_MAX) die(0, "sendcnt overflow INT_MAX in Alltoallv");
        if(recvcnt64[p] > INT_MAX) die(0, "recvcnt overflow INT_MAX in Alltoallv");
        if(sdisp64[p]   > INT_MAX) die(0, "sdisp overflow INT_MAX in Alltoallv");
        if(rdisp64[p]   > INT_MAX) die(0, "rdisp overflow INT_MAX in Alltoallv");
        sc[p]=(int)sendcnt64[p];
        rc[p]=(int)recvcnt64[p];
        sd[p]=(int)sdisp64[p];
        rd[p]=(int)rdisp64[p];
    }

    MPI_Alltoallv(sendbuf, sc, sd, triplet_type,
                  recvbuf, rc, rd, triplet_type,
                  cart);

    int64_t bytes=0;
    for(int p=0;p<world_size;p++) bytes += (int64_t)sc[p] * (int64_t)sizeof(Triplet);
    *out_bytes = bytes;

    free(sendcnt64); free(sdisp64); free(sendbuf);
    free(recvcnt64); free(rdisp64);
    free(sc); free(rc); free(sd); free(rd);

    *out_nlocal = total_recv;
    return recvbuf;
}

/* COO->CSR local block */
static CSR build_csr_block(Triplet *t, int64_t nt,
                           int64_t r0,int64_t r1,
                           int64_t c0,int64_t c1)
{
    (void)c1;
    CSR A;
    A.nrows_local=(int)(r1-r0);
    A.ncols_local=(int)(c1-c0);
    A.nnz=nt;

    A.rowptr=(int*)xmalloc((size_t)(A.nrows_local+1)*sizeof(int));
    A.colind=(int*)xmalloc((size_t)(A.nnz>0?A.nnz:1)*sizeof(int));
    A.vals  =(double*)xmalloc((size_t)(A.nnz>0?A.nnz:1)*sizeof(double));

    qsort(t,(size_t)nt,sizeof(Triplet),cmp_triplet_rowcol);
    memset(A.rowptr,0,(size_t)(A.nrows_local+1)*sizeof(int));

    for(int64_t k=0;k<nt;k++){
        int lr=(int)((int64_t)t[k].r - r0);
        if(lr>=0 && lr<A.nrows_local) A.rowptr[lr+1]++;
    }
    for(int i=0;i<A.nrows_local;i++) A.rowptr[i+1]+=A.rowptr[i];

    int *cursor=(int*)xmalloc((size_t)A.nrows_local*sizeof(int));
    memcpy(cursor,A.rowptr,(size_t)A.nrows_local*sizeof(int));

    for(int64_t k=0;k<nt;k++){
        int lr=(int)((int64_t)t[k].r - r0);
        int lc=(int)((int64_t)t[k].c - c0);
        if(lr<0 || lr>=A.nrows_local) continue;
        int pos=cursor[lr]++;
        A.colind[pos]=lc;
        A.vals[pos]=t[k].v;
    }
    free(cursor);
    return A;
}

static void spmv_local_2d(const CSR *A, const double *x_local, double *y_partial, int use_omp){
#ifdef _OPENMP
    if(use_omp){
        #pragma omp parallel for schedule(static)
        for(int i=0;i<A->nrows_local;i++){
            double sum=0.0;
            for(int k=A->rowptr[i]; k<A->rowptr[i+1]; k++){
                sum += A->vals[k] * x_local[A->colind[k]];
            }
            y_partial[i]=sum;
        }
        return;
    }
#endif
    for(int i=0;i<A->nrows_local;i++){
        double sum=0.0;
        for(int k=A->rowptr[i]; k<A->rowptr[i+1]; k++){
            sum += A->vals[k] * x_local[A->colind[k]];
        }
        y_partial[i]=sum;
    }
}

static int64_t count_diag_local_2d(const CSR *A, int64_t r0, int64_t c0){
    int64_t diag=0;
    for(int i=0;i<A->nrows_local;i++){
        int64_t grow=r0+(int64_t)i;
        for(int k=A->rowptr[i]; k<A->rowptr[i+1]; k++){
            int64_t gcol=c0+(int64_t)A->colind[k];
            if(gcol==grow) diag++;
        }
    }
    return diag;
}

/* args */
typedef enum { READ_PARALLEL=0, READ_MPIIO=1 } ReadMode;

static void usage_2d(int rank, const char *prog){
    if(rank!=0) return;
    fprintf(stderr,"Usage: %s matrix.mtx [--read=parallel|mpiio] [--no-omp] [--threads N]\n", prog);
}
static void parse_args_2d(int argc, char **argv, int rank,
                          const char **fname_out, ReadMode *read_mode,
                          int *use_omp, int *threads_out)
{
    *fname_out=NULL;
    *read_mode=READ_PARALLEL;
    *use_omp=1;
    *threads_out=0;
    if(argc<2){ usage_2d(rank, argv[0]); MPI_Abort(MPI_COMM_WORLD,1); }
    *fname_out=argv[1];

    for(int i=2;i<argc;i++){
        const char *a=argv[i];
        if(strcmp(a,"--no-omp")==0){
            *use_omp=0;
        } else if(strncmp(a,"--threads",9)==0){
            const char *v=NULL;
            if(strcmp(a,"--threads")==0){
                if(i+1>=argc) die(rank,"--threads requires a value");
                v=argv[++i];
            } else if(a[9]=='='){
                v=a+10;
            } else die(rank,"Invalid --threads syntax");
            *threads_out=atoi(v);
            if(*threads_out<1) die(rank,"--threads must be >= 1");
        } else if(strncmp(a,"--read=",7)==0){
            const char *v=a+7;
            if(strcmp(v,"parallel")==0) *read_mode=READ_PARALLEL;
            else if(strcmp(v,"mpiio")==0) *read_mode=READ_MPIIO;
            else die(rank,"--read must be parallel or mpiio");
        } else {
            if(rank==0) fprintf(stderr,"Unknown arg: %s\n", a);
            usage_2d(rank, argv[0]);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
}

int main(int argc,char **argv){
    MPI_Init(&argc,&argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    int tmp_rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&tmp_rank);

    const char *fname=NULL;
    ReadMode read_mode=READ_PARALLEL;
    int use_omp=1;
    int threads=0;
    parse_args_2d(argc,argv,tmp_rank,&fname,&read_mode,&use_omp,&threads);

#ifdef _OPENMP
    if(!use_omp) omp_set_num_threads(1);
    else if(threads>0) omp_set_num_threads(threads);
#else
    use_omp=0;
#endif

    int dims[2]={0,0};
    MPI_Dims_create(world_size,2,dims);
    int Pr=dims[0], Pc=dims[1];

    int periods[2]={0,0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&cart);

    int rank;
    MPI_Comm_rank(cart,&rank);

    int coords[2];
    MPI_Cart_coords(cart,rank,2,coords);
    int my_pr=coords[0], my_pc=coords[1];

    MPI_Comm row_comm;
    int keep_row[2]={0,1};
    MPI_Cart_sub(cart, keep_row, &row_comm);

    MPI_Datatype triplet_type = make_mpi_triplet_type();

    int64_t M=0,N=0,NNZfile=0;
    int is_sym=0,is_pattern=0;
    int64_t data_start=0,fsize=0;

    double t_setup0=MPI_Wtime();

    double t_hdr0=MPI_Wtime();
    if(rank==0){
        read_header_rank0(fname, rank, &M,&N,&NNZfile, &is_sym,&is_pattern, &data_start,&fsize);
    }
    MPI_Bcast(&M,1,MPI_INT64_T,0,cart);
    MPI_Bcast(&N,1,MPI_INT64_T,0,cart);
    MPI_Bcast(&NNZfile,1,MPI_INT64_T,0,cart);
    MPI_Bcast(&is_sym,1,MPI_INT,0,cart);
    MPI_Bcast(&is_pattern,1,MPI_INT,0,cart);
    MPI_Bcast(&data_start,1,MPI_INT64_T,0,cart);
    MPI_Bcast(&fsize,1,MPI_INT64_T,0,cart);
    double t_hdr1=MPI_Wtime();

    int64_t *rstart=(int64_t*)xmalloc((size_t)Pr*sizeof(int64_t));
    int64_t *rend  =(int64_t*)xmalloc((size_t)Pr*sizeof(int64_t));
    int64_t *cstart=(int64_t*)xmalloc((size_t)Pc*sizeof(int64_t));
    int64_t *cend  =(int64_t*)xmalloc((size_t)Pc*sizeof(int64_t));
    for(int pr=0;pr<Pr;pr++) block_range_1d(M,Pr,pr,&rstart[pr],&rend[pr]);
    for(int pc=0;pc<Pc;pc++) block_range_1d(N,Pc,pc,&cstart[pc],&cend[pc]);

    int64_t r0=rstart[my_pr], r1=rend[my_pr];
    int64_t c0=cstart[my_pc], c1=cend[my_pc];

    double t_read0=MPI_Wtime();
    int64_t ntrip_chunk=0;
    Triplet *chunk = (read_mode==READ_MPIIO)
        ? mpiio_read_chunk(fname, rank, world_size, data_start, fsize, is_sym, is_pattern, &ntrip_chunk)
        : parallel_read_chunk(fname, rank, world_size, data_start, fsize, is_sym, is_pattern, &ntrip_chunk);
    double t_read1=MPI_Wtime();

    double t_dist0=MPI_Wtime();
    int64_t ntrip_local=0, dist_bytes_local=0;
    Triplet *tloc = redistribute_2d(chunk, ntrip_chunk, cart, world_size,
                                    Pr,Pc, rstart,rend, cstart,cend,
                                    &ntrip_local, &dist_bytes_local,
                                    triplet_type);
    double t_dist1=MPI_Wtime();
    free(chunk);

    double t_csr0=MPI_Wtime();
    int64_t w=0;
    for(int64_t k=0;k<ntrip_local;k++){
        int64_t gr=tloc[k].r, gc=tloc[k].c;
        if(gr>=r0 && gr<r1 && gc>=c0 && gc<c1) tloc[w++] = tloc[k];
    }
    ntrip_local=w;

    CSR A = build_csr_block(tloc, ntrip_local, r0,r1, c0,c1);
    double t_csr1=MPI_Wtime();
    free(tloc);

    int64_t nnz_local=A.nnz, nnz_sum=0, nnz_min=0, nnz_max=0;
    MPI_Allreduce(&nnz_local,&nnz_sum,1,MPI_INT64_T,MPI_SUM,cart);
    MPI_Allreduce(&nnz_local,&nnz_min,1,MPI_INT64_T,MPI_MIN,cart);
    MPI_Allreduce(&nnz_local,&nnz_max,1,MPI_INT64_T,MPI_MAX,cart);
    double nnz_avg = (double)nnz_sum / (double)world_size;

    int64_t diag_local=count_diag_local_2d(&A,r0,c0), diag_sum=0;
    MPI_Allreduce(&diag_local,&diag_sum,1,MPI_INT64_T,MPI_SUM,cart);

    int ncols_local=(int)(c1-c0);
    double *x_local=(double*)xmalloc((size_t)(ncols_local>0?ncols_local:1)*sizeof(double));
    for(int i=0;i<ncols_local;i++) x_local[i]=1.0;

    int nrows_local=(int)(r1-r0);
    double *y_partial=(double*)xmalloc((size_t)(nrows_local>0?nrows_local:1)*sizeof(double));
    double *y=(double*)xmalloc((size_t)(nrows_local>0?nrows_local:1)*sizeof(double));

    double t_comm[ITERS], t_comp[ITERS], t_tot[ITERS];

    /*
     * Timing stabilization:
     * - Barrier before timing aligns iteration start across the full 2D grid.
     * - Barrier after timing prevents inter-iteration drift.
     *   (Barrier time is NOT included in the measured times.)
     */
    for(int it=0; it<ITERS; it++){
        MPI_Barrier(cart);
        double t0=MPI_Wtime();

        double tp0=MPI_Wtime();
        spmv_local_2d(&A,x_local,y_partial,use_omp);
        double tp1=MPI_Wtime();

        double tc0=MPI_Wtime();
        MPI_Allreduce(y_partial,y,nrows_local,MPI_DOUBLE,MPI_SUM,row_comm);
        double tc1=MPI_Wtime();

        double t1=MPI_Wtime();
        MPI_Barrier(cart);

        t_comp[it]=tp1-tp0;
        t_comm[it]=tc1-tc0;
        t_tot[it]=t1-t0;
    }

    double local_sum=0.0;
    if(my_pc==0){
        for(int i=0;i<nrows_local;i++) local_sum += y[i];
    }
    double checksum=0.0;
    MPI_Allreduce(&local_sum,&checksum,1,MPI_DOUBLE,MPI_SUM,cart);

    double max_tot[ITERS];
    int max_rank[ITERS];
    for(int it=0; it<ITERS; it++){
        struct { double val; int rank; } in, out;
        in.val = t_tot[it];
        in.rank = rank;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, cart);
        max_tot[it] = out.val;
        max_rank[it] = out.rank;
    }

    int it_p90 = 0;
    if(rank==0){
        it_p90 = p90_index_of_array(max_tot, ITERS);
    }
    MPI_Bcast(&it_p90, 1, MPI_INT, 0, cart);

    int owner_rank = max_rank[it_p90];
    MPI_Bcast(&owner_rank, 1, MPI_INT, 0, cart);

    double comm_owner=0.0, comp_owner=0.0;
    if(rank==owner_rank){
        comm_owner = t_comm[it_p90];
        comp_owner = t_comp[it_p90];
        if(rank!=0){
            double buf[2] = {comm_owner, comp_owner};
            MPI_Send(buf, 2, MPI_DOUBLE, 0, 9902, cart);
        }
    }
    if(rank==0){
        if(owner_rank==0){
            comm_owner = t_comm[it_p90];
            comp_owner = t_comp[it_p90];
        } else {
            double buf[2];
            MPI_Recv(buf, 2, MPI_DOUBLE, owner_rank, 9902, cart, MPI_STATUS_IGNORE);
            comm_owner = buf[0];
            comp_owner = buf[1];
        }
    }

    double hdr_s=t_hdr1-t_hdr0, read_s=t_read1-t_read0, dist_s=t_dist1-t_dist0, csr_s=t_csr1-t_csr0;
    double hdr_max=0, read_max=0, dist_max=0, csr_max=0;
    MPI_Reduce(&hdr_s,&hdr_max,1,MPI_DOUBLE,MPI_MAX,0,cart);
    MPI_Reduce(&read_s,&read_max,1,MPI_DOUBLE,MPI_MAX,0,cart);
    MPI_Reduce(&dist_s,&dist_max,1,MPI_DOUBLE,MPI_MAX,0,cart);
    MPI_Reduce(&csr_s,&csr_max,1,MPI_DOUBLE,MPI_MAX,0,cart);

    int64_t dist_bytes_sum=0;
    MPI_Reduce(&dist_bytes_local,&dist_bytes_sum,1,MPI_INT64_T,MPI_SUM,0,cart);

    int64_t iter_comm_bytes_local=(int64_t)nrows_local*(int64_t)sizeof(double);
    int64_t iter_comm_bytes_sum=0;
    MPI_Reduce(&iter_comm_bytes_local,&iter_comm_bytes_sum,1,MPI_INT64_T,MPI_SUM,0,cart);

    double t_setup1=MPI_Wtime();
    double setup_s=t_setup1-t_setup0, setup_max=0;
    MPI_Reduce(&setup_s,&setup_max,1,MPI_DOUBLE,MPI_MAX,0,cart);

    if(rank==0){
        int64_t nnz_expected = is_sym ? (2*NNZfile - diag_sum) : NNZfile;
        double total_p90 = max_tot[it_p90];
        double gflops = (2.0*(double)nnz_sum) / total_p90 / 1e9;

        printf("===== SpMV RESULT =====\n");
        printf("ALG=SPMV2D_CART\n");
        printf("OMP_ENABLED=%d\n", use_omp);
        printf("P=%d  OMP=%d  GRID=%dx%d\n", world_size,
#ifdef _OPENMP
               omp_get_max_threads(),
#else
               1,
#endif
               Pr, Pc);
        printf("M=%"PRId64"  N=%"PRId64"  SYM=%d  PATTERN=%d\n", M, N, is_sym, is_pattern);
        printf("NNZ_FILE=%"PRId64"  NNZ_DIAG=%"PRId64"  NNZ_EXPANDED=%"PRId64"  NNZ_EXPECTED=%"PRId64"\n",
               NNZfile, diag_sum, nnz_sum, nnz_expected);
        printf("NNZ_STATS: MIN=%"PRId64"  AVG=%.2f  MAX=%"PRId64"\n", nnz_min, nnz_avg, nnz_max);
        printf("ITERS=%d  PCTL=90  COMM_MODE=ALLREDUCE_ROW  OVERLAP=0\n", ITERS);
        printf("\n");
        printf("SETUP_TIMES_MS: HDR=%.3f  READ=%.3f  DIST=%.3f  CSR=%.3f  SETUP=%.3f\n",
               to_ms(hdr_max), to_ms(read_max), to_ms(dist_max), to_ms(csr_max), to_ms(setup_max));
        printf("SPMV_P90_MS:  COMM=%.3f  COMP=%.3f  TOTAL=%.3f  (it=%d owner=%d)\n",
               to_ms(comm_owner), to_ms(comp_owner), to_ms(total_p90), it_p90, owner_rank);
        printf("\n");
        printf("COMM_MB: DIST_SUM_MB=%.3f  ITER_SUM_MB=%.3f\n",
               (double)dist_bytes_sum / (1024.0 * 1024.0),
               (double)iter_comm_bytes_sum / (1024.0 * 1024.0));
        printf("GFLOPS=%.6f  CHECKSUM=%.12e\n", gflops, checksum);
        printf("=======================\n");
    }

    free(rstart); free(rend); free(cstart); free(cend);
    free(A.rowptr); free(A.colind); free(A.vals);
    free(x_local); free(y_partial); free(y);

    MPI_Type_free(&triplet_type);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&cart);

    MPI_Finalize();
    return 0;
}
