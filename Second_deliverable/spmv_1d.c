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

#include "mmio.h"

enum { ITERS = 15 };

typedef struct { int32_t r,c; double v; } Triplet;

typedef struct {
    int nrows;
    int64_t nnz;
    int *rowptr;        
    int *row_local_end; 
    int *idx;           
    double *vals;       
} CSR1D;

static inline double to_ms(double s){ return 1000.0*s; }

static void die(int rank, const char *msg){
    fprintf(stderr,"[rank %d] ERROR: %s\n", rank, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}
static void* xmalloc(size_t n){ void *p=malloc(n); if(!p){fprintf(stderr,"malloc failed\n"); exit(1);} return p; }
static void* xrealloc(void* p,size_t n){ void *q=realloc(p,n); if(!q){fprintf(stderr,"realloc failed\n"); exit(1);} return q; }

static int cmp_triplet_rowcol(const void *a,const void *b){
    const Triplet *x=(const Triplet*)a, *y=(const Triplet*)b;
    if(x->r!=y->r) return (x->r<y->r)?-1:1;
    if(x->c!=y->c) return (x->c<y->c)?-1:1;
    return 0;
}
static int cmp_i32(const void *a,const void *b){
    int32_t x=*(const int32_t*)a, y=*(const int32_t*)b;
    return (x>y)-(x<y);
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

static int local_count_cyclic(int64_t n_global, int rank, int P){
    if((int64_t)rank >= n_global) return 0;
    return (int)((n_global - rank + (P-1)) / P);
}
static inline int local_index_cyclic(int32_t g, int P){ return (int)(g / P); }

/* header with mmio (rank0)  */
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

/*  parallel chunk read (newline correct)  */
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


/*  MPI-IO parallel chunk read (MPI_File_read_at_all) */
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

/*  rank0 full read (serial)  */
static Triplet* root_read_all(const char *fname, int rank,
                              int64_t data_start,
                              int is_sym, int is_pattern,
                              int64_t *out_n)
{
    if(rank!=0){ *out_n=0; return NULL; }

    FILE *f=fopen(fname,"r");
    if(!f) die(rank,"fopen failed (root read)");
    if(fseeko(f,(off_t)data_start,SEEK_SET)!=0) die(rank,"fseeko failed (root read)");

    char line[512];
    int64_t cap=1024, n=0;
    Triplet *tr=(Triplet*)xmalloc((size_t)cap*sizeof(Triplet));

    while(fgets(line,(int)sizeof(line),f)){
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

/*  redistribute to cyclic row owners  */
static Triplet* redistribute_rows(const Triplet *in, int64_t nin,
                                 int P,
                                 int64_t *out_nlocal,
                                 int64_t *out_bytes)
{
    int64_t *sendcnt=(int64_t*)calloc((size_t)P,sizeof(int64_t));
    for(int64_t k=0;k<nin;k++) sendcnt[in[k].r % P]++;

    int64_t total_send=0;
    int64_t *sdisp=(int64_t*)xmalloc((size_t)P*sizeof(int64_t));
    for(int p=0;p<P;p++){ sdisp[p]=total_send; total_send+=sendcnt[p]; }

    Triplet *sendbuf=(Triplet*)xmalloc((size_t)(total_send>0?total_send:1)*sizeof(Triplet));
    int64_t *pos=(int64_t*)xmalloc((size_t)P*sizeof(int64_t));
    memcpy(pos,sdisp,(size_t)P*sizeof(int64_t));
    for(int64_t k=0;k<nin;k++){
        int dest=in[k].r % P;
        sendbuf[pos[dest]++] = in[k];
    }
    free(pos);

    int64_t *recvcnt=(int64_t*)xmalloc((size_t)P*sizeof(int64_t));
    MPI_Alltoall(sendcnt,1,MPI_INT64_T, recvcnt,1,MPI_INT64_T, MPI_COMM_WORLD);

    int64_t total_recv=0;
    int64_t *rdisp=(int64_t*)xmalloc((size_t)P*sizeof(int64_t));
    for(int p=0;p<P;p++){ rdisp[p]=total_recv; total_recv+=recvcnt[p]; }

    Triplet *recvbuf=(Triplet*)xmalloc((size_t)(total_recv>0?total_recv:1)*sizeof(Triplet));

    int *sc=(int*)xmalloc((size_t)P*sizeof(int));
    int *rc=(int*)xmalloc((size_t)P*sizeof(int));
    int *sd=(int*)xmalloc((size_t)P*sizeof(int));
    int *rd=(int*)xmalloc((size_t)P*sizeof(int));
    for(int p=0;p<P;p++){
        sc[p]=(int)(sendcnt[p]*(int64_t)sizeof(Triplet));
        rc[p]=(int)(recvcnt[p]*(int64_t)sizeof(Triplet));
        sd[p]=(int)(sdisp[p]  *(int64_t)sizeof(Triplet));
        rd[p]=(int)(rdisp[p]  *(int64_t)sizeof(Triplet));
    }

    MPI_Alltoallv(sendbuf, sc, sd, MPI_BYTE,
                  recvbuf, rc, rd, MPI_BYTE,
                  MPI_COMM_WORLD);

    int64_t bytes=0;
    for(int p=0;p<P;p++) bytes += (int64_t)sc[p];
    *out_bytes = bytes;

    free(sendcnt); free(sdisp); free(sendbuf);
    free(recvcnt); free(rdisp);
    free(sc); free(rc); free(sd); free(rd);

    *out_nlocal = total_recv;
    return recvbuf;
}

/* build remote cols (sorted unique)  */
static int32_t* build_remote_cols_from_triplets(const Triplet *t, int64_t nt, int rank, int P, int64_t *out_n){
    int64_t cap=1024, n=0;
    int32_t *tmp=(int32_t*)xmalloc((size_t)cap*sizeof(int32_t));
    for(int64_t k=0;k<nt;k++){
        int32_t col=t[k].c;
        if((col % P) != rank){
            if(n==cap){ cap*=2; tmp=(int32_t*)xrealloc(tmp,(size_t)cap*sizeof(int32_t)); }
            tmp[n++] = col;
        }
    }
    if(n==0){ free(tmp); *out_n=0; return NULL; }

    qsort(tmp,(size_t)n,sizeof(int32_t),cmp_i32);
    int64_t u=1;
    for(int64_t i=1;i<n;i++) if(tmp[i]!=tmp[u-1]) tmp[u++]=tmp[i];
    tmp=(int32_t*)xrealloc(tmp,(size_t)u*sizeof(int32_t));
    *out_n=u;
    return tmp;
}

static int64_t remote_pos_bsearch(const int32_t *remote_cols, int64_t nremote, int32_t col){
    int64_t lo=0, hi=nremote;
    while(lo<hi){
        int64_t mid=(lo+hi)/2;
        int32_t v=remote_cols[mid];
        if(v==col) return mid;
        if(v<col) lo=mid+1; else hi=mid;
    }
    return -1;
}

/*  CSR build  */
static CSR1D build_csr_1d_with_idx(const Triplet *t_in, int64_t nt,
                                  int64_t M, int rank, int P,
                                  const int32_t *remote_cols, int64_t nremote)
{
    Triplet *t=(Triplet*)xmalloc((size_t)nt*sizeof(Triplet));
    memcpy(t,t_in,(size_t)nt*sizeof(Triplet));
    qsort(t,(size_t)nt,sizeof(Triplet),cmp_triplet_rowcol);

    CSR1D A;
    A.nrows = local_count_cyclic(M, rank, P);
    A.nnz = nt;
    A.rowptr = (int*)xmalloc((size_t)(A.nrows+1)*sizeof(int));
    A.row_local_end = (int*)xmalloc((size_t)A.nrows*sizeof(int));
    A.idx = (int*)xmalloc((size_t)(A.nnz>0?A.nnz:1)*sizeof(int));
    A.vals= (double*)xmalloc((size_t)(A.nnz>0?A.nnz:1)*sizeof(double));

    memset(A.rowptr,0,(size_t)(A.nrows+1)*sizeof(int));
    for(int64_t k=0;k<nt;k++){
        int lr = local_index_cyclic(t[k].r, P);
        A.rowptr[lr+1]++;
    }
    for(int i=0;i<A.nrows;i++) A.rowptr[i+1]+=A.rowptr[i];

    int *local_cnt=(int*)calloc((size_t)A.nrows,sizeof(int));
    for(int64_t k=0;k<nt;k++){
        int lr=local_index_cyclic(t[k].r,P);
        if((t[k].c % P)==rank) local_cnt[lr]++;
    }

    int *cur_local=(int*)xmalloc((size_t)A.nrows*sizeof(int));
    int *cur_remote=(int*)xmalloc((size_t)A.nrows*sizeof(int));
    for(int i=0;i<A.nrows;i++){
        cur_local[i] = A.rowptr[i];
        A.row_local_end[i] = A.rowptr[i] + local_cnt[i];
        cur_remote[i] = A.row_local_end[i];
    }

    for(int64_t k=0;k<nt;k++){
        int32_t gr=t[k].r, gc=t[k].c;
        int lr=local_index_cyclic(gr,P);
        double v=t[k].v;

        if((gc % P)==rank){
            int pos=cur_local[lr]++;
            A.vals[pos]=v;
            A.idx[pos]=local_index_cyclic(gc,P);
        } else {
            int64_t rp = remote_pos_bsearch(remote_cols,nremote,gc);
            if(rp<0) die(rank,"remote_pos_bsearch failed (should not happen)");
            int pos=cur_remote[lr]++;
            A.vals[pos]=v;
            A.idx[pos]=-(int)(rp+1);
        }
    }

    free(t);
    free(local_cnt);
    free(cur_local);
    free(cur_remote);

    return A;
}

static int64_t count_diag_local_1d(const CSR1D *A, int rank, int P){
    int64_t diag=0;
    for(int i=0;i<A->nrows;i++){
        int64_t grow=(int64_t)rank + (int64_t)i*(int64_t)P;
        for(int k=A->rowptr[i]; k<A->rowptr[i+1]; k++){
            if(A->idx[k] >= 0){
                int64_t gcol = (int64_t)rank + (int64_t)A->idx[k]*(int64_t)P;
                if(gcol == grow) diag++;
            }
        }
    }
    return diag;
}

/*  comm plan  */
typedef struct {
    int P;
    int64_t *sendcnt_req;
    int64_t *sdisp_req;
    int32_t *send_idx_req;
    int64_t *sendpos_req;

    int64_t *recvcnt_req;
    int64_t *rdisp_req;
    int32_t *recv_idx_req;

    int *send_li;
} CommPlan1D;

static CommPlan1D build_comm_plan_1d(int P,
                                     const int32_t *remote_cols, int64_t nremote)
{
    CommPlan1D plan;
    plan.P=P;

    plan.sendcnt_req=(int64_t*)calloc((size_t)P,sizeof(int64_t));
    for(int64_t i=0;i<nremote;i++){
        int owner = remote_cols[i] % P;
        plan.sendcnt_req[owner]++;
    }

    plan.sdisp_req=(int64_t*)xmalloc((size_t)P*sizeof(int64_t));
    int64_t total_send_req=0;
    for(int p=0;p<P;p++){ plan.sdisp_req[p]=total_send_req; total_send_req += plan.sendcnt_req[p]; }

    plan.send_idx_req=(int32_t*)xmalloc((size_t)(total_send_req>0?total_send_req:1)*sizeof(int32_t));
    plan.sendpos_req =(int64_t*)xmalloc((size_t)(total_send_req>0?total_send_req:1)*sizeof(int64_t));

    int64_t *pos=(int64_t*)xmalloc((size_t)P*sizeof(int64_t));
    memcpy(pos,plan.sdisp_req,(size_t)P*sizeof(int64_t));
    for(int64_t i=0;i<nremote;i++){
        int owner = remote_cols[i] % P;
        int64_t k = pos[owner]++;
        plan.send_idx_req[k]=remote_cols[i];
        plan.sendpos_req[k]=i;
    }
    free(pos);

    plan.recvcnt_req=(int64_t*)xmalloc((size_t)P*sizeof(int64_t));
    MPI_Alltoall(plan.sendcnt_req,1,MPI_INT64_T, plan.recvcnt_req,1,MPI_INT64_T, MPI_COMM_WORLD);

    plan.rdisp_req=(int64_t*)xmalloc((size_t)P*sizeof(int64_t));
    int64_t total_recv_req=0;
    for(int p=0;p<P;p++){ plan.rdisp_req[p]=total_recv_req; total_recv_req += plan.recvcnt_req[p]; }

    plan.recv_idx_req=(int32_t*)xmalloc((size_t)(total_recv_req>0?total_recv_req:1)*sizeof(int32_t));

    int *sc=(int*)xmalloc((size_t)P*sizeof(int));
    int *rc=(int*)xmalloc((size_t)P*sizeof(int));
    int *sd=(int*)xmalloc((size_t)P*sizeof(int));
    int *rd=(int*)xmalloc((size_t)P*sizeof(int));
    for(int p=0;p<P;p++){
        sc[p]=(int)(plan.sendcnt_req[p]*(int64_t)sizeof(int32_t));
        rc[p]=(int)(plan.recvcnt_req[p]*(int64_t)sizeof(int32_t));
        sd[p]=(int)(plan.sdisp_req[p]  *(int64_t)sizeof(int32_t));
        rd[p]=(int)(plan.rdisp_req[p]  *(int64_t)sizeof(int32_t));
    }

    MPI_Alltoallv(plan.send_idx_req, sc, sd, MPI_BYTE,
                  plan.recv_idx_req, rc, rd, MPI_BYTE,
                  MPI_COMM_WORLD);

    free(sc); free(rc); free(sd); free(rd);

    plan.send_li = (int*)xmalloc((size_t)(total_recv_req>0?total_recv_req:1)*sizeof(int));
    for(int64_t i=0;i<total_recv_req;i++){
        plan.send_li[i] = local_index_cyclic(plan.recv_idx_req[i], P);
    }

    return plan;
}

static void free_comm_plan_1d(CommPlan1D *p){
    free(p->sendcnt_req);
    free(p->sdisp_req);
    free(p->send_idx_req);
    free(p->sendpos_req);
    free(p->recvcnt_req);
    free(p->rdisp_req);
    free(p->recv_idx_req);
    free(p->send_li);
}

static void exchange_values_start(const CommPlan1D *plan,
                                 const double *x_local,
                                 double *recv_vals_grouped,
                                 double *send_vals_grouped,
                                 MPI_Request *reqs,
                                 int *nreq_out)
{
    int P=plan->P;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int nreq=0;

    for(int p=0;p<P;p++){
        if(p==rank) continue;
        if(plan->sendcnt_req[p] <= 0) continue;
        MPI_Irecv(recv_vals_grouped + plan->sdisp_req[p],
                  (int)plan->sendcnt_req[p], MPI_DOUBLE,
                  p, 777, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    int64_t total_recv_req = plan->rdisp_req[P-1] + plan->recvcnt_req[P-1];
    for(int64_t i=0;i<total_recv_req;i++){
        send_vals_grouped[i] = x_local[ plan->send_li[i] ];
    }

    for(int p=0;p<P;p++){
        if(p==rank) continue;
        if(plan->recvcnt_req[p] <= 0) continue;
        MPI_Isend(send_vals_grouped + plan->rdisp_req[p],
                  (int)plan->recvcnt_req[p], MPI_DOUBLE,
                  p, 777, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    *nreq_out = nreq;
}

static void scatter_to_ghost(const CommPlan1D *plan,
                             const double *recv_vals_grouped,
                             double *ghost_vals_aligned)
{
    int P=plan->P;
    for(int p=0;p<P;p++){
        int64_t cnt = plan->sendcnt_req[p];
        int64_t disp = plan->sdisp_req[p];
        for(int64_t k=0;k<cnt;k++){
            int64_t global_pos = disp + k;
            int64_t rp = plan->sendpos_req[global_pos];
            ghost_vals_aligned[rp] = recv_vals_grouped[global_pos];
        }
    }
}

/* spmv split (runtime switch use_omp) */
static void spmv_local_part(const CSR1D *A, const double *x_local, double *y, int use_omp){
#ifdef _OPENMP
    if(use_omp){
        #pragma omp parallel for schedule(static)
        for(int i=0;i<A->nrows;i++){
            double sum=0.0;
            for(int k=A->rowptr[i]; k<A->row_local_end[i]; k++){
                sum += A->vals[k] * x_local[A->idx[k]];
            }
            y[i]=sum;
        }
        return;
    }
#endif
    for(int i=0;i<A->nrows;i++){
        double sum=0.0;
        for(int k=A->rowptr[i]; k<A->row_local_end[i]; k++){
            sum += A->vals[k] * x_local[A->idx[k]];
        }
        y[i]=sum;
    }
}

static void spmv_remote_part_add(const CSR1D *A, const double *ghost_vals, double *y, int use_omp){
#ifdef _OPENMP
    if(use_omp){
        #pragma omp parallel for schedule(static)
        for(int i=0;i<A->nrows;i++){
            double sum=y[i];
            for(int k=A->row_local_end[i]; k<A->rowptr[i+1]; k++){
                int g = -A->idx[k]-1;
                sum += A->vals[k] * ghost_vals[g];
            }
            y[i]=sum;
        }
        return;
    }
#endif
    for(int i=0;i<A->nrows;i++){
        double sum=y[i];
        for(int k=A->row_local_end[i]; k<A->rowptr[i+1]; k++){
            int g = -A->idx[k]-1;
            sum += A->vals[k] * ghost_vals[g];
        }
        y[i]=sum;
    }
}

/*  arg parsing  */
typedef enum { READ_PARALLEL=0, READ_ROOT=1, READ_MPIIO=2 } ReadMode;

static void usage_1d(int rank, const char *prog){
    if(rank!=0) return;
    fprintf(stderr,
        "Usage: %s matrix.mtx [--read=parallel|mpiio|root] [--no-omp] [--threads N]\n", prog);
}

static void parse_args_1d(int argc, char **argv, int rank,
                          const char **fname_out,
                          ReadMode *read_mode,
                          int *use_omp,
                          int *threads_out)
{
    *fname_out = NULL;
    *read_mode = READ_PARALLEL;
    *use_omp = 1;
    *threads_out = 0;

    if(argc < 2){ usage_1d(rank, argv[0]); MPI_Abort(MPI_COMM_WORLD,1); }

    *fname_out = argv[1];

    for(int i=2;i<argc;i++){
        const char *a = argv[i];
        if(strcmp(a,"--no-omp")==0){
            *use_omp = 0;
        } else if(strncmp(a,"--threads",9)==0){
            const char *v = NULL;
            if(strcmp(a,"--threads")==0){
                if(i+1>=argc) die(rank,"--threads requires a value");
                v = argv[++i];
            } else if(a[9]=='='){
                v = a+10;
            } else die(rank,"Invalid --threads syntax");
            *threads_out = atoi(v);
            if(*threads_out < 1) die(rank,"--threads must be >= 1");
        } else if(strncmp(a,"--read=",7)==0){
            const char *v=a+7;
            if(strcmp(v,"parallel")==0) *read_mode=READ_PARALLEL;
            else if(strcmp(v,"root")==0) *read_mode=READ_ROOT;
            else if(strcmp(v,"mpiio")==0) *read_mode=READ_MPIIO;
            else die(rank,"--read must be parallel, mpiio, or root");
        } else {
            if(rank==0) fprintf(stderr,"Unknown arg: %s\n", a);
            usage_1d(rank, argv[0]);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
}

/*  MAIN  */
int main(int argc,char **argv){
    MPI_Init(&argc,&argv);
    int rank,P;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&P);

    const char *fname=NULL;
    ReadMode read_mode=READ_PARALLEL;
    int use_omp=1;
    int threads=0;

    parse_args_1d(argc,argv,rank,&fname,&read_mode,&use_omp,&threads);

#ifdef _OPENMP
    if(!use_omp){
        omp_set_num_threads(1);
    } else if(threads>0){
        omp_set_num_threads(threads);
    }
#else
    use_omp = 0;
#endif

    int64_t M=0,N=0,NNZfile=0;
    int is_sym=0,is_pattern=0;
    int64_t data_start=0,fsize=0;

    double t_setup0=MPI_Wtime();

    double t_hdr0=MPI_Wtime();
    if(rank==0){
        read_header_rank0(fname, rank, &M,&N,&NNZfile, &is_sym,&is_pattern, &data_start,&fsize);
    }
    MPI_Bcast(&M,1,MPI_INT64_T,0,MPI_COMM_WORLD);
    MPI_Bcast(&N,1,MPI_INT64_T,0,MPI_COMM_WORLD);
    MPI_Bcast(&NNZfile,1,MPI_INT64_T,0,MPI_COMM_WORLD);
    MPI_Bcast(&is_sym,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&is_pattern,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&data_start,1,MPI_INT64_T,0,MPI_COMM_WORLD);
    MPI_Bcast(&fsize,1,MPI_INT64_T,0,MPI_COMM_WORLD);
    double t_hdr1=MPI_Wtime();

    double t_read0=MPI_Wtime();
    int64_t ntrip_chunk=0;
    Triplet *chunk=NULL;

    if(read_mode==READ_PARALLEL){
        chunk = parallel_read_chunk(fname, rank, P, data_start, fsize, is_sym, is_pattern, &ntrip_chunk);
    } else if(read_mode==READ_MPIIO){
        chunk = mpiio_read_chunk(fname, rank, P, data_start, fsize, is_sym, is_pattern, &ntrip_chunk);
    } else {
        chunk = root_read_all(fname, rank, data_start, is_sym, is_pattern, &ntrip_chunk);
    }
    double t_read1=MPI_Wtime();

    double t_dist0=MPI_Wtime();
    int64_t ntrip_local=0, dist_bytes_local=0;
    Triplet *tloc = redistribute_rows(chunk, ntrip_chunk, P, &ntrip_local, &dist_bytes_local);
    double t_dist1=MPI_Wtime();
    free(chunk);

    int64_t nremote=0;
    int32_t *remote_cols = build_remote_cols_from_triplets(tloc, ntrip_local, rank, P, &nremote);

    double t_csr0=MPI_Wtime();
    CSR1D A = build_csr_1d_with_idx(tloc, ntrip_local, M, rank, P, remote_cols, nremote);
    double t_csr1=MPI_Wtime();
    free(tloc);

    int64_t nnz_local=A.nnz, nnz_sum=0, nnz_min=0, nnz_max=0;
    MPI_Allreduce(&nnz_local,&nnz_sum,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&nnz_local,&nnz_min,1,MPI_INT64_T,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&nnz_local,&nnz_max,1,MPI_INT64_T,MPI_MAX,MPI_COMM_WORLD);
    double nnz_avg = (double)nnz_sum / (double)P;

    int64_t diag_local=count_diag_local_1d(&A, rank, P), diag_sum=0;
    MPI_Allreduce(&diag_local,&diag_sum,1,MPI_INT64_T,MPI_SUM,MPI_COMM_WORLD);

    int x_local_len=local_count_cyclic(N, rank, P);
    double *x_local=(double*)xmalloc((size_t)(x_local_len>0?x_local_len:1)*sizeof(double));
    for(int i=0;i<x_local_len;i++) x_local[i]=1.0;

    CommPlan1D plan = build_comm_plan_1d(P, remote_cols, nremote);

    int64_t total_send_req = plan.sdisp_req[P-1] + plan.sendcnt_req[P-1];
    int64_t total_recv_req = plan.rdisp_req[P-1] + plan.recvcnt_req[P-1];

    double *recv_vals_grouped=(double*)xmalloc((size_t)(total_send_req>0?total_send_req:1)*sizeof(double));
    double *send_vals_grouped=(double*)xmalloc((size_t)(total_recv_req>0?total_recv_req:1)*sizeof(double));
    double *ghost_vals=(double*)xmalloc((size_t)(nremote>0?nremote:1)*sizeof(double));

    MPI_Request *reqs=(MPI_Request*)xmalloc((size_t)(2*(P-1)+4)*sizeof(MPI_Request));
    double *y=(double*)xmalloc((size_t)(A.nrows>0?A.nrows:1)*sizeof(double));

    int64_t iter_comm_bytes_local_const=0;
    for(int p=0;p<P;p++){
        iter_comm_bytes_local_const += (int64_t)plan.sendcnt_req[p] * (int64_t)sizeof(double);
        iter_comm_bytes_local_const += (int64_t)plan.recvcnt_req[p] * (int64_t)sizeof(double);
    }

    double t_comm[ITERS], t_comp[ITERS], t_tot[ITERS];

    /*
     * Timing stabilization:
     * - Barrier before starting the timer: aligns iteration start across ranks.
     * - Barrier after finishing the iteration: prevents drift accumulating across iterations.
     *   (Barrier time is NOT included in the measured times.)
     */
    for(int it=0; it<ITERS; it++){
        MPI_Barrier(MPI_COMM_WORLD);
        double t0=MPI_Wtime();

        double tc_post0=MPI_Wtime();
        int nreq=0;
        exchange_values_start(&plan, x_local, recv_vals_grouped, send_vals_grouped, reqs, &nreq);
        double tc_post1=MPI_Wtime();

        double tp0=MPI_Wtime();
        spmv_local_part(&A, x_local, y, use_omp);
        double tp1=MPI_Wtime();

        double tc_wait0=MPI_Wtime();
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        double tc_wait1=MPI_Wtime();

        if(nremote>0) scatter_to_ghost(&plan, recv_vals_grouped, ghost_vals);

        double tp2=MPI_Wtime();
        if(nremote>0) spmv_remote_part_add(&A, ghost_vals, y, use_omp);
        double tp3=MPI_Wtime();

        double t1=MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        t_comm[it] = (tc_post1 - tc_post0) + (tc_wait1 - tc_wait0);
        t_comp[it] = (tp1 - tp0) + (tp3 - tp2);
        t_tot[it]  = (t1 - t0);
    }

    double local_sum=0.0;
    for(int i=0;i<A.nrows;i++) local_sum += y[i];
    double checksum=0.0;
    MPI_Allreduce(&local_sum,&checksum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    double max_tot[ITERS];
    int max_rank[ITERS];
    for(int it=0; it<ITERS; it++){
        struct { double val; int rank; } in, out;
        in.val = t_tot[it];
        in.rank = rank;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        max_tot[it] = out.val;
        max_rank[it] = out.rank;
    }

    int it_p90 = 0;
    if(rank==0){
        it_p90 = p90_index_of_array(max_tot, ITERS);
    }
    MPI_Bcast(&it_p90, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int owner_rank = max_rank[it_p90];
    MPI_Bcast(&owner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double comm_owner=0.0, comp_owner=0.0;
    if(rank==owner_rank){
        comm_owner = t_comm[it_p90];
        comp_owner = t_comp[it_p90];
        if(rank!=0){
            double buf[2] = {comm_owner, comp_owner};
            MPI_Send(buf, 2, MPI_DOUBLE, 0, 9901, MPI_COMM_WORLD);
        }
    }
    if(rank==0){
        if(owner_rank==0){
            comm_owner = t_comm[it_p90];
            comp_owner = t_comp[it_p90];
        } else {
            double buf[2];
            MPI_Recv(buf, 2, MPI_DOUBLE, owner_rank, 9901, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            comm_owner = buf[0];
            comp_owner = buf[1];
        }
    }

    double hdr_s=t_hdr1-t_hdr0, read_s=t_read1-t_read0, dist_s=t_dist1-t_dist0, csr_s=t_csr1-t_csr0;
    double hdr_max=0, read_max=0, dist_max=0, csr_max=0;
    MPI_Reduce(&hdr_s,&hdr_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&read_s,&read_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&dist_s,&dist_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&csr_s,&csr_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    int64_t dist_bytes_sum=0, iter_comm_bytes_sum=0;
    MPI_Reduce(&dist_bytes_local,&dist_bytes_sum,1,MPI_INT64_T,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&iter_comm_bytes_local_const,&iter_comm_bytes_sum,1,MPI_INT64_T,MPI_SUM,0,MPI_COMM_WORLD);

    double t_setup1=MPI_Wtime();
    double setup_s=t_setup1-t_setup0, setup_max=0;
    MPI_Reduce(&setup_s,&setup_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(rank==0){
        int64_t nnz_expected = is_sym ? (2*NNZfile - diag_sum) : NNZfile;
        double total_p90 = max_tot[it_p90];
        double gflops = (2.0*(double)nnz_sum) / total_p90 / 1e9;

        printf("===== SpMV RESULT =====\n");
        printf("ALG=SPMV1D_CYCLIC\n");
        const char *rm = (read_mode==READ_ROOT) ? "ROOT" : (read_mode==READ_MPIIO) ? "MPIIO" : "PARALLEL";
        printf("READ_MODE=%s  OMP_ENABLED=%d\n", rm, use_omp);
        printf("P=%d  OMP=%d  GRID=%dx%d\n", P,
#ifdef _OPENMP
               omp_get_max_threads(),
#else
               1,
#endif
               1, 1);
        printf("M=%"PRId64"  N=%"PRId64"  SYM=%d  PATTERN=%d\n", M, N, is_sym, is_pattern);
        printf("NNZ_FILE=%"PRId64"  NNZ_DIAG=%"PRId64"  NNZ_EXPANDED=%"PRId64"  NNZ_EXPECTED=%"PRId64"\n",
               NNZfile, diag_sum, nnz_sum, nnz_expected);
        printf("NNZ_STATS: MIN=%"PRId64"  AVG=%.2f  MAX=%"PRId64"\n", nnz_min, nnz_avg, nnz_max);
        printf("ITERS=%d  PCTL=90  COMM_MODE=TWOSIDED_VALUES  OVERLAP=1\n", ITERS);
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

    free(remote_cols);
    free_comm_plan_1d(&plan);

    free(recv_vals_grouped);
    free(send_vals_grouped);
    free(ghost_vals);
    free(reqs);

    free(A.rowptr);
    free(A.row_local_end);
    free(A.idx);
    free(A.vals);

    free(x_local);
    free(y);

    MPI_Finalize();
    return 0;
}
