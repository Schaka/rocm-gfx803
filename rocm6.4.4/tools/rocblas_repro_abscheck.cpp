#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP err %s\n",hipGetErrorString(e));exit(1);} } while(0)
#define CHECK_RB(x) do { rocblas_status s=(x); if(s!=rocblas_status_success){fprintf(stderr,"RB err %d\n",(int)s);exit(1);} } while(0)
int main(int argc, char** argv){
    const int M=argc>1?atoi(argv[1]):40, K=argc>2?atoi(argv[2]):72, N=argc>3?atoi(argv[3]):800;
    const int ITERS=argc>4?atoi(argv[4]):500;
    rocblas_handle handle; CHECK_RB(rocblas_create_handle(&handle));
    size_t a_elems=(size_t)M*K,b_elems=(size_t)K*N,c_elems=(size_t)M*N;
    float *h_a=(float*)malloc(a_elems*4),*h_b=(float*)malloc(b_elems*4),*h_c=(float*)malloc(c_elems*4);
    void *d_a,*d_b,*d_c;
    CHECK_HIP(hipMalloc(&d_a,a_elems*4)); CHECK_HIP(hipMalloc(&d_b,b_elems*4)); CHECK_HIP(hipMalloc(&d_c,c_elems*4));
    const float alpha=1.0f, beta=0.0f;
    int total_bad_real=0, total_bad_noise=0;
    for(int iter=0; iter<ITERS; iter++){
        for(size_t i=0;i<a_elems;i++) h_a[i]=(float)(rand()%1000)/1000.0f-0.5f;
        for(size_t i=0;i<b_elems;i++) h_b[i]=(float)(rand()%1000)/1000.0f-0.5f;
        CHECK_HIP(hipMemcpy(d_a,h_a,a_elems*4,hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_b,h_b,b_elems*4,hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(d_c,0,c_elems*4));
        CHECK_RB(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, M,N,K,&alpha,(const float*)d_a,M,(const float*)d_b,K,&beta,(float*)d_c,M));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_c,d_c,c_elems*4,hipMemcpyDeviceToHost));
        int bad_real=0, bad_noise=0;
        double max_abs=0, max_rel=0;
        for(int n=0;n<N;n++) for(int m=0;m<M;m++){
            double ref=0.0;
            for(int k=0;k<K;k++) ref += (double)h_a[k*M+m]*(double)h_b[n*K+k];
            double got=h_c[n*M+m];
            double err_abs=fabs(got-ref);
            double err_rel=err_abs/(fabs(ref)+1e-6);
            if(err_abs>0.05 && err_rel>1e-2){ bad_real++; if(err_abs>max_abs)max_abs=err_abs; }
            else if(err_rel>1e-2){ bad_noise++; if(err_rel>max_rel)max_rel=err_rel; }
        }
        if(bad_real){ total_bad_real++; fprintf(stderr,"iter %d: REAL %d bad, max_abs=%f\n", iter, bad_real, max_abs); }
        if(bad_noise){ total_bad_noise++; fprintf(stderr,"iter %d: NOISE-ONLY %d flagged-by-relative-only, max_rel=%f\n", iter, bad_noise, max_rel); }
    }
    fprintf(stderr,"TOTAL: %d/%d real corruption, %d/%d relative-only-noise\n", total_bad_real, ITERS, total_bad_noise, ITERS);
    return 0;
}
