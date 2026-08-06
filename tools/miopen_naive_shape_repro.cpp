// Targets the exact shape decoded from the crashing kernel's arg trace
// (naive_conv_ab_nonpacked_fwd_nchw_float_double_float):
//   hi=16 wi=50 n=1 k_per_group=240 c_per_group=40 ho=16 wo=50
//   sy=1 sx=1 dy=1 dx=1 py=0 px=0 fy=1 fx=1 group=1
// i.e. a 1x1 pointwise conv, 40->240 channels, 16x50 spatial -- a real
// layer from CLAP's audio backbone, apparently uncovered by any tuned
// gfx803 solver so MIOpen falls back to the naive/reference kernel, which
// is what faults. Single call, no session/handle churn, to see if this is
// shape-triggered on its own.
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP err %s at line %d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CHECK_MIO(x) do { miopenStatus_t s=(x); if(s!=miopenStatusSuccess){fprintf(stderr,"MIOpen err %d at line %d\n",(int)s,__LINE__);exit(1);} } while(0)

int main(int argc, char** argv) {
    int N_ITERS = argc > 1 ? atoi(argv[1]) : 1;
    const int N = 1, C = 40, H = 16, W = 50, K = 240, R = 1, S = 1;

    size_t x_elems = (size_t)N*C*H*W;
    size_t w_elems = (size_t)K*C*R*S;
    int OH = H, OW = W; // 1x1 conv, stride1, pad0 -> same spatial dims
    size_t y_elems = (size_t)N*K*OH*OW;

    fprintf(stderr, "shape: N=%d C=%d H=%d W=%d K=%d R=%d S=%d OH=%d OW=%d\n", N,C,H,W,K,R,S,OH,OW);
    fprintf(stderr, "x_elems=%zu w_elems=%zu y_elems=%zu\n", x_elems, w_elems, y_elems);

    void *d_x, *d_w, *d_y;
    CHECK_HIP(hipMalloc(&d_x, x_elems*4));
    CHECK_HIP(hipMalloc(&d_w, w_elems*4));
    CHECK_HIP(hipMalloc(&d_y, y_elems*4));
    CHECK_HIP(hipMemset(d_x, 0, x_elems*4));
    CHECK_HIP(hipMemset(d_w, 0, w_elems*4));
    CHECK_HIP(hipMemset(d_y, 0, y_elems*4));

    for (int iter = 0; iter < N_ITERS; iter++) {
        miopenHandle_t handle;
        CHECK_MIO(miopenCreate(&handle));

        miopenTensorDescriptor_t xDesc, wDesc, yDesc;
        CHECK_MIO(miopenCreateTensorDescriptor(&xDesc));
        CHECK_MIO(miopenCreateTensorDescriptor(&wDesc));
        CHECK_MIO(miopenCreateTensorDescriptor(&yDesc));
        CHECK_MIO(miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W));
        CHECK_MIO(miopenSet4dTensorDescriptor(wDesc, miopenFloat, K, C, R, S));
        CHECK_MIO(miopenSet4dTensorDescriptor(yDesc, miopenFloat, N, K, OH, OW));

        miopenConvolutionDescriptor_t convDesc;
        CHECK_MIO(miopenCreateConvolutionDescriptor(&convDesc));
        CHECK_MIO(miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 0, 0, 1, 1, 1, 1));

        size_t ws_size = 0;
        miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc, convDesc, yDesc, &ws_size);
        void* d_ws = nullptr;
        if (ws_size > 0) CHECK_HIP(hipMalloc(&d_ws, ws_size));
        fprintf(stderr, "iter %d: workspace size = %zu\n", iter, ws_size);

        int returned_algo_count = 0;
        miopenConvAlgoPerf_t perf;
        CHECK_MIO(miopenFindConvolutionForwardAlgorithm(handle, xDesc, d_x, wDesc, d_w, convDesc, yDesc, d_y,
                                                        1, &returned_algo_count, &perf, d_ws, ws_size, false));
        fprintf(stderr, "iter %d: selected algo=%d workspace_used=%zu\n", iter, (int)perf.fwd_algo, perf.memory);

        float alpha = 1.0f, beta = 0.0f;
        CHECK_MIO(miopenConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w, convDesc,
                                           perf.fwd_algo, &beta, yDesc, d_y, d_ws, ws_size));
        CHECK_HIP(hipDeviceSynchronize());
        fprintf(stderr, "iter %d: OK\n", iter);

        if (d_ws) hipFree(d_ws);
        miopenDestroyConvolutionDescriptor(convDesc);
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(wDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroy(handle);
    }

    fprintf(stderr, "ALL %d ITERS COMPLETED\n", N_ITERS);
    hipFree(d_x); hipFree(d_w); hipFree(d_y);
    return 0;
}
