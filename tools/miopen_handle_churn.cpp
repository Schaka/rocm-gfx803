// Isolates whether the ConvActivationFusion crash (hard GPU page fault after
// N ORT session churns, single long-lived session survives 180+ inferences
// fine) is in MIOpen's own handle create/destroy lifecycle, independent of
// ORT/ROCMExecutionProvider entirely. Creates a fresh MIOpen handle, runs
// one small fused conv+bias+activation, destroys the handle, repeat -- the
// same session-churn pattern clap_churn_repro.py exercises via ORT, but
// with nothing but MIOpen itself in the loop.
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP err %s at line %d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CHECK_MIO(x) do { miopenStatus_t s=(x); if(s!=miopenStatusSuccess){fprintf(stderr,"MIOpen err %d at line %d\n",(int)s,__LINE__);exit(1);} } while(0)

int main(int argc, char** argv) {
    int N_CHURN = argc > 1 ? atoi(argv[1]) : 100;
    // small conv shape, deliberately modest so this runs fast
    const int N = 1, C = 8, H = 32, W = 32, K = 16, R = 3, S = 3;

    size_t x_elems = (size_t)N*C*H*W;
    size_t w_elems = (size_t)K*C*R*S;
    int OH = H - R + 1, OW = W - S + 1;
    size_t y_elems = (size_t)N*K*OH*OW;
    size_t bias_elems = K;

    void *d_x, *d_w, *d_y, *d_bias;
    CHECK_HIP(hipMalloc(&d_x, x_elems*4));
    CHECK_HIP(hipMalloc(&d_w, w_elems*4));
    CHECK_HIP(hipMalloc(&d_y, y_elems*4));
    CHECK_HIP(hipMalloc(&d_bias, bias_elems*4));
    CHECK_HIP(hipMemset(d_x, 0, x_elems*4));
    CHECK_HIP(hipMemset(d_w, 0, w_elems*4));
    CHECK_HIP(hipMemset(d_bias, 0, bias_elems*4));

    for (int iter = 0; iter < N_CHURN; iter++) {
        miopenHandle_t handle;
        CHECK_MIO(miopenCreate(&handle));

        miopenTensorDescriptor_t xDesc, wDesc, yDesc, biasDesc;
        CHECK_MIO(miopenCreateTensorDescriptor(&xDesc));
        CHECK_MIO(miopenCreateTensorDescriptor(&wDesc));
        CHECK_MIO(miopenCreateTensorDescriptor(&yDesc));
        CHECK_MIO(miopenCreateTensorDescriptor(&biasDesc));
        CHECK_MIO(miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W));
        CHECK_MIO(miopenSet4dTensorDescriptor(wDesc, miopenFloat, K, C, R, S));
        CHECK_MIO(miopenSet4dTensorDescriptor(yDesc, miopenFloat, N, K, OH, OW));
        CHECK_MIO(miopenSet4dTensorDescriptor(biasDesc, miopenFloat, 1, K, 1, 1));

        miopenConvolutionDescriptor_t convDesc;
        CHECK_MIO(miopenCreateConvolutionDescriptor(&convDesc));
        CHECK_MIO(miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 0, 0, 1, 1, 1, 1));

        miopenActivationDescriptor_t actDesc;
        CHECK_MIO(miopenCreateActivationDescriptor(&actDesc));
        CHECK_MIO(miopenSetActivationDescriptor(actDesc, miopenActivationRELU, 0, 0, 0));

        size_t ws_size = 0;
        miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc, convDesc, yDesc, &ws_size);
        void* d_ws = nullptr;
        if (ws_size > 0) CHECK_HIP(hipMalloc(&d_ws, ws_size));

        int returned_algo_count = 0;
        miopenConvAlgoPerf_t perf;
        CHECK_MIO(miopenFindConvolutionForwardAlgorithm(handle, xDesc, d_x, wDesc, d_w, convDesc, yDesc, d_y,
                                                        1, &returned_algo_count, &perf, d_ws, ws_size, false));

        float alpha = 1.0f, beta = 0.0f;
        CHECK_MIO(miopenConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w, convDesc,
                                           perf.fwd_algo, &beta, yDesc, d_y, d_ws, ws_size));

        float actAlpha = 1.0f, actBeta = 0.0f;
        CHECK_MIO(miopenActivationForward(handle, actDesc, &actAlpha, yDesc, d_y, &actBeta, yDesc, d_y));

        CHECK_HIP(hipDeviceSynchronize());

        if (d_ws) hipFree(d_ws);
        miopenDestroyConvolutionDescriptor(convDesc);
        miopenDestroyActivationDescriptor(actDesc);
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(wDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(biasDesc);
        miopenDestroy(handle);

        if (iter % 10 == 0) fprintf(stderr, "churn %d/%d OK\n", iter, N_CHURN);
    }

    fprintf(stderr, "ALL %d CHURNS COMPLETED\n", N_CHURN);
    hipFree(d_x); hipFree(d_w); hipFree(d_y); hipFree(d_bias);
    return 0;
}
