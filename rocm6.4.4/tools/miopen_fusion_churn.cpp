// Targets the ACTUAL API ORT's ConvActivationFusion optimizer uses --
// MIOpen's fusion-plan API (miopenCreateFusionPlan / OpConvForward /
// OpBiasForward / OpActivationForward / CompileFusionPlan /
// ExecuteFusionPlan), not the plain sequential conv+activation calls
// miopen_handle_churn.cpp used (which came back 100/100 clean, ruling out
// generic MIOpen handle lifecycle but NOT this specific fused code path).
// Churns fresh handle + fusion plan creation/compile/execute/destroy in a
// loop, matching the ORT session-churn pattern that crashes at ~session 4.
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP err %s at line %d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CHECK_MIO(x) do { miopenStatus_t s=(x); if(s!=miopenStatusSuccess){fprintf(stderr,"MIOpen err %d at line %d\n",(int)s,__LINE__);exit(1);} } while(0)

int main(int argc, char** argv) {
    int N_CHURN = argc > 1 ? atoi(argv[1]) : 50;
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

        miopenFusionPlanDescriptor_t fusePlan;
        CHECK_MIO(miopenCreateFusionPlan(&fusePlan, miopenVerticalFusion, xDesc));

        miopenFusionOpDescriptor_t convOp, biasOp, activOp;
        CHECK_MIO(miopenCreateOpConvForward(fusePlan, &convOp, convDesc, wDesc));
        CHECK_MIO(miopenCreateOpBiasForward(fusePlan, &biasOp, biasDesc));
        CHECK_MIO(miopenCreateOpActivationForward(fusePlan, &activOp, miopenActivationRELU));

        miopenStatus_t compile_status = miopenCompileFusionPlan(handle, fusePlan);
        if (compile_status != miopenStatusSuccess) {
            fprintf(stderr, "iter %d: compile failed status=%d (may be unsupported fusion combo, not necessarily a bug)\n", iter, (int)compile_status);
            miopenDestroyFusionPlan(fusePlan);
            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(xDesc);
            miopenDestroyTensorDescriptor(wDesc);
            miopenDestroyTensorDescriptor(yDesc);
            miopenDestroyTensorDescriptor(biasDesc);
            miopenDestroy(handle);
            continue;
        }

        miopenOperatorArgs_t args;
        CHECK_MIO(miopenCreateOperatorArgs(&args));
        float alpha = 1.0f, beta = 0.0f;
        CHECK_MIO(miopenSetOpArgsConvForward(args, convOp, &alpha, &beta, d_w));
        CHECK_MIO(miopenSetOpArgsBiasForward(args, biasOp, &alpha, &beta, d_bias));
        CHECK_MIO(miopenSetOpArgsActivForward(args, activOp, &alpha, &beta, 0.0, 0.0, 0.0));

        CHECK_MIO(miopenExecuteFusionPlan(handle, fusePlan, xDesc, d_x, yDesc, d_y, args));
        CHECK_HIP(hipDeviceSynchronize());

        miopenDestroyOperatorArgs(args);
        miopenDestroyFusionPlan(fusePlan);
        miopenDestroyConvolutionDescriptor(convDesc);
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(wDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(biasDesc);
        miopenDestroy(handle);

        fprintf(stderr, "churn %d/%d OK\n", iter, N_CHURN);
    }

    fprintf(stderr, "ALL %d FUSION CHURNS COMPLETED\n", N_CHURN);
    hipFree(d_x); hipFree(d_w); hipFree(d_y); hipFree(d_bias);
    return 0;
}
