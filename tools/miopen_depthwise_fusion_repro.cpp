// Targets the exact depthwise conv shape decoded from the real crash
// (K=240, C=1 per group, group_count=240, R=S=3, stride1, spatial 16x50,
// same-padding), via the fusion-plan API (Conv+Bias+Activation) that
// dispatches ConvOclDirectFwdFused -> MIOpenConvUniBatchNormActiv -- the
// exact kernel that faulted with an out-of-bounds weight read.
// ConvOclDirectFwd::IsApplicable skips its filter-size gate entirely for
// GroupCount() != 1, which is the lead being tested here.
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP err %s at line %d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CHECK_MIO(x) do { miopenStatus_t s=(x); if(s!=miopenStatusSuccess){fprintf(stderr,"MIOpen err %d at line %d\n",(int)s,__LINE__);exit(1);} } while(0)

int main(int argc, char** argv) {
    int N_ITERS = argc > 1 ? atoi(argv[1]) : 1;
    // depthwise: N=1, C=240 total (1 per group), K=240 total (1 per group), groups=240, R=S=3
    const int N = 1, C = 240, H = 16, W = 50, K = 240, R = 3, S = 3, GROUPS = 240;

    size_t x_elems = (size_t)N*C*H*W;
    size_t w_elems = (size_t)K*(C/GROUPS)*R*S; // = K*1*3*3
    int OH = H, OW = W; // stride1, pad1 (same)
    size_t y_elems = (size_t)N*K*OH*OW;
    size_t bias_elems = K;

    fprintf(stderr, "shape: N=%d C=%d H=%d W=%d K=%d R=%d S=%d groups=%d OH=%d OW=%d\n", N,C,H,W,K,R,S,GROUPS,OH,OW);
    fprintf(stderr, "x_elems=%zu w_elems=%zu y_elems=%zu\n", x_elems, w_elems, y_elems);

    void *d_x, *d_w, *d_y, *d_bias;
    CHECK_HIP(hipMalloc(&d_x, x_elems*4));
    CHECK_HIP(hipMalloc(&d_w, w_elems*4));
    CHECK_HIP(hipMalloc(&d_y, y_elems*4));
    CHECK_HIP(hipMalloc(&d_bias, bias_elems*4));
    CHECK_HIP(hipMemset(d_x, 0, x_elems*4));
    CHECK_HIP(hipMemset(d_w, 0, w_elems*4));
    CHECK_HIP(hipMemset(d_bias, 0, bias_elems*4));

    for (int iter = 0; iter < N_ITERS; iter++) {
        miopenHandle_t handle;
        CHECK_MIO(miopenCreate(&handle));

        miopenTensorDescriptor_t xDesc, wDesc, yDesc, biasDesc;
        CHECK_MIO(miopenCreateTensorDescriptor(&xDesc));
        CHECK_MIO(miopenCreateTensorDescriptor(&wDesc));
        CHECK_MIO(miopenCreateTensorDescriptor(&yDesc));
        CHECK_MIO(miopenCreateTensorDescriptor(&biasDesc));
        CHECK_MIO(miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W));
        CHECK_MIO(miopenSet4dTensorDescriptor(wDesc, miopenFloat, K, C/GROUPS, R, S));
        CHECK_MIO(miopenSet4dTensorDescriptor(yDesc, miopenFloat, N, K, OH, OW));
        CHECK_MIO(miopenSet4dTensorDescriptor(biasDesc, miopenFloat, 1, K, 1, 1));

        miopenConvolutionDescriptor_t convDesc;
        CHECK_MIO(miopenCreateConvolutionDescriptor(&convDesc));
        CHECK_MIO(miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 1, 1, 1, 1, 1, 1));
        CHECK_MIO(miopenSetConvolutionGroupCount(convDesc, GROUPS));

        miopenFusionPlanDescriptor_t fusePlan;
        CHECK_MIO(miopenCreateFusionPlan(&fusePlan, miopenVerticalFusion, xDesc));

        miopenFusionOpDescriptor_t convOp, biasOp, activOp;
        CHECK_MIO(miopenCreateOpConvForward(fusePlan, &convOp, convDesc, wDesc));
        CHECK_MIO(miopenCreateOpBiasForward(fusePlan, &biasOp, biasDesc));
        CHECK_MIO(miopenCreateOpActivationForward(fusePlan, &activOp, miopenActivationRELU));

        miopenStatus_t compile_status = miopenCompileFusionPlan(handle, fusePlan);
        if (compile_status != miopenStatusSuccess) {
            fprintf(stderr, "iter %d: compile failed status=%d\n", iter, (int)compile_status);
            miopenDestroyFusionPlan(fusePlan);
            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(xDesc);
            miopenDestroyTensorDescriptor(wDesc);
            miopenDestroyTensorDescriptor(yDesc);
            miopenDestroyTensorDescriptor(biasDesc);
            miopenDestroy(handle);
            continue;
        }
        fprintf(stderr, "iter %d: compiled OK\n", iter);

        miopenOperatorArgs_t args;
        CHECK_MIO(miopenCreateOperatorArgs(&args));
        float alpha = 1.0f, beta = 0.0f;
        CHECK_MIO(miopenSetOpArgsConvForward(args, convOp, &alpha, &beta, d_w));
        CHECK_MIO(miopenSetOpArgsBiasForward(args, biasOp, &alpha, &beta, d_bias));
        CHECK_MIO(miopenSetOpArgsActivForward(args, activOp, &alpha, &beta, 0.0, 0.0, 0.0));

        CHECK_MIO(miopenExecuteFusionPlan(handle, fusePlan, xDesc, d_x, yDesc, d_y, args));
        CHECK_HIP(hipDeviceSynchronize());
        fprintf(stderr, "iter %d: executed OK\n", iter);

        miopenDestroyOperatorArgs(args);
        miopenDestroyFusionPlan(fusePlan);
        miopenDestroyConvolutionDescriptor(convDesc);
        miopenDestroyTensorDescriptor(xDesc);
        miopenDestroyTensorDescriptor(wDesc);
        miopenDestroyTensorDescriptor(yDesc);
        miopenDestroyTensorDescriptor(biasDesc);
        miopenDestroy(handle);
    }

    fprintf(stderr, "ALL %d ITERS COMPLETED\n", N_ITERS);
    hipFree(d_x); hipFree(d_w); hipFree(d_y); hipFree(d_bias);
    return 0;
}
