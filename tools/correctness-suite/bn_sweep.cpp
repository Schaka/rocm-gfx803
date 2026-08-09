// gfx803 BatchNorm forward-inference correctness sweep (bucket C,
// BnFwdInference -- simple gate, no size-boundary math visible in
// IsApplicable, but never gfx803-specific-verified). Spatial + PerActivation
// modes, several channel/spatial sizes, fp32.
#include "common.hpp"

static void cpu_bn(const std::vector<float>& x, std::vector<float>& y,
                    const std::vector<float>& scale, const std::vector<float>& shift,
                    const std::vector<float>& mean, const std::vector<float>& var,
                    int C,int H,int W, double eps, bool per_activation) {
    y.resize(x.size());
    for (int c=0;c<C;c++) {
        for (int hw=0; hw<H*W; hw++) {
            size_t idx = (size_t)c*H*W + hw;
            size_t pidx = per_activation ? idx : (size_t)c; // param tensor: (1,C,H,W) vs (1,C,1,1)
            double inv = 1.0 / std::sqrt((double)var[pidx] + eps);
            y[idx] = (float)(((double)x[idx] - mean[pidx]) * inv * scale[pidx] + shift[pidx]);
        }
    }
}

static bool run_one(miopenBatchNormMode_t mode, int C, int H, int W) {
    size_t elems = (size_t)C*H*W;
    bool per_activation = (mode == miopenBNPerActivation);
    size_t pelems = per_activation ? elems : (size_t)C;
    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> vdist(0.1f, 2.0f);
    std::vector<float> h_x(elems), h_scale(pelems), h_shift(pelems), h_mean(pelems), h_var(pelems), h_ref, h_gpu(elems);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_scale) v = dist(rng)*0.5f+1.0f;
    for (auto& v : h_shift) v = dist(rng)*0.1f;
    for (auto& v : h_mean) v = dist(rng)*0.3f;
    for (auto& v : h_var) v = vdist(rng);
    double eps = 1e-5;
    cpu_bn(h_x, h_ref, h_scale, h_shift, h_mean, h_var, C, H, W, eps, per_activation);

    void *d_x,*d_y,*d_scale,*d_shift,*d_mean,*d_var;
    CHECK_HIP(hipMalloc(&d_x, elems*4));
    CHECK_HIP(hipMalloc(&d_y, elems*4));
    CHECK_HIP(hipMalloc(&d_scale, pelems*4));
    CHECK_HIP(hipMalloc(&d_shift, pelems*4));
    CHECK_HIP(hipMalloc(&d_mean, pelems*4));
    CHECK_HIP(hipMalloc(&d_var, pelems*4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), elems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_scale, h_scale.data(), pelems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_shift, h_shift.data(), pelems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_mean, h_mean.data(), pelems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_var, h_var.data(), pelems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, elems*4));

    miopenHandle_t handle; miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc,yDesc,bnDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);
    miopenCreateTensorDescriptor(&bnDesc);
    miopenSet4dTensorDescriptor(xDesc, miopenFloat, 1, C, H, W);
    miopenSet4dTensorDescriptor(yDesc, miopenFloat, 1, C, H, W);
    miopenDeriveBNTensorDescriptor(bnDesc, xDesc, mode);

    float a=1.0f,b=0.0f;
    miopenStatus_t es = miopenBatchNormalizationForwardInference(
        handle, mode, &a, &b, xDesc, d_x, yDesc, d_y, bnDesc,
        d_scale, d_shift, d_mean, d_var, eps);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    const char* mn = (mode == miopenBNSpatial) ? "SPATIAL" : "PERACT";
    if (es != miopenStatusSuccess) {
        printf("mode=%-8s C=%d H=%d W=%d -> EXEC FAILED status=%d\n", mn,C,H,W,(int)es);
        ok = false;
    } else {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, elems*4, hipMemcpyDeviceToHost));
        double cs = cos_sim(h_ref, h_gpu);
        printf("mode=%-8s C=%d H=%d W=%d -> cos=%.5f%s\n", mn,C,H,W,cs, cs<0.999 ? "  <-- WRONG" : "  OK");
        if (cs<0.999) ok=false;
    }

    hipFree(d_x); hipFree(d_y); hipFree(d_scale); hipFree(d_shift); hipFree(d_mean); hipFree(d_var);
    miopenDestroyTensorDescriptor(xDesc); miopenDestroyTensorDescriptor(yDesc); miopenDestroyTensorDescriptor(bnDesc);
    miopenDestroy(handle);
    return ok;
}

int main() {
    miopenBatchNormMode_t modes[] = {miopenBNSpatial, miopenBNPerActivation};
    struct Case { int C,H,W; };
    Case cases[] = {
        {24, 32, 100}, {64, 16, 16}, {1, 1, 1}, {3, 224, 224}, {512, 4, 4}, {16, 100, 1}
    };
    int total=0, wrong=0;
    for (auto mode : modes) for (auto& c : cases) {
        total++;
        if (!run_one(mode, c.C,c.H,c.W)) wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong>0 ? 1 : 0;
}
