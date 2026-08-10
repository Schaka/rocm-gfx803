// gfx803 LayerNorm forward correctness sweep (bucket C, arch-blind
// LayernormForward -- generic OCL kernel, LOCAL_SIZE=256 reduction, never
// gfx803-specific-verified). 2D tensor (outer, norm_dim), normalized_dim=1.
// Sweeps norm_dim straddling LOCAL_SIZE=256.
#include "common.hpp"

static void cpu_ln(const std::vector<float>& x, std::vector<float>& y,
                    const std::vector<float>& w, const std::vector<float>& b,
                    int outer, int C, double eps) {
    y.resize(x.size());
    for (int o=0;o<outer;o++) {
        double mean=0;
        for (int c=0;c<C;c++) mean += x[(size_t)o*C+c];
        mean /= C;
        double var=0;
        for (int c=0;c<C;c++) { double d = x[(size_t)o*C+c]-mean; var += d*d; }
        var /= C;
        double inv = 1.0/std::sqrt(var+eps);
        for (int c=0;c<C;c++) {
            size_t idx = (size_t)o*C+c;
            y[idx] = (float)(((double)x[idx]-mean)*inv*w[c] + b[c]);
        }
    }
}

static bool run_one(int outer, int C) {
    size_t elems = (size_t)outer*C;
    std::mt19937 rng(9);
    std::normal_distribution<float> dist(0.0f, 1.5f);
    std::vector<float> h_x(elems), h_w(C), h_b(C), h_ref, h_gpu(elems), h_mean(outer), h_rstd(outer);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_w) v = dist(rng)*0.3f + 1.0f;
    for (auto& v : h_b) v = dist(rng)*0.1f;
    double eps = 1e-5;
    cpu_ln(h_x, h_ref, h_w, h_b, outer, C, eps);

    void *d_x,*d_y,*d_w,*d_b,*d_mean,*d_rstd;
    CHECK_HIP(hipMalloc(&d_x, elems*4));
    CHECK_HIP(hipMalloc(&d_y, elems*4));
    CHECK_HIP(hipMalloc(&d_w, C*4));
    CHECK_HIP(hipMalloc(&d_b, C*4));
    CHECK_HIP(hipMalloc(&d_mean, outer*4));
    CHECK_HIP(hipMalloc(&d_rstd, outer*4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), elems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_w, h_w.data(), C*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.data(), C*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, elems*4));

    miopenHandle_t handle; miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc,wDesc,bDesc,yDesc,meanDesc,rstdDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&wDesc);
    miopenCreateTensorDescriptor(&bDesc);
    miopenCreateTensorDescriptor(&yDesc);
    miopenCreateTensorDescriptor(&meanDesc);
    miopenCreateTensorDescriptor(&rstdDesc);
    int xlens[2] = {outer, C};
    miopenSetTensorDescriptor(xDesc, miopenFloat, 2, xlens, nullptr);
    miopenSetTensorDescriptor(yDesc, miopenFloat, 2, xlens, nullptr);
    int wlens[1] = {C};
    miopenSetTensorDescriptor(wDesc, miopenFloat, 1, wlens, nullptr);
    miopenSetTensorDescriptor(bDesc, miopenFloat, 1, wlens, nullptr);
    int mlens[1] = {outer};
    miopenSetTensorDescriptor(meanDesc, miopenFloat, 1, mlens, nullptr);
    miopenSetTensorDescriptor(rstdDesc, miopenFloat, 1, mlens, nullptr);

    miopenStatus_t es = miopenLayerNormForward(handle, MIOPEN_WEIGHT_BIAS,
        xDesc, d_x, wDesc, d_w, bDesc, d_b, (float)eps, 1,
        yDesc, d_y, meanDesc, d_mean, rstdDesc, d_rstd);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    if (es != miopenStatusSuccess) {
        printf("outer=%d C=%d -> EXEC FAILED status=%d\n", outer,C,(int)es);
        ok = false;
    } else {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, elems*4, hipMemcpyDeviceToHost));
        double cs = cos_sim(h_ref, h_gpu);
        printf("outer=%d C=%d -> cos=%.5f%s\n", outer,C,cs, cs<0.999 ? "  <-- WRONG" : "  OK");
        if (cs<0.999) ok=false;
    }

    hipFree(d_x); hipFree(d_y); hipFree(d_w); hipFree(d_b); hipFree(d_mean); hipFree(d_rstd);
    miopenDestroyTensorDescriptor(xDesc); miopenDestroyTensorDescriptor(wDesc);
    miopenDestroyTensorDescriptor(bDesc); miopenDestroyTensorDescriptor(yDesc);
    miopenDestroyTensorDescriptor(meanDesc); miopenDestroyTensorDescriptor(rstdDesc);
    miopenDestroy(handle);
    return ok;
}

int main() {
    struct Case { int outer, C; };
    Case cases[] = {
        {4, 64}, {4, 128}, {4, 255}, {4, 256}, {4, 257}, {4, 512}, {4, 768}, {4, 1024},
        {1, 1}, {8, 1000}, {2, 100}, {16, 24}
    };
    int total=0, wrong=0;
    for (auto& c : cases) {
        total++;
        if (!run_one(c.outer, c.C)) wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong>0 ? 1 : 0;
}
