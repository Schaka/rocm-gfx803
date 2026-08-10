// gfx803 GroupNorm forward correctness sweep (bucket C, arch-blind
// GroupNormForward). IsApplicable requires N*num_groups>=32 and
// C/num_groups<64 -- boundary risk zone, never gfx803-verified.
#include "common.hpp"

static void cpu_gn(const std::vector<float>& x, std::vector<float>& y,
                    const std::vector<float>& w, const std::vector<float>& b,
                    int N,int C,int HW,int num_groups, double eps) {
    y.resize(x.size());
    int Cg = C/num_groups;
    for (int n=0;n<N;n++) {
        for (int g=0; g<num_groups; g++) {
            double mean=0; size_t cnt=(size_t)Cg*HW;
            for (int cc=0;cc<Cg;cc++) { int c=g*Cg+cc; for(int hw=0;hw<HW;hw++) mean += x[((size_t)n*C+c)*HW+hw]; }
            mean /= cnt;
            double var=0;
            for (int cc=0;cc<Cg;cc++) { int c=g*Cg+cc; for(int hw=0;hw<HW;hw++) { double d=x[((size_t)n*C+c)*HW+hw]-mean; var+=d*d; } }
            var /= cnt;
            double inv = 1.0/std::sqrt(var+eps);
            for (int cc=0;cc<Cg;cc++) {
                int c=g*Cg+cc;
                for (int hw=0;hw<HW;hw++) {
                    size_t idx=((size_t)n*C+c)*HW+hw;
                    y[idx] = (float)(((double)x[idx]-mean)*inv*w[c] + b[c]);
                }
            }
        }
    }
}

static bool run_one(int N,int C,int H,int W,int num_groups) {
    int HW=H*W;
    size_t elems=(size_t)N*C*HW;
    std::mt19937 rng(13);
    std::normal_distribution<float> dist(0.0f,1.5f);
    std::vector<float> h_x(elems), h_w(C), h_b(C), h_ref, h_gpu(elems), h_mean((size_t)N*num_groups), h_rstd((size_t)N*num_groups);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_w) v = dist(rng)*0.3f+1.0f;
    for (auto& v : h_b) v = dist(rng)*0.1f;
    double eps=1e-5;
    cpu_gn(h_x,h_ref,h_w,h_b,N,C,HW,num_groups,eps);

    void *d_x,*d_y,*d_w,*d_b,*d_mean,*d_rstd;
    CHECK_HIP(hipMalloc(&d_x, elems*4));
    CHECK_HIP(hipMalloc(&d_y, elems*4));
    CHECK_HIP(hipMalloc(&d_w, C*4));
    CHECK_HIP(hipMalloc(&d_b, C*4));
    CHECK_HIP(hipMalloc(&d_mean, N*num_groups*4));
    CHECK_HIP(hipMalloc(&d_rstd, N*num_groups*4));
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
    miopenSet4dTensorDescriptor(xDesc, miopenFloat, N,C,H,W);
    miopenSet4dTensorDescriptor(yDesc, miopenFloat, N,C,H,W);
    int wlens[1]={C};
    miopenSetTensorDescriptor(wDesc, miopenFloat, 1, wlens, nullptr);
    miopenSetTensorDescriptor(bDesc, miopenFloat, 1, wlens, nullptr);
    int mlens[1]={N*num_groups};
    miopenSetTensorDescriptor(meanDesc, miopenFloat, 1, mlens, nullptr);
    miopenSetTensorDescriptor(rstdDesc, miopenFloat, 1, mlens, nullptr);

    miopenStatus_t es = miopenGroupNormForward(handle, MIOPEN_WEIGHT_BIAS,
        xDesc, d_x, wDesc, d_w, bDesc, d_b, (uint64_t)num_groups, (float)eps,
        yDesc, d_y, meanDesc, d_mean, rstdDesc, d_rstd);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok=true;
    if (es != miopenStatusSuccess) {
        printf("N=%d C=%d H=%d W=%d groups=%d -> EXEC FAILED status=%d\n",N,C,H,W,num_groups,(int)es);
        ok=false;
    } else {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, elems*4, hipMemcpyDeviceToHost));
        double cs = cos_sim(h_ref, h_gpu);
        printf("N=%d C=%d H=%d W=%d groups=%d -> cos=%.5f%s\n",N,C,H,W,num_groups,cs, cs<0.999?"  <-- WRONG":"  OK");
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
    struct Case { int N,C,H,W,groups; };
    Case cases[] = {
        {8, 64, 4, 4, 4},    // N*g=32 boundary, C/g=16
        {32, 4, 4, 4, 1},    // N*g=32 boundary, C/g=4
        {1, 64, 8, 8, 32},   // N*g=32, C/g=2
        {16, 128, 4, 4, 8},  // N*g=128, C/g=16
        {8, 256, 2, 2, 4},   // C/g=64 -- should be REJECTED (>=64)
        {8, 252, 2, 2, 4},   // C/g=63, just under boundary
        {4, 32, 1, 1, 1},    // N*g=4 -- should be REJECTED (<32)
    };
    int total=0, wrong=0;
    for (auto& c : cases) {
        total++;
        if (!run_one(c.N,c.C,c.H,c.W,c.groups)) wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG (EXEC FAILED on rejected-boundary cases is expected, not a bug)\n", total, wrong);
    return 0;
}
