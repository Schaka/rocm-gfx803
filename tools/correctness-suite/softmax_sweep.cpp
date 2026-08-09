// gfx803 softmax correctness sweep (bucket C, arch-blind Softmax solver).
// Boundary risk: num_batch/batch_size/u_batch_size dispatch keyed on
// vector_size vs 256 threshold with nextPow2 rounding -- same shape as
// prior bugs (size-threshold dispatch math, never gfx803-specific,
// never gfx803-verified). Sweeps vector_size straddling 256, both
// INSTANCE/CHANNEL modes, both spatial and non-spatial shapes.
#include "common.hpp"

// CHANNEL mode: softmax over C, independently per (N,H,W).
// INSTANCE mode: softmax over C*H*W, independently per N.
static void cpu_softmax(const std::vector<float>& x, std::vector<float>& y,
                         int N,int C,int H,int W, miopenSoftmaxMode_t mode) {
    y.resize(x.size());
    if (mode == MIOPEN_SOFTMAX_MODE_CHANNEL) {
        for (int n=0;n<N;n++) for (int h=0;h<H;h++) for (int w=0;w<W;w++) {
            double mx = -1e300;
            for (int c=0;c<C;c++) mx = std::max(mx, (double)x[((size_t)n*C+c)*H*W + h*W + w]);
            double sum=0;
            std::vector<double> e(C);
            for (int c=0;c<C;c++) { e[c] = std::exp((double)x[((size_t)n*C+c)*H*W+h*W+w]-mx); sum+=e[c]; }
            for (int c=0;c<C;c++) y[((size_t)n*C+c)*H*W+h*W+w] = (float)(e[c]/sum);
        }
    } else { // INSTANCE
        for (int n=0;n<N;n++) {
            size_t base = (size_t)n*C*H*W;
            size_t len = (size_t)C*H*W;
            double mx=-1e300;
            for (size_t i=0;i<len;i++) mx = std::max(mx, (double)x[base+i]);
            std::vector<double> e(len);
            double sum=0;
            for (size_t i=0;i<len;i++) { e[i]=std::exp((double)x[base+i]-mx); sum+=e[i]; }
            for (size_t i=0;i<len;i++) y[base+i] = (float)(e[i]/sum);
        }
    }
}

static bool run_one(miopenSoftmaxMode_t mode, int N,int C,int H,int W) {
    size_t elems = (size_t)N*C*H*W;
    std::mt19937 rng(5);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    std::vector<float> h_x(elems), h_ref, h_gpu(elems);
    for (auto& v : h_x) v = dist(rng);
    cpu_softmax(h_x, h_ref, N,C,H,W, mode);

    void *d_x,*d_y;
    CHECK_HIP(hipMalloc(&d_x, elems*4));
    CHECK_HIP(hipMalloc(&d_y, elems*4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), elems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, elems*4));

    miopenHandle_t handle; miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc,yDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);
    miopenSet4dTensorDescriptor(xDesc, miopenFloat, N,C,H,W);
    miopenSet4dTensorDescriptor(yDesc, miopenFloat, N,C,H,W);

    float a=1.0f,b=0.0f;
    miopenStatus_t es = miopenSoftmaxForward_V2(handle, &a, xDesc, d_x, &b, yDesc, d_y,
                                                 MIOPEN_SOFTMAX_ACCURATE, mode);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    const char* mn = (mode == MIOPEN_SOFTMAX_MODE_CHANNEL) ? "CHANNEL" : "INSTANCE";
    if (es != miopenStatusSuccess) {
        printf("mode=%-8s N=%d C=%d H=%d W=%d -> EXEC FAILED status=%d\n", mn,N,C,H,W,(int)es);
        ok = false;
    } else {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, elems*4, hipMemcpyDeviceToHost));
        double cs = cos_sim(h_ref, h_gpu);
        printf("mode=%-8s N=%d C=%d H=%d W=%d -> cos=%.5f%s\n", mn,N,C,H,W,cs, cs<0.999 ? "  <-- WRONG" : "  OK");
        if (cs<0.999) ok=false;
    }

    hipFree(d_x); hipFree(d_y);
    miopenDestroyTensorDescriptor(xDesc); miopenDestroyTensorDescriptor(yDesc);
    miopenDestroy(handle);
    return ok;
}

int main() {
    miopenSoftmaxMode_t modes[] = {MIOPEN_SOFTMAX_MODE_CHANNEL, MIOPEN_SOFTMAX_MODE_INSTANCE};
    // vector_size for CHANNEL mode is C; for INSTANCE mode it's C*H*W.
    // Straddle the 256 num_batch threshold from both sides.
    struct Case { int N,C,H,W; };
    Case cases[] = {
        {1, 128, 1, 1},   // vector_size 128 < 256
        {1, 255, 1, 1},   // just under 256
        {1, 256, 1, 1},   // exactly 256
        {1, 257, 1, 1},   // just over 256
        {1, 512, 1, 1},
        {1, 1000, 1, 1},  // classifier-head-sized
        {2, 24, 32, 100}, // real CLAP-ish spatial, CHANNEL mode vector_size=24
        {1, 1, 1, 1},
        {4, 10, 1, 1},
        {1, 8, 8, 8},     // INSTANCE vector_size = 512
        {1, 4, 4, 4},     // INSTANCE vector_size = 64 < 256
    };
    int total=0, wrong=0;
    for (auto mode : modes) for (auto& c : cases) {
        total++;
        if (!run_one(mode, c.N,c.C,c.H,c.W)) wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong>0 ? 1 : 0;
}
