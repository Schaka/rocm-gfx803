// Activation-forward correctness sweep (arch-blind solvers ActivFwdSolver0/1
// -- vectorized-read-unit and packed-vs-strided dispatch are the
// boundary-risk areas, analogous to the Winograd Ceiling()-padding bug:
// solver picks a read_unit (4/2/1) and a packed/strided kernel variant based
// on tensor shape). Sweeps every activation mode x several widths (incl.
// widths not divisible by 4 or 2, forcing read_unit=1) x packed and strided
// (padded stride) layouts. Reference computed on host per mode's documented
// formula.
#include "common.hpp"
#include <random>

static float activ_ref(miopenActivationMode_t mode, float x, double alpha, double beta, double gamma)
{
    switch(mode)
    {
    case miopenActivationPASTHRU: return x;
    case miopenActivationLOGISTIC: return (float)(1.0 / (1.0 + std::exp(-(double)x)));
    case miopenActivationTANH: return (float)(beta * std::tanh(alpha * x));
    case miopenActivationRELU: return x > 0 ? x : 0.f;
    case miopenActivationSOFTRELU: return (float)std::log1p(std::exp((double)x));
    case miopenActivationABS: return std::fabs(x);
    case miopenActivationPOWER:
    {
        double v = alpha + beta * x;
        return v <= 0 ? 0.f : (float)std::pow(v, gamma);
    }
    case miopenActivationCLIPPEDRELU: return (float)std::min(alpha, (double)std::max(0.f, x));
    case miopenActivationLEAKYRELU: return x > 0 ? x : (float)(alpha * x);
    case miopenActivationELU: return x > 0 ? x : (float)(alpha * (std::exp((double)x) - 1.0));
    default: return x;
    }
}

static const char* mode_name(miopenActivationMode_t m)
{
    switch(m)
    {
    case miopenActivationPASTHRU: return "PASTHRU";
    case miopenActivationLOGISTIC: return "LOGISTIC";
    case miopenActivationTANH: return "TANH";
    case miopenActivationRELU: return "RELU";
    case miopenActivationSOFTRELU: return "SOFTRELU";
    case miopenActivationABS: return "ABS";
    case miopenActivationPOWER: return "POWER";
    case miopenActivationCLIPPEDRELU: return "CLIPPEDRELU";
    case miopenActivationLEAKYRELU: return "LEAKYRELU";
    case miopenActivationELU: return "ELU";
    default: return "?";
    }
}

static bool run_one(miopenActivationMode_t mode, int N, int C, int H, int W, bool strided)
{
    double alpha = 1.0, beta = 1.0, gamma = 1.0;
    if(mode == miopenActivationCLIPPEDRELU)
        alpha = 6.0;
    if(mode == miopenActivationLEAKYRELU)
        alpha = 0.1;
    if(mode == miopenActivationELU)
        alpha = 1.0;
    if(mode == miopenActivationPOWER)
    {
        alpha = 0.5;
        beta  = 1.0;
        gamma = 2.0;
    }

    size_t elems    = (size_t)N * C * H * W;
    int strides[4]  = {C * H * W, H * W, W, 1}; // packed default
    if(strided)
    {
        // Padded stride on the height dim so W != stride -- forces the
        // non-packed "2DLite" kernel variant instead of the packed one.
        strides[2] = W + 3;
        strides[1] = strides[2] * H;
        strides[0] = strides[1] * C;
    }
    size_t buf_elems = strided ? (size_t)N * strides[0] : elems;

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> h_x(buf_elems, 0.f), h_ref(buf_elems, 0.f), h_gpu(buf_elems, 0.f);
    std::vector<bool> valid(buf_elems, false);
    for(int n = 0; n < N; n++)
        for(int c = 0; c < C; c++)
            for(int h = 0; h < H; h++)
                for(int w = 0; w < W; w++)
                {
                    size_t off = (size_t)n * strides[0] + (size_t)c * strides[1] +
                                 (size_t)h * strides[2] + (size_t)w * strides[3];
                    h_x[off]   = dist(rng);
                    valid[off] = true;
                }
    for(size_t i = 0; i < buf_elems; i++)
        if(valid[i])
            h_ref[i] = activ_ref(mode, h_x[i], alpha, beta, gamma);

    void *d_x, *d_y;
    CHECK_HIP(hipMalloc(&d_x, buf_elems * 4));
    CHECK_HIP(hipMalloc(&d_y, buf_elems * 4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), buf_elems * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, buf_elems * 4));

    miopenHandle_t handle;
    miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc, yDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);

    if(!strided)
    {
        miopenSet4dTensorDescriptor(xDesc, miopenFloat, N, C, H, W);
        miopenSet4dTensorDescriptor(yDesc, miopenFloat, N, C, H, W);
    }
    else
    {
        int lens[4] = {N, C, H, W};
        miopenSetTensorDescriptor(xDesc, miopenFloat, 4, lens, strides);
        miopenSetTensorDescriptor(yDesc, miopenFloat, 4, lens, strides);
    }

    miopenActivationDescriptor_t activDesc;
    miopenCreateActivationDescriptor(&activDesc);
    miopenSetActivationDescriptor(activDesc, mode, alpha, beta, gamma);

    float a = 1.0f, b = 0.0f;
    miopenStatus_t es = miopenActivationForward(handle, activDesc, &a, xDesc, d_x, &b, yDesc, d_y);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    if(es != miopenStatusSuccess)
    {
        printf("mode=%-12s N=%d C=%d H=%d W=%d strided=%d -> EXEC FAILED status=%d\n",
               mode_name(mode), N, C, H, W, strided, (int)es);
        ok = false;
    }
    else
    {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, buf_elems * 4, hipMemcpyDeviceToHost));
        double metric;
        bool used_absdiff;
        bool match = vectors_match(h_ref, h_gpu, &metric, &used_absdiff);
        printf("mode=%-12s N=%d C=%d H=%d W=%d strided=%d -> %s=%.5f%s\n",
               mode_name(mode), N, C, H, W, strided, used_absdiff ? "maxabsdiff" : "cos", metric,
               match ? "  OK" : "  <-- WRONG");
        ok = match;
    }

    hipFree(d_x);
    hipFree(d_y);
    miopenDestroyActivationDescriptor(activDesc);
    miopenDestroyTensorDescriptor(xDesc);
    miopenDestroyTensorDescriptor(yDesc);
    miopenDestroy(handle);
    return ok;
}

int main()
{
    miopenActivationMode_t modes[] = {
        miopenActivationPASTHRU, miopenActivationLOGISTIC, miopenActivationTANH,
        miopenActivationRELU,    miopenActivationSOFTRELU, miopenActivationABS,
        miopenActivationPOWER,   miopenActivationCLIPPEDRELU, miopenActivationLEAKYRELU,
        miopenActivationELU};
    // Widths chosen to hit read_unit=4 (W%4==0), read_unit=2 (W%2==0,%4!=0),
    // and read_unit=1 (W odd) boundaries.
    struct Shape
    {
        int N, C, H, W;
    };
    Shape shapes[] = {
        {1, 24, 32, 100}, // %4==0
        {1, 24, 32, 98},  // %2==0, %4!=0
        {1, 24, 32, 97},  // odd
        {2, 8, 4, 4},
        {1, 1, 1, 1},
        {1, 64, 16, 16},
    };

    int total = 0, wrong = 0;
    for(auto mode : modes)
        for(auto& s : shapes)
            for(int strided = 0; strided <= 1; strided++)
            {
                total++;
                if(!run_one(mode, s.N, s.C, s.H, s.W, strided))
                    wrong++;
            }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong > 0 ? 1 : 0;
}
