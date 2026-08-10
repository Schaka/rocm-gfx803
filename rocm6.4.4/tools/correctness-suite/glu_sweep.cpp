// GLU (Gated Linear Unit) forward correctness sweep. Only solver
// (GLUForward) requires dim==0: input's flat buffer is split into two equal
// halves by raw element count (not a real "axis" split), output = first_half
// * sigmoid(second_half). Formula confirmed from src/kernels/MIOpenGLU.cpp.
#include "common.hpp"

static bool run_one(int N)
{
    size_t in_elems = (size_t)2 * N;
    std::mt19937 rng(31);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> h_x(in_elems), h_ref(N), h_gpu(N);
    for(auto& v : h_x)
        v = dist(rng);
    for(int i = 0; i < N; i++)
    {
        double sig  = 1.0 / (1.0 + std::exp(-(double)h_x[N + i]));
        h_ref[i]    = (float)((double)h_x[i] * sig);
    }

    void *d_x, *d_y;
    CHECK_HIP(hipMalloc(&d_x, in_elems * 4));
    CHECK_HIP(hipMalloc(&d_y, (size_t)N * 4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), in_elems * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, (size_t)N * 4));

    miopenHandle_t handle;
    miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc, yDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);
    int xlens[1] = {(int)in_elems};
    miopenSetTensorDescriptor(xDesc, miopenFloat, 1, xlens, nullptr);
    int ylens[1] = {N};
    miopenSetTensorDescriptor(yDesc, miopenFloat, 1, ylens, nullptr);

    miopenStatus_t es = miopenGLUForward(handle, xDesc, d_x, yDesc, d_y, 0);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    if(es != miopenStatusSuccess)
    {
        printf("N=%d -> EXEC FAILED status=%d\n", N, (int)es);
        ok = false;
    }
    else
    {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, (size_t)N * 4, hipMemcpyDeviceToHost));
        double metric;
        bool used_absdiff;
        bool match = vectors_match(h_ref, h_gpu, &metric, &used_absdiff);
        printf("N=%d -> %s=%.5f%s\n", N, used_absdiff ? "maxabsdiff" : "cos", metric,
               match ? "  OK" : "  <-- WRONG");
        ok = match;
    }

    hipFree(d_x);
    hipFree(d_y);
    miopenDestroyTensorDescriptor(xDesc);
    miopenDestroyTensorDescriptor(yDesc);
    miopenDestroy(handle);
    return ok;
}

int main()
{
    int ns[] = {1, 2, 3, 4, 16, 100, 255, 256, 257, 1000, 24 * 32 * 100};
    int total = 0, wrong = 0;
    for(int n : ns)
    {
        total++;
        if(!run_one(n))
            wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong > 0 ? 1 : 0;
}
