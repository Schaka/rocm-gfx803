// RoPE (Rotary Position Embedding) forward correctness sweep -- relevant to
// any modern transformer decoder in the stack. Formula confirmed from
// src/kernels/MIOpenRoPE.cpp: interleaved-pair rotate_half (even index i:
// rotate_half[i]=-x[i+1]; odd index i: rotate_half[i]=x[i-1]), cos/sin
// indexed via gid % rotary_numel (broadcast across outer dims), output =
// x*cos + rotate_half(x)*sin.
#include "common.hpp"

static bool run_one(int outer, int rotary_numel)
{
    size_t elems = (size_t)outer * rotary_numel;
    std::mt19937 rng(41);
    std::normal_distribution<float> dist(0.0f, 1.5f);
    std::uniform_real_distribution<float> angle(-3.14159f, 3.14159f);
    std::vector<float> h_x(elems), h_cos(rotary_numel), h_sin(rotary_numel), h_ref(elems),
        h_gpu(elems);
    for(auto& v : h_x)
        v = dist(rng);
    for(int i = 0; i < rotary_numel; i++)
    {
        float a    = angle(rng);
        h_cos[i]   = std::cos(a);
        h_sin[i]   = std::sin(a);
    }
    for(size_t gid = 0; gid < elems; gid++)
    {
        size_t freqs_id = gid % rotary_numel;
        float rot = (gid % 2 == 0) ? -h_x[gid + 1] : h_x[gid - 1];
        h_ref[gid] = h_x[gid] * h_cos[freqs_id] + rot * h_sin[freqs_id];
    }

    void *d_x, *d_cos, *d_sin, *d_y;
    CHECK_HIP(hipMalloc(&d_x, elems * 4));
    CHECK_HIP(hipMalloc(&d_cos, (size_t)rotary_numel * 4));
    CHECK_HIP(hipMalloc(&d_sin, (size_t)rotary_numel * 4));
    CHECK_HIP(hipMalloc(&d_y, elems * 4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), elems * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_cos, h_cos.data(), (size_t)rotary_numel * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sin, h_sin.data(), (size_t)rotary_numel * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, elems * 4));

    miopenHandle_t handle;
    miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc, cosDesc, sinDesc, yDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&cosDesc);
    miopenCreateTensorDescriptor(&sinDesc);
    miopenCreateTensorDescriptor(&yDesc);
    int xlens[2] = {outer, rotary_numel};
    miopenSetTensorDescriptor(xDesc, miopenFloat, 2, xlens, nullptr);
    miopenSetTensorDescriptor(yDesc, miopenFloat, 2, xlens, nullptr);
    int clens[1] = {rotary_numel};
    miopenSetTensorDescriptor(cosDesc, miopenFloat, 1, clens, nullptr);
    miopenSetTensorDescriptor(sinDesc, miopenFloat, 1, clens, nullptr);

    miopenStatus_t es =
        miopenRoPEForward(handle, xDesc, d_x, cosDesc, d_cos, sinDesc, d_sin, yDesc, d_y);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    if(es != miopenStatusSuccess)
    {
        printf("outer=%d rotary_numel=%d -> EXEC FAILED status=%d\n", outer, rotary_numel, (int)es);
        ok = false;
    }
    else
    {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, elems * 4, hipMemcpyDeviceToHost));
        double metric;
        bool used_absdiff;
        bool match = vectors_match(h_ref, h_gpu, &metric, &used_absdiff);
        printf("outer=%d rotary_numel=%d -> %s=%.5f%s\n", outer, rotary_numel,
               used_absdiff ? "maxabsdiff" : "cos", metric, match ? "  OK" : "  <-- WRONG");
        ok = match;
    }

    hipFree(d_x);
    hipFree(d_cos);
    hipFree(d_sin);
    hipFree(d_y);
    miopenDestroyTensorDescriptor(xDesc);
    miopenDestroyTensorDescriptor(cosDesc);
    miopenDestroyTensorDescriptor(sinDesc);
    miopenDestroyTensorDescriptor(yDesc);
    miopenDestroy(handle);
    return ok;
}

int main()
{
    struct Case
    {
        int outer, rotary_numel;
    };
    Case cases[] = {{4, 64}, {4, 128}, {1, 2}, {8, 256}, {2, 1024}, {16, 32}};
    int total = 0, wrong = 0;
    for(auto& c : cases)
    {
        total++;
        if(!run_one(c.outer, c.rotary_numel))
            wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong > 0 ? 1 : 0;
}
