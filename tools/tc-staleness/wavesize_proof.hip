// Faithful replica of rocSOLVER's wave-partial pattern (lapack_device_functions.hpp
// LACN2 / lange / larfg / larf / latrd):
//     __shared__ S sval[BLOCK / WarpSize];
//     reduce_wave_sum(sum);                    // every lane holds the full-wave sum
//     if (tid % WarpSize == 0) sval[tid / WarpSize] = sum;
//     combine sval[0 .. BLOCK/WarpSize-1]
// Compiled with WarpSize = 32 (what upstream picks for __GFX8__) and 64 on a
// wave64 device, against a hand-counted expectation.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK 256
#define N 4096

__device__ float wave_sum_full(float v)
{
    // butterfly over a full 64-lane wave (what reduce_wave_sum delivers)
#pragma unroll
    for(int o = 32; o > 0; o >>= 1)
        v += __shfl_down(v, o, 64);
    v = __shfl(v, 63, 64);
    return v;
}

template <int WS>
__global__ void partial_pattern(const float* x, float* out)
{
    int tid = threadIdx.x;
    float sum = 0.f;
    for(int i = tid; i < N; i += BLOCK)
        sum += std::abs(x[i]);
    sum = wave_sum_full(sum);
    __shared__ float sval[BLOCK / WS];
    if(tid % WS == 0)
        sval[tid / WS] = sum;
    __syncthreads();
    if(tid == 0)
    {
        float t = sval[0];
        for(int k = 1; k < BLOCK / WS; k++)
            t += sval[k];
        *out = t;
    }
}

int main()
{
    float* x;
    float* o;
    hipMalloc(&x, N * sizeof(float));
    hipMalloc(&o, sizeof(float));
    float h[N];
    double expect = 0.0;
    for(int i = 0; i < N; i++)
    {
        h[i] = 1.0f + (i % 7) * 0.5f;
        expect += h[i];
    }
    hipMemcpy(x, h, N * sizeof(float), hipMemcpyHostToDevice);
    float got32 = 0, got64 = 0;
    partial_pattern<32><<<1, BLOCK>>>(x, o);
    printf("launch32 err: %s\n", hipGetErrorString(hipGetLastError()));
    hipDeviceSynchronize();
    printf("sync32  err: %s  rc=%s\n", hipGetErrorString(hipGetLastError()),
           hipGetErrorString(hipMemcpy(&got32, o, sizeof(float), hipMemcpyDeviceToHost)));
    partial_pattern<64><<<1, BLOCK>>>(x, o);
    printf("launch64 err: %s\n", hipGetErrorString(hipGetLastError()));
    hipDeviceSynchronize();
    printf("sync64  err: %s  rc=%s\n", hipGetErrorString(hipGetLastError()),
           hipGetErrorString(hipMemcpy(&got64, o, sizeof(float), hipMemcpyDeviceToHost)));
    printf("warpSize(device)=%d expected=%.1f\n", 64, expect);
    printf("  WarpSize=32 (upstream on gfx803): %.1f  ratio=%.3f\n", got32, got32 / expect);
    printf("  WarpSize=64 (patched)           : %.1f  ratio=%.3f\n", got64, got64 / expect);
    hipFree(x);
    hipFree(o);
    return 0;
}
