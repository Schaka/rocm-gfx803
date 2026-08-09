// Shared helpers for the MIOpen correctness-suite sweeps. Each sweep is a
// standalone binary (own main()) that includes this header -- there's no
// shared library, just shared boilerplate, so any one sweep can still be
// copy-built in isolation if needed.
#pragma once

#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <limits>

#define CHECK_HIP(x)                                                                             \
    do                                                                                            \
    {                                                                                             \
        hipError_t e = (x);                                                                       \
        if(e != hipSuccess)                                                                       \
        {                                                                                         \
            fprintf(stderr, "HIP err %s at line %d\n", hipGetErrorString(e), __LINE__);            \
            exit(1);                                                                              \
        }                                                                                          \
    } while(0)

#define CHECK_MIO(x)                                                                              \
    do                                                                                            \
    {                                                                                             \
        miopenStatus_t s = (x);                                                                    \
        if(s != miopenStatusSuccess)                                                              \
        {                                                                                          \
            fprintf(stderr, "MIOpen err %d at line %d\n", (int)s, __LINE__);                       \
            exit(1);                                                                              \
        }                                                                                          \
    } while(0)

inline double cos_sim(const std::vector<float>& a, const std::vector<float>& b)
{
    double dot = 0, na = 0, nb = 0;
    for(size_t i = 0; i < a.size(); i++)
    {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    // IEEE754 double division is safe down to ~1e-308 -- an absolute epsilon
    // here (the previous `+ 1e-12`) silently corrupts the ratio whenever both
    // vectors have a legitimately tiny-but-nonzero magnitude (e.g. a Prod
    // reduction over many sub-1 values easily lands at 1e-20ish): 1e-12
    // dwarfs denom in that regime and crushes a correct near-1.0 cosine down
    // to ~0, misreported as WRONG. Only guard the actual 0/0 case.
    if(denom == 0.0)
        return 1.0; // both vectors exactly zero -- legitimate match
    return dot / denom;
}

// cos_sim degenerates when the reference vector is (near-)zero -- legitimate
// for clamping ops (RELU/CLIPPEDRELU/POWER) on small/negative input. Falls
// back to max-abs-diff in that regime. Returns true (match) / false (wrong)
// via the return value; *out_metric/*out_used_absdiff describe what ran.
inline bool vectors_match(const std::vector<float>& ref,
                          const std::vector<float>& gpu,
                          double* out_metric,
                          bool* out_used_absdiff,
                          double cos_threshold = 0.999,
                          double absdiff_threshold = 1e-4)
{
    double na = 0;
    for(float v : ref)
        na += (double)v * v;
    na = std::sqrt(na);
    if(na < 1e-6)
    {
        double maxdiff = 0;
        for(size_t i = 0; i < ref.size(); i++)
            maxdiff = std::max(maxdiff, (double)std::fabs(ref[i] - gpu[i]));
        *out_metric      = maxdiff;
        *out_used_absdiff = true;
        return maxdiff < absdiff_threshold;
    }
    *out_metric      = cos_sim(ref, gpu);
    *out_used_absdiff = false;
    return *out_metric >= cos_threshold;
}

// Host-side reference conv (NCHW, grouped, no dilation != 1 support needed
// so far -- add a dilation param if a future sweep needs it).
inline void cpu_conv(const std::vector<float>& x,
                     const std::vector<float>& w,
                     std::vector<float>& y,
                     int C, int H, int W, int K, int R, int S, int group, int stride, int pad)
{
    int OH = (H + 2 * pad - R) / stride + 1;
    int OW = (W + 2 * pad - S) / stride + 1;
    int Cg = C / group, Kg = K / group;
    y.assign((size_t)K * OH * OW, 0.f);
    for(int g = 0; g < group; g++)
        for(int kk = 0; kk < Kg; kk++)
        {
            int k = g * Kg + kk;
            for(int oh = 0; oh < OH; oh++)
                for(int ow = 0; ow < OW; ow++)
                {
                    double acc = 0;
                    for(int cc = 0; cc < Cg; cc++)
                    {
                        int c = g * Cg + cc;
                        for(int r = 0; r < R; r++)
                        {
                            int ih = oh * stride - pad + r;
                            if(ih < 0 || ih >= H)
                                continue;
                            for(int s = 0; s < S; s++)
                            {
                                int iw = ow * stride - pad + s;
                                if(iw < 0 || iw >= W)
                                    continue;
                                acc += (double)x[(size_t)c * H * W + ih * W + iw] *
                                       w[(size_t)k * Cg * R * S + cc * R * S + r * S + s];
                            }
                        }
                    }
                    y[(size_t)k * OH * OW + oh * OW + ow] = (float)acc;
                }
        }
}
