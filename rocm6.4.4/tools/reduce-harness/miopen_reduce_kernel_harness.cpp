// Faithful kernel-level repro harness for MIOpen's dynamic-CK miopenReduceTensor
// on gfx803. Drives the ACTUAL compiled CK reduction kernel code objects via
// hipModule, with NO MIOpen C++ dispatch, and malloc/free data-buffer address
// reuse each call (the confirmed trigger). See README.md and KERNEL_BUGS.md
// ("The ReduceSum kernel-cache mystery") for the full bug write-up.
//
// This is a portable repro: re-build build_variants.sh's .hsaco code objects and
// re-run against any ROCm stack (incl. the ROCm 7 port) to re-validate.
//
// Sweep: reduces axis 0 of [m,n] float arrays; shapes come from the program arg
// list (default: MIOpen's original 52-shape sweep). Each shape prints
//   p[k] m=.. n=.. OK|FAIL bad=..
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <random>

#define CHK(x)                                                                \
    do {                                                                      \
        hipError_t e = (x);                                                   \
        if(e != hipSuccess) {                                                 \
            fprintf(stderr, "HIP err %s (line %d)\n", hipGetErrorString(e),   \
                    __LINE__);                                                \
            exit(1);                                                          \
        }                                                                     \
    } while(0)

// ================= MIOpen ReductionKernelConfigurator replica ===============
// src/reducetensor.cpp; default_tunable_generic_reduction={256,8,2,2}; warp=64.
static const int BlockSize = 256, WarpSize = 64, NumWarps = 4;
static const int GTB = 8, GAIW = 2, GAIB = 2;

enum class M { TW, WP, BW, MB };

static const char* mname(M m)
{
    switch(m)
    {
    case M::TW: return "tw";
    case M::WP: return "wp";
    case M::BW: return "bw";
    default: return "mb";
    }
}

static M method(size_t inv, size_t tor)
{
    if(inv == 1) return tor <= 1024 ? M::BW : M::MB;
    if(tor <= 64) return M::TW;
    if(tor <= 256) return M::WP;
    if(tor <= 1024) return M::BW;
    return M::MB;
}

static int gridsize(M m, size_t inv, size_t tor)
{
    switch(m)
    {
    case M::TW: return (int)((inv + BlockSize - 1) / BlockSize);
    case M::WP: return (int)((inv + NumWarps - 1) / NumWarps);
    case M::BW: return (int)inv;
    default: {
        size_t epr = (tor + 1024 - 1) / 1024;
        return (int)(epr > 32 ? inv * 32 : inv * epr);
    }
    }
}

struct Pad { int src, dst; };

static Pad padding(M m, size_t inv, size_t tor, int g)
{
    switch(m)
    {
    case M::TW: {
        int c = GTB;
        return {(int)(inv < (size_t)g * BlockSize || tor % c > 0),
                (int)(inv < (size_t)g * BlockSize)};
    }
    case M::WP: {
        int c = WarpSize * GAIW;
        return {(int)(inv < (size_t)g * BlockSize / WarpSize || tor % c > 0),
                (int)(inv < (size_t)g * BlockSize / WarpSize)};
    }
    case M::BW: {
        int c = BlockSize * GAIB;
        return {(int)(tor % c > 0), 0};
    }
    default: {
        int blkg = g / (int)inv, c = BlockSize * GAIB;
        int rsb  = (((tor + blkg - 1) / blkg + c - 1) / c) * c;
        return {(int)(tor < (size_t)rsb * blkg), 0};
    }
    }
}

// ================= variant code-object loader (lazy, on first use) ==========
struct Variant { hipModule_t mod; hipFunction_t prep, mainf; };
static std::map<std::string, Variant> g_vars;

static Variant& loadV(const char* prefix, M m, Pad p)
{
    char key[80];
    snprintf(key, sizeof(key), "%s_%s_%d%d", prefix, mname(m), p.src, p.dst);
    auto it = g_vars.find(key);
    if(it != g_vars.end()) return it->second;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.hsaco", getenv("REDUCE_HSACO_DIR") ?: "/scratch/ckbuild",
             key);
    Variant v;
    if(hipModuleLoad(&v.mod, path) != hipSuccess)
    { fprintf(stderr, "FATAL: cannot load %s\n", path); exit(2); }
    int call = prefix[1] == '2' ? 2 : 1;   // prefix "v" (first) or "v2" (second)
    char pn[128], mn[128];
    snprintf(pn, sizeof(pn), "gridwise_generic_reduce_%d_prepare", call);
    snprintf(mn, sizeof(mn), "gridwise_generic_reduce_%d", call);
    CHK(hipModuleGetFunction(&v.prep, v.mod, pn));
    CHK(hipModuleGetFunction(&v.mainf, v.mod, mn));
    g_vars[key] = v;
    return g_vars[key];
}

// ================= one reduce call =================
static int reduceOne(const char* tag, int64_t m, int64_t n, std::mt19937& rng)
{
    std::vector<float> h_x((size_t)m * n), h_ref((size_t)n, 0.f), h_gpu((size_t)n);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for(int64_t i = 0; i < m; ++i)
        for(int64_t j = 0; j < n; ++j)
        { float v = dist(rng) / (float)m; h_x[i * n + j] = v; h_ref[j] += v; }

    size_t inv = n, tor = m;
    M me   = method(inv, tor);
    int g  = gridsize(me, inv, tor);
    int blkg = (me == M::MB) ? (int)(g / inv) : 0;
    Pad p = padding(me, inv, tor, g);

    void *d_x, *d_y, *d_ws;
    CHK(hipMalloc(&d_x, h_x.size() * 4));
    CHK(hipMalloc(&d_y, (size_t)n * 4));
    CHK(hipMalloc(&d_ws, 8192));
    CHK(hipMemcpy(d_x, h_x.data(), h_x.size() * 4, hipMemcpyHostToDevice));
    CHK(hipMemset(d_y, 0xCD, (size_t)n * 4));
    CHK(hipMemset(d_ws, 0, 8192));

    int inLen[6] = {(int)n, (int)m, 0, 0, 0, 0};
    int inStr[6] = {1, (int)n, 0, 0, 0, 0};
    int outStr[6] = {1, 0, 0, 0, 0, 0};
    float alpha = 1.f, beta = 0.f;
    long ws_off = 0;
    int orig = (int)m;
    void* ip = nullptr;

    Variant& v = loadV("v", me, p);
    // prepare + main (first call)
    {
        void* a[21];
        a[0] = &g; a[1] = &blkg;
        for(int i = 0; i < 6; ++i)
        { a[2 + i] = &inLen[i]; a[8 + i] = &inStr[i]; a[14 + i] = &outStr[i]; }
        a[20] = &d_ws;
        CHK(hipModuleLaunchKernel(v.prep, g, 1, 1, 256, 1, 1, 0, 0, a, nullptr));
    }
    {
        void* a[9];
        a[0] = &orig; a[1] = &blkg; a[2] = &alpha; a[3] = &d_x;
        a[4] = &beta; a[5] = &d_y; a[6] = &d_ws; a[7] = &ws_off; a[8] = &ip;
        CHK(hipModuleLaunchKernel(v.mainf, g, 1, 1, 256, 1, 1, 0, 0, a, nullptr));
    }
    if(me == M::MB)
    {
        size_t tor2 = blkg;
        M me2 = tor2 <= WarpSize / 4 ? M::TW : (tor2 <= BlockSize ? M::WP : M::BW);
        int g2 = tor2 <= WarpSize / 4 ? (int)((inv + BlockSize - 1) / BlockSize)
               : tor2 <= BlockSize ? (int)((inv + NumWarps - 1) / NumWarps) : (int)inv;
        Pad p2 = padding(me2, inv, tor2, g2);
        Variant& v2 = loadV("v2", me2, p2);
        // second-call prepare + main
        {
            void* a[21];
            a[0] = &g2; a[1] = &blkg;
            for(int i = 0; i < 6; ++i)
            { a[2 + i] = &inLen[i]; a[8 + i] = &inStr[i]; a[14 + i] = &outStr[i]; }
            a[20] = &d_ws;
            CHK(hipModuleLaunchKernel(v2.prep, g2, 1, 1, 256, 1, 1, 0, 0, a, nullptr));
        }
        {
            void* a[9];
            a[0] = &orig; a[1] = &blkg; a[2] = &alpha; a[3] = &d_x;
            a[4] = &beta; a[5] = &d_y; a[6] = &d_ws; a[7] = &ws_off; a[8] = &ip;
            CHK(hipModuleLaunchKernel(v2.mainf, g2, 1, 1, 256, 1, 1, 0, 0, a, nullptr));
        }
    }
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(h_gpu.data(), d_y, (size_t)n * 4, hipMemcpyDeviceToHost));

    int bad = 0;
    for(size_t j = 0; j < (size_t)n; ++j)
    {
        float diff = std::fabs(h_gpu[j] - h_ref[j]);
        float tol  = 1e-3f + 1e-3f * std::fabs(h_ref[j]);
        if(diff > tol) bad++;
    }
    hipFree(d_x); hipFree(d_y); hipFree(d_ws);
    return bad;
}

int main(int argc, char** argv)
{
    // default sweep: MIOpen's original 52-shape set (reduce axis 0)
    std::vector<std::pair<int64_t, int64_t>> shapes;
    if(argc > 1 && std::string(argv[1]) == "--shapes")
    {
        for(int i = 2; i + 1 < argc; i += 2)
            shapes.emplace_back(atoll(argv[i]), atoll(argv[i + 1]));
    }
    else
    {
        for(int64_t m = 1; m < 2049; m *= 8)
            for(int64_t n = 2; n < 2049; n *= 8)
            {
                if(m * n > 32768) continue;
                for(int64_t mm : {m, m + 1, m + 3, m + 5})
                    shapes.emplace_back(mm, n);
            }
    }
    std::mt19937 rng(0);
    int failed = 0, k = 0;
    for(const auto& [m, n] : shapes)
    {
        ++k;
        int bad = reduceOne("pb", m, n, rng);
        if(bad) { ++failed; }
        fprintf(stderr, "[%d] m=%lld n=%lld %s bad=%d\n", k, (long long)m, (long long)n,
                bad ? "FAIL" : "ok", bad);
    }
    fprintf(stderr, "Total: %zu, failed: %d\n", shapes.size(), failed);
    return failed ? 1 : 0;
}
