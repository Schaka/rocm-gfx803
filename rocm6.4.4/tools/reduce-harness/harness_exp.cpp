// Experiment variant of the committed reduce-harness for mechanism pinning.
// Adds: per-call device-address logging (d_x/d_y/d_ws, sizes, reuse flags) and
// a buffer-holding control mode (HAR_MODE=reuse|hold-dxdy|hold-all).
// All other logic is identical to gfx803/tools/reduce-harness/miopen_reduce_kernel_harness.cpp.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <chrono>

#define CHK(x)                                                                \
    do {                                                                      \
        hipError_t e = (x);                                                   \
        if(e != hipSuccess) {                                                 \
            fprintf(stderr, "HIP err %s (line %d)\n", hipGetErrorString(e),   \
                    __LINE__);                                                \
            exit(1);                                                          \
        }                                                                     \
    } while(0)

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
    int call = prefix[1] == '2' ? 2 : 1;
    char pn[128], mn[128];
    snprintf(pn, sizeof(pn), "gridwise_generic_reduce_%d_prepare", call);
    snprintf(mn, sizeof(mn), "gridwise_generic_reduce_%d", call);
    CHK(hipModuleGetFunction(&v.prep, v.mod, pn));
    CHK(hipModuleGetFunction(&v.mainf, v.mod, mn));
    g_vars[key] = v;
    return g_vars[key];
}

// -------- experiment control --------
static std::string g_mode = "reuse";   // reuse | hold-dxdy | hold-all | fresh
static int g_logaddr = 1;
static int g_preload = 0;              // preload all variants before running
static std::vector<std::pair<int64_t,int64_t>> g_shapes; // empty => default sweep
static std::vector<void*> g_leaks;     // for "fresh" mode: never-freed distinct buffers
static int g_ring=0;                     // ring size for "ring" mode
static std::vector<void*> g_ringbuf;     // bounded fresh-distinct src buffers
static int g_ringidx=0;
static std::vector<std::pair<int64_t,int64_t>> g_bench;  // HAR_BENCH shapes
static int g_bencheps=0;                 // HAR_BENCH_REPS
// "held" buffers that are never freed in hold modes
struct Held { void* dx=nullptr; void* dy=nullptr; void* ws=nullptr;
              size_t sdx=0, sdy=0, sws=0; };
static Held g_held;
// persistence of last-returned addresses for reuse detection
static void* g_last_dx=nullptr; static void* g_last_dy=nullptr; static void* g_last_ws=nullptr;

static void* alloc_or_reuse(void** held, size_t* heldsz, size_t sz, bool hold)
{
    if(hold && *held) { return *held; }        // keep same distinct buffer
    if(hold && !*held) {                        // first time: allocate & remember
        void* p=nullptr; CHK(hipMalloc(&p, sz)); *held=p; return p;
    }
    void* p=nullptr; CHK(hipMalloc(&p, sz)); return p;   // normal alloc/free
}

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

    bool rec_in  = (g_mode=="reuse" || g_mode=="reuse-in" || g_mode=="copyin" || g_mode=="copyin-rec" || g_mode=="ring");
    bool rec_out = (g_mode=="reuse" || g_mode=="reuse-out" || g_mode=="copyin" || g_mode=="copyin-rec" || g_mode=="ring");

    size_t sdx=h_x.size()*4, sdy=(size_t)n*4, sws=8192;
    void *d_x, *d_y, *d_ws;
    // input: recycled (malloc/free) or fresh-distinct (leaked)
    if(rec_in)  { CHK(hipMalloc(&d_x, sdx)); }
    else        { CHK(hipMalloc(&d_x, sdx)); g_leaks.push_back(d_x); }
    // output
    if(rec_out) { CHK(hipMalloc(&d_y, sdy)); }
    else        { CHK(hipMalloc(&d_y, sdy)); g_leaks.push_back(d_y); }
    // ws always recycled (malloc/free each call)
    CHK(hipMalloc(&d_ws, sws));
    // MIOpen-mitigation model: reduce from a fresh non-recycled internal input
    void* src = d_x;
    if(g_mode=="copyin") { CHK(hipMalloc(&src, sdx)); g_leaks.push_back(src); }
    else if(g_mode=="copyin-rec") { CHK(hipMalloc(&src, sdx)); }   // recycled internal src
    else if(g_mode=="ring") {
        // bounded ring of distinct src buffers (max-sized) -- realistic MIOpen pool
        while((int)g_ringbuf.size() < g_ring) { void* p=nullptr; CHK(hipMalloc(&p, 131072)); g_ringbuf.push_back(p); }
        src = g_ringbuf[g_ringidx++ % (g_ring?g_ring:1)];
    }

    bool rx = (d_x==g_last_dx), ry=(d_y==g_last_dy), rw=(d_ws==g_last_ws);
    g_last_dx=d_x; g_last_dy=d_y; g_last_ws=d_ws;

    int bad = 0;
    // keep signature: run [m,n] reduce
    // (fill buffers + launch identical to committed harness)
    CHK(hipMemcpy(d_x, h_x.data(), h_x.size() * 4, hipMemcpyHostToDevice));
    if(g_mode=="copyin" || g_mode=="copyin-rec" || g_mode=="ring") CHK(hipMemcpy(src, d_x, h_x.size() * 4, hipMemcpyDeviceToDevice));
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
        a[0] = &orig; a[1] = &blkg; a[2] = &alpha; a[3] = &src;
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
            a[0] = &orig; a[1] = &blkg; a[2] = &alpha; a[3] = &src;
            a[4] = &beta; a[5] = &d_y; a[6] = &d_ws; a[7] = &ws_off; a[8] = &ip;
            CHK(hipModuleLaunchKernel(v2.mainf, g2, 1, 1, 256, 1, 1, 0, 0, a, nullptr));
        }
    }
    CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(h_gpu.data(), d_y, (size_t)n * 4, hipMemcpyDeviceToHost));

    for(size_t j = 0; j < (size_t)n; ++j)
    {
        float diff = std::fabs(h_gpu[j] - h_ref[j]);
        float tol  = 1e-3f + 1e-3f * std::fabs(h_ref[j]);
        if(diff > tol) bad++;
    }

    if(g_logaddr)
        fprintf(stderr, "%s m=%lld n=%lld %s bad=%d dx=%p(rx=%d) dy=%p(ry=%d) ws=%p(rw=%d)\n",
                tag, (long long)m, (long long)n, bad?"FAIL":"ok", bad, d_x, rx, d_y, ry, d_ws, rw);

    if(rec_in) CHK(hipFree(d_x));
    if(g_mode=="copyin-rec") CHK(hipFree(src));
    if(rec_out) CHK(hipFree(d_y));
    CHK(hipFree(d_ws));
    return bad;
}


// ---------- bench: device-side per-call timing (no host rng) ----------
static void launchReduce(size_t inv, size_t tor, const void* src, void* d_y, void* d_ws)
{
    M me = method(inv, tor);
    int g  = gridsize(me, inv, tor);
    int blkg = (me == M::MB) ? (int)(g / inv) : 0;
    Pad p = padding(me, inv, tor, g);
    int inLen[6] = {(int)inv, (int)tor, 0, 0, 0, 0};
    int inStr[6] = {1, (int)inv, 0, 0, 0, 0};
    int outStr[6] = {1, 0, 0, 0, 0, 0};
    float alpha = 1.f, beta = 0.f;
    long ws_off = 0;
    int orig = (int)tor;
    void* ip = nullptr;
    Variant& v = loadV("v", me, p);
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
        a[0] = &orig; a[1] = &blkg; a[2] = &alpha; a[3] = (void**)&src;
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
            a[0] = &orig; a[1] = &blkg; a[2] = &alpha; a[3] = (void**)&src;
            a[4] = &beta; a[5] = &d_y; a[6] = &d_ws; a[7] = &ws_off; a[8] = &ip;
            CHK(hipModuleLaunchKernel(v2.mainf, g2, 1, 1, 256, 1, 1, 0, 0, a, nullptr));
        }
    }
}

static double benchDev(const std::vector<std::pair<int64_t,int64_t>>& shapes,
                       int64_t reps, int use_copy, int ringK)
{
    std::vector<std::vector<float>> hx;
    std::vector<size_t> sdx, sdy;
    for(const auto& [m, n] : shapes)
    {
        std::vector<float> h((size_t)m * n, 1.0f);
        sdx.push_back(h.size() * 4); sdy.push_back((size_t)n * 4);
        hx.push_back(std::move(h));
    }
    std::vector<void*> ring;
    int idx = 0;
    auto t0 = std::chrono::steady_clock::now();
    for(int64_t r = 0; r < reps; ++r)
        for(size_t i = 0; i < shapes.size(); ++i)
        {
            const auto& [m, n] = shapes[i];
            void *d_x, *d_y, *d_ws;
            CHK(hipMalloc(&d_x, sdx[i])); CHK(hipMalloc(&d_y, sdy[i])); CHK(hipMalloc(&d_ws, 8192));
            CHK(hipMemcpy(d_x, hx[i].data(), sdx[i], hipMemcpyHostToDevice));
            void* src = d_x;
            if(use_copy)
            {
                while((int)ring.size() < ringK)
                { void* p = nullptr; CHK(hipMalloc(&p, 131072)); ring.push_back(p); }
                src = ring[idx++ % ringK];
                CHK(hipMemcpy(src, d_x, sdx[i], hipMemcpyDeviceToDevice));
            }
            CHK(hipMemset(d_y, 0xCD, sdy[i])); CHK(hipMemset(d_ws, 0, 8192));
            launchReduce((size_t)n, (size_t)m, src, d_y, d_ws);
            CHK(hipDeviceSynchronize());
            CHK(hipFree(d_x)); CHK(hipFree(d_y)); CHK(hipFree(d_ws));
        }
    auto t1 = std::chrono::steady_clock::now();
    long long calls = reps * (long long)shapes.size();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)calls;
}

int main(int argc, char** argv)
{
    const char* mo = getenv("HAR_MODE");
    if(mo) g_mode = mo;
    const char* la = getenv("HAR_LOGADDR");
    if(la) g_logaddr = atoi(la);
    const char* pl = getenv("HAR_PRELOAD");
    if(pl) g_preload = atoi(pl);
    const char* rk = getenv("HAR_RING");
    if(rk) g_ring = atoi(rk);
    const char* bp = getenv("HAR_BENCH");
    if(bp){ // parse m,n,m2,n2,... plus reps from HAR_BENCH_REPS
        int v[64]={0}, nv=0;
        char* p=strdup(bp); char* tok=strtok(p,",");
        while(tok && nv<64){ v[nv++]=atoi(tok); tok=strtok(nullptr,",");}
        free(p);
        if(nv>=2 && v[0]>0){ for(int i=0;i+1<nv;i+=2) g_bench.emplace_back(v[i],v[i+1]); }
    }
    const char* br = getenv("HAR_BENCH_REPS");
    if(br) g_bencheps = atoi(br);
    if(g_bencheps<=0) g_bencheps=20000;

    for(int i = 1; i + 1 < argc; i += 2)
    {
        long long m = atoll(argv[i]), n = atoll(argv[i + 1]);
        if(m) g_shapes.emplace_back(m, n);
    }

    if(g_preload)
    {
        // (key, methodCode, srcpad, dstpad)
        static const char* K[][4] = {
            {"v_tw_00","0","0","0"}, {"v_tw_10","0","1","0"}, {"v_tw_11","0","1","1"},
            {"v_wp_10","1","1","0"}, {"v_wp_11","1","1","1"}, {"v_bw_00","2","0","0"},
            {"v_bw_10","2","1","0"}, {"v_mb_10","3","1","0"}, {"v2_tw_11","0","1","1"},
            {"v2_wp_10","1","1","0"}, {"v2_bw_00","2","0","0"}
        };
        for(auto& e : K)
        {
            const char* k = e[0];
            M me = (M)atoi(e[1]);
            Pad pp = {atoi(e[2]), atoi(e[3])};
            loadV(k[1]=='2'?"v2":"v", me, pp);
        }
        fprintf(stderr, "preloaded all variants\n");
    }

    std::vector<std::pair<int64_t, int64_t>> shapes = g_shapes;
    if(shapes.empty())
    for(int64_t m = 1; m < 2049; m *= 8)
        for(int64_t n = 2; n < 2049; n *= 8)
        {
            if(m * n > 32768) continue;
            for(int64_t mm : {m, m + 1, m + 3, m + 5})
                shapes.emplace_back(mm, n);
        }
    std::mt19937 rng(0);
    int failed = 0, k = 0;
    for(const auto& [m, n] : shapes)
    {
        ++k;
        char tag[32]; snprintf(tag, sizeof(tag), "[%d]", k);
        int bad = reduceOne(tag, m, n, rng);
        if(bad) ++failed;
    }
    fprintf(stderr, "HAR_MODE=%s Total: %zu, failed: %d\n", g_mode.c_str(), shapes.size(), failed);

    // ---- benchmark hot loop (device-side only) ----
    if(!g_bench.empty())
    {
        for(int pass = 1; pass <= 2; ++pass)
        {
            int use_copy = (pass == 2);
            const char* nm = use_copy ? "mit-ring" : "baseline";
            int ringK = use_copy ? g_ring : 0;
            double us = benchDev(g_bench, g_bencheps, use_copy, ringK);
            fprintf(stderr, "BENCH %s ring=%d us/call=%.3f (shapes=%zu mode=%s)\n",
                    nm, ringK, us, g_bench.size(), g_mode.c_str());
        }
    }
    return failed ? 1 : 0;
}
