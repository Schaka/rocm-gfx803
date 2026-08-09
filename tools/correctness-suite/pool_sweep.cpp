// gfx803 pooling-forward correctness sweep (bucket C, arch-blind
// PoolingForward2d). Boundary risk: work-group tile sizing branches on
// out_width/out_height crossing 8/16/32/64/128 -- same shape as prior bugs
// (dispatch math keyed off size thresholds, never gfx803-specific,
// never gfx803-verified). Sweeps Max/Average/AverageInclusive across
// kernel sizes, strides, pad, and output sizes that straddle each
// grp_tile threshold.
#include "common.hpp"

static void cpu_pool(const std::vector<float>& x, std::vector<float>& y,
                      int C,int H,int W,int kh,int kw,int stride,int pad,
                      miopenPoolingMode_t mode, int& OH, int& OW) {
    OH = (H + 2*pad - kh)/stride + 1;
    OW = (W + 2*pad - kw)/stride + 1;
    y.assign((size_t)C*OH*OW, 0.f);
    for (int c=0;c<C;c++) {
        for (int oh=0; oh<OH; oh++) {
            for (int ow=0; ow<OW; ow++) {
                double acc = (mode == miopenPoolingMax) ? -std::numeric_limits<double>::infinity() : 0.0;
                int count = 0, count_incl = kh*kw;
                for (int r=0;r<kh;r++) {
                    int ih = oh*stride - pad + r;
                    for (int s=0;s<kw;s++) {
                        int iw = ow*stride - pad + s;
                        bool inb = (ih>=0 && ih<H && iw>=0 && iw<W);
                        double v = inb ? (double)x[(size_t)c*H*W+ih*W+iw] : 0.0;
                        if (mode == miopenPoolingMax) {
                            if (inb) acc = std::max(acc, v);
                        } else {
                            acc += v;
                            if (inb) count++;
                        }
                    }
                }
                if (mode == miopenPoolingMax) {
                    if (count_incl == 0) acc = 0;
                } else if (mode == miopenPoolingAverage) {
                    acc = count > 0 ? acc / count : 0.0;
                } else { // AverageInclusive
                    acc = acc / count_incl;
                }
                y[(size_t)c*OH*OW + oh*OW + ow] = (float)acc;
            }
        }
    }
}

static const char* mode_name(miopenPoolingMode_t m) {
    if (m == miopenPoolingMax) return "MAX";
    if (m == miopenPoolingAverage) return "AVG";
    return "AVG_INCL";
}

static bool run_one(miopenPoolingMode_t mode, int C,int H,int W,int kh,int kw,int stride,int pad) {
    int OH,OW;
    std::vector<float> h_x((size_t)C*H*W), h_ref;
    std::mt19937 rng(3);
    std::normal_distribution<float> dist(0.0f,2.0f);
    for (auto& v : h_x) v = dist(rng);
    cpu_pool(h_x, h_ref, C,H,W,kh,kw,stride,pad,mode,OH,OW);
    if (OH<=0||OW<=0) { printf("mode=%-9s C=%d H=%d W=%d k=%dx%d stride=%d pad=%d -> SKIP bad out size\n",mode_name(mode),C,H,W,kh,kw,stride,pad); return true; }

    size_t x_elems=(size_t)C*H*W, y_elems=(size_t)C*OH*OW;
    void *d_x,*d_y;
    CHECK_HIP(hipMalloc(&d_x, x_elems*4));
    CHECK_HIP(hipMalloc(&d_y, y_elems*4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), x_elems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, y_elems*4));

    miopenHandle_t handle; miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc,yDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);
    miopenSet4dTensorDescriptor(xDesc, miopenFloat, 1, C, H, W);
    miopenSet4dTensorDescriptor(yDesc, miopenFloat, 1, C, OH, OW);

    miopenPoolingDescriptor_t poolDesc;
    miopenCreatePoolingDescriptor(&poolDesc);
    miopenSet2dPoolingDescriptor(poolDesc, mode, kh, kw, pad, pad, stride, stride);

    size_t ws_size = 0;
    miopenPoolingGetWorkSpaceSizeV2(poolDesc, yDesc, &ws_size);
    void* d_ws = nullptr;
    if (ws_size > 0) CHECK_HIP(hipMalloc(&d_ws, ws_size));

    float a=1.0f,b=0.0f;
    miopenStatus_t es = miopenPoolingForward(handle, poolDesc, &a, xDesc, d_x, &b, yDesc, d_y, true, d_ws, ws_size);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    if (es != miopenStatusSuccess) {
        printf("mode=%-9s C=%d H=%d W=%d k=%dx%d stride=%d pad=%d -> EXEC FAILED status=%d\n",mode_name(mode),C,H,W,kh,kw,stride,pad,(int)es);
        ok = false;
    } else {
        std::vector<float> h_gpu(y_elems);
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, y_elems*4, hipMemcpyDeviceToHost));
        double cs = cos_sim(h_ref, h_gpu);
        printf("mode=%-9s C=%d H=%d W=%d k=%dx%d stride=%d pad=%d OH=%d OW=%d -> cos=%.5f%s\n",
               mode_name(mode),C,H,W,kh,kw,stride,pad,OH,OW,cs, cs<0.999 ? "  <-- WRONG" : "  OK");
        if (cs<0.999) ok=false;
    }

    if (d_ws) hipFree(d_ws);
    hipFree(d_x); hipFree(d_y);
    miopenDestroyPoolingDescriptor(poolDesc);
    miopenDestroyTensorDescriptor(xDesc); miopenDestroyTensorDescriptor(yDesc);
    miopenDestroy(handle);
    return ok;
}

int main() {
    miopenPoolingMode_t modes[] = {miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    // out_width/out_height straddling grp_tile thresholds: 8,16,32,64,128
    struct Case { int C,H,W,kh,kw,stride,pad; };
    Case cases[] = {
        {8, 16,16, 2,2, 2, 0},   // OH=OW=8
        {8, 18,18, 2,2, 2, 0},   // OH=OW=9 (just above 8)
        {8, 32,32, 2,2, 2, 0},   // OH=OW=16
        {8, 34,34, 2,2, 2, 0},   // OH=OW=17
        {8, 64,64, 2,2, 2, 0},   // OH=OW=32
        {8, 66,66, 2,2, 2, 0},   // OH=OW=33
        {8, 128,128, 2,2, 2, 0}, // OH=OW=64
        {8, 130,130, 2,2, 2, 0}, // OH=OW=65
        {8, 256,256, 2,2, 2, 0}, // OH=OW=128
        {24, 32,100, 3,3, 2, 1}, // real CLAP-ish asymmetric
        {24, 32,100, 3,3, 1, 1},
        {16, 7,7, 7,7, 1, 0},    // global-pool style, k==spatial
        {16, 5,5, 3,3, 2, 1},
        {16, 1,1, 1,1, 1, 0},
    };
    int total=0, wrong=0;
    for (auto mode : modes) for (auto& c : cases) {
        total++;
        if (!run_one(mode, c.C,c.H,c.W,c.kh,c.kw,c.stride,c.pad)) wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong>0 ? 1 : 0;
}
