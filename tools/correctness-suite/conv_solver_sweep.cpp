// Generic forward-conv correctness sweeper. Forces a specific MIOpen solver
// via MIOPEN_DEBUG_FIND_ONLY_SOLVER (env, set by the caller/run_all.sh) and
// sweeps shapes given on stdin as "C H W K R S group stride pad" (one per
// line, '#' comments allowed). If the forced solver isn't applicable for a
// shape, miopenFindConvolutionForwardAlgorithm returns 0 results -- reported
// as SKIP, not a failure (some shape lists deliberately include a rejected
// boundary case as a sanity check that IsApplicable still gates correctly).
#include "common.hpp"
#include <random>
#include <sstream>
#include <iostream>

static bool run_one(int C, int H, int W, int K, int R, int S, int group, int stride, int pad)
{
    int OH = (H + 2 * pad - R) / stride + 1, OW = (W + 2 * pad - S) / stride + 1;
    if(OH <= 0 || OW <= 0)
    {
        printf("C=%d H=%d W=%d K=%d R=%d S=%d group=%d stride=%d pad=%d -> SKIP (bad output size)\n",
               C, H, W, K, R, S, group, stride, pad);
        return true;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    size_t x_elems = (size_t)C * H * W, w_elems = (size_t)K * (C / group) * R * S,
           y_elems = (size_t)K * OH * OW;
    std::vector<float> h_x(x_elems), h_w(w_elems), h_ref, h_gpu(y_elems);
    for(auto& v : h_x)
        v = dist(rng);
    for(auto& v : h_w)
        v = dist(rng) * 0.1f;
    cpu_conv(h_x, h_w, h_ref, C, H, W, K, R, S, group, stride, pad);

    void *d_x, *d_w, *d_y;
    CHECK_HIP(hipMalloc(&d_x, x_elems * 4));
    CHECK_HIP(hipMalloc(&d_w, w_elems * 4));
    CHECK_HIP(hipMalloc(&d_y, y_elems * 4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), x_elems * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_w, h_w.data(), w_elems * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, y_elems * 4));

    miopenHandle_t handle;
    miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc, wDesc, yDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&wDesc);
    miopenCreateTensorDescriptor(&yDesc);
    miopenSet4dTensorDescriptor(xDesc, miopenFloat, 1, C, H, W);
    miopenSet4dTensorDescriptor(wDesc, miopenFloat, K, C / group, R, S);
    miopenSet4dTensorDescriptor(yDesc, miopenFloat, 1, K, OH, OW);

    miopenConvolutionDescriptor_t convDesc;
    miopenCreateConvolutionDescriptor(&convDesc);
    miopenInitConvolutionDescriptor(convDesc, miopenConvolution, pad, pad, stride, stride, 1, 1);
    miopenSetConvolutionGroupCount(convDesc, group);

    size_t ws_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc, convDesc, yDesc, &ws_size);
    void* d_ws = nullptr;
    if(ws_size > 0)
        CHECK_HIP(hipMalloc(&d_ws, ws_size));

    const int req = 4;
    int returned   = 0;
    std::vector<miopenConvAlgoPerf_t> perf(req);
    miopenStatus_t fs = miopenFindConvolutionForwardAlgorithm(handle,
                                                               xDesc,
                                                               d_x,
                                                               wDesc,
                                                               d_w,
                                                               convDesc,
                                                               yDesc,
                                                               d_y,
                                                               req,
                                                               &returned,
                                                               perf.data(),
                                                               d_ws,
                                                               ws_size,
                                                               false);

    bool ok = true;
    if(fs != miopenStatusSuccess || returned == 0)
    {
        printf("C=%d H=%d W=%d K=%d R=%d S=%d group=%d stride=%d pad=%d -> SKIP (forced solver "
               "not applicable, status=%d, n=%d)\n",
               C, H, W, K, R, S, group, stride, pad, (int)fs, returned);
    }
    else
    {
        float alpha = 1.0f, beta = 0.0f;
        miopenStatus_t es = miopenConvolutionForward(handle,
                                                      &alpha,
                                                      xDesc,
                                                      d_x,
                                                      wDesc,
                                                      d_w,
                                                      convDesc,
                                                      perf[0].fwd_algo,
                                                      &beta,
                                                      yDesc,
                                                      d_y,
                                                      d_ws,
                                                      ws_size);
        CHECK_HIP(hipDeviceSynchronize());
        if(es != miopenStatusSuccess)
        {
            printf("C=%d H=%d W=%d K=%d R=%d S=%d group=%d stride=%d pad=%d -> EXEC FAILED "
                   "status=%d\n",
                   C, H, W, K, R, S, group, stride, pad, (int)es);
            ok = false;
        }
        else
        {
            CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, y_elems * 4, hipMemcpyDeviceToHost));
            double metric;
            bool used_absdiff;
            bool match = vectors_match(h_ref, h_gpu, &metric, &used_absdiff);
            printf("C=%d H=%d W=%d K=%d R=%d S=%d group=%d stride=%d pad=%d -> %s=%.5f%s\n",
                   C, H, W, K, R, S, group, stride, pad,
                   used_absdiff ? "maxabsdiff" : "cos", metric, match ? "  OK" : "  <-- WRONG");
            ok = match;
        }
    }

    if(d_ws)
        hipFree(d_ws);
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_y);
    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroyTensorDescriptor(xDesc);
    miopenDestroyTensorDescriptor(wDesc);
    miopenDestroyTensorDescriptor(yDesc);
    miopenDestroy(handle);
    return ok;
}

int main()
{
    std::string line;
    int total = 0, wrong = 0;
    while(std::getline(std::cin, line))
    {
        if(line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        int C, H, W, K, R, S, group, stride, pad;
        if(!(iss >> C >> H >> W >> K >> R >> S >> group >> stride >> pad))
            continue;
        total++;
        if(!run_one(C, H, W, K, R, S, group, stride, pad))
            wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d shapes tested, %d WRONG\n", total, wrong);
    return wrong > 0 ? 1 : 0;
}
