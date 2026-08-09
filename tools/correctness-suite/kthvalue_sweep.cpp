// Kthvalue forward correctness sweep -- returns the k-th smallest value (and
// its index) along an axis. IsImprovementOverROCm requires the reduce dim
// to be the LAST axis (stride==1) with size>=300 -- below that MIOpen has
// no solver reachable from this direct API call (same "no fallback outside
// ORT's dispatch layer" situation as cat_sweep's size floor). Host
// reference via std::nth_element (1-indexed k, matching PyTorch's
// torch.kthvalue convention that MIOpen's kthvalue mirrors).
#include "common.hpp"
#include <algorithm>

static bool run_one(int outer, int inner, int reduce, size_t k)
{
    // Layout (outer, inner, reduce) -- reduce is the last, contiguous axis.
    size_t elems  = (size_t)outer * inner * reduce;
    size_t yelems = (size_t)outer * inner;
    std::mt19937 rng(53);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> h_x(elems);
    for(auto& v : h_x)
        v = dist(rng);

    std::vector<float> h_ref_val(yelems);
    for(int o = 0; o < outer; o++)
        for(int i = 0; i < inner; i++)
        {
            size_t base = ((size_t)o * inner + i) * reduce;
            std::vector<float> col(h_x.begin() + base, h_x.begin() + base + reduce);
            std::nth_element(col.begin(), col.begin() + (k - 1), col.end());
            h_ref_val[(size_t)o * inner + i] = col[k - 1];
        }

    void *d_x, *d_y;
    int64_t* d_idx;
    CHECK_HIP(hipMalloc(&d_x, elems * 4));
    CHECK_HIP(hipMalloc(&d_y, yelems * 4));
    CHECK_HIP(hipMalloc((void**)&d_idx, yelems * sizeof(int64_t)));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), elems * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, yelems * 4));
    CHECK_HIP(hipMemset(d_idx, 0, yelems * sizeof(int64_t)));

    miopenHandle_t handle;
    miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc, yDesc, idxDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);
    miopenCreateTensorDescriptor(&idxDesc);
    int xlens[3] = {outer, inner, reduce};
    miopenSetTensorDescriptor(xDesc, miopenFloat, 3, xlens, nullptr);
    int ylens[2] = {outer, inner};
    miopenSetTensorDescriptor(yDesc, miopenFloat, 2, ylens, nullptr);
    miopenSetTensorDescriptor(idxDesc, miopenInt64, 2, ylens, nullptr);

    miopenStatus_t es = miopenKthvalueForward(handle, xDesc, d_x, yDesc, d_y, idxDesc,
                                              (size_t*)d_idx, k, 2, false);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    if(es != miopenStatusSuccess)
    {
        printf("outer=%d inner=%d reduce=%d k=%zu -> EXEC FAILED status=%d\n", outer, inner,
               reduce, k, (int)es);
        ok = false;
    }
    else
    {
        std::vector<float> h_gpu_val(yelems);
        CHECK_HIP(hipMemcpy(h_gpu_val.data(), d_y, yelems * 4, hipMemcpyDeviceToHost));
        double metric;
        bool used_absdiff;
        bool match = vectors_match(h_ref_val, h_gpu_val, &metric, &used_absdiff);
        printf("outer=%d inner=%d reduce=%d k=%zu -> %s=%.5f%s\n", outer, inner, reduce, k,
               used_absdiff ? "maxabsdiff" : "cos", metric, match ? "  OK" : "  <-- WRONG");
        ok = match;
    }

    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_idx);
    miopenDestroyTensorDescriptor(xDesc);
    miopenDestroyTensorDescriptor(yDesc);
    miopenDestroyTensorDescriptor(idxDesc);
    miopenDestroy(handle);
    return ok;
}

int main()
{
    struct Case
    {
        int outer, inner, reduce;
        size_t k;
    };
    Case cases[] = {
        {4, 10, 300, 1}, {4, 10, 300, 150}, {4, 10, 300, 300}, {4, 10, 500, 250},
        {1, 5, 1000, 500}, {2, 100, 300, 1}, {8, 7, 300, 150},
    };
    int total = 0, wrong = 0;
    for(auto& c : cases)
    {
        total++;
        if(!run_one(c.outer, c.inner, c.reduce, c.k))
            wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong > 0 ? 1 : 0;
}
