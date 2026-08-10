// Cat (tensor concatenation) forward correctness sweep. Concatenates 2-3
// packed tensors along a given dim; IsApplicable requires all-packed
// tensors and caps input count (IsUnderXCountLimit).
#include "common.hpp"

static bool run_one(int dim, std::vector<std::vector<int>> shapes)
{
    int ndim = (int)shapes[0].size();
    int xCount = (int)shapes.size();

    std::vector<size_t> elems(xCount);
    std::vector<std::vector<float>> h_xs(xCount);
    std::mt19937 rng(37);
    std::normal_distribution<float> dist(0.0f, 1.5f);
    for(int i = 0; i < xCount; i++)
    {
        size_t e = 1;
        for(int v : shapes[i])
            e *= v;
        elems[i] = e;
        h_xs[i].resize(e);
        for(auto& v : h_xs[i])
            v = dist(rng);
    }

    std::vector<int> out_shape = shapes[0];
    for(int i = 1; i < xCount; i++)
        out_shape[dim] += shapes[i][dim];
    size_t y_elems = 1;
    for(int v : out_shape)
        y_elems *= v;

    // Host reference: NCHW-style strides, concatenate along `dim`.
    auto strides_for = [&](const std::vector<int>& shape) {
        std::vector<size_t> s(ndim);
        s[ndim - 1] = 1;
        for(int i = ndim - 2; i >= 0; i--)
            s[i] = s[i + 1] * shape[i + 1];
        return s;
    };
    std::vector<float> h_ref(y_elems);
    auto out_strides = strides_for(out_shape);
    int dim_offset    = 0;
    for(int i = 0; i < xCount; i++)
    {
        auto in_strides = strides_for(shapes[i]);
        size_t n        = elems[i];
        for(size_t flat = 0; flat < n; flat++)
        {
            size_t rem = flat;
            std::vector<int> idx(ndim);
            for(int d = 0; d < ndim; d++)
            {
                idx[d] = (int)(rem / in_strides[d]);
                rem %= in_strides[d];
            }
            size_t out_off = 0;
            for(int d = 0; d < ndim; d++)
            {
                int oi = idx[d] + (d == dim ? dim_offset : 0);
                out_off += (size_t)oi * out_strides[d];
            }
            h_ref[out_off] = h_xs[i][flat];
        }
        dim_offset += shapes[i][dim];
    }

    std::vector<void*> d_xs(xCount);
    std::vector<miopenTensorDescriptor_t> xDescs(xCount);
    for(int i = 0; i < xCount; i++)
    {
        CHECK_HIP(hipMalloc(&d_xs[i], elems[i] * 4));
        CHECK_HIP(hipMemcpy(d_xs[i], h_xs[i].data(), elems[i] * 4, hipMemcpyHostToDevice));
        miopenCreateTensorDescriptor(&xDescs[i]);
        std::vector<int> lens = shapes[i];
        miopenSetTensorDescriptor(xDescs[i], miopenFloat, ndim, lens.data(), nullptr);
    }
    void* d_y;
    CHECK_HIP(hipMalloc(&d_y, y_elems * 4));
    CHECK_HIP(hipMemset(d_y, 0, y_elems * 4));
    miopenTensorDescriptor_t yDesc;
    miopenCreateTensorDescriptor(&yDesc);
    miopenSetTensorDescriptor(yDesc, miopenFloat, ndim, out_shape.data(), nullptr);

    miopenHandle_t handle;
    miopenCreate(&handle);
    std::vector<const void*> xs_const(d_xs.begin(), d_xs.end());
    miopenStatus_t es = miopenCatForward(handle, xCount, xDescs.data(), xs_const.data(), yDesc,
                                        d_y, dim);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok = true;
    if(es != miopenStatusSuccess)
    {
        printf("dim=%d xCount=%d -> EXEC FAILED status=%d\n", dim, xCount, (int)es);
        ok = false;
    }
    else
    {
        std::vector<float> h_gpu(y_elems);
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, y_elems * 4, hipMemcpyDeviceToHost));
        double metric;
        bool used_absdiff;
        bool match = vectors_match(h_ref, h_gpu, &metric, &used_absdiff);
        printf("dim=%d xCount=%d -> %s=%.5f%s\n", dim, xCount, used_absdiff ? "maxabsdiff" : "cos",
               metric, match ? "  OK" : "  <-- WRONG");
        ok = match;
    }

    for(int i = 0; i < xCount; i++)
    {
        hipFree(d_xs[i]);
        miopenDestroyTensorDescriptor(xDescs[i]);
    }
    hipFree(d_y);
    miopenDestroyTensorDescriptor(yDesc);
    miopenDestroy(handle);
    return ok;
}

int main()
{
    int total = 0, wrong = 0;
    struct Case
    {
        int dim;
        std::vector<std::vector<int>> shapes;
    };
    // IsImprovementOverROCm requires output element count >= 1,000,000 --
    // below that MIOpen intentionally defers to rocPRIM/generic ROCm and
    // has no fallback solver reachable from this direct API call, so small
    // shapes correctly report "no solver found" (not a bug).
    Case cases[] = {
        {0, {{500, 200, 8}, {600, 200, 8}}},
        {1, {{4, 800, 400}, {4, 600, 400}, {4, 300, 400}}},
        {2, {{4, 800, 800}, {4, 800, 900}}},
        {0, {{4, 500, 500}, {4, 500, 500}}},
        {1, {{20, 2400, 32}, {20, 2400, 32}}},
    };
    for(auto& c : cases)
    {
        total++;
        if(!run_one(c.dim, c.shapes))
            wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong > 0 ? 1 : 0;
}
