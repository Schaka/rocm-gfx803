// ReduceExtreme (Min/Max/Argmin/Argmax) forward correctness sweep. Same
// source-file family (src/reduceextreme.cpp / src/kernels/MIOpenReduceExtreme.*)
// as the ReduceCalculation Prod-always-zero bug found via reduce_sweep.cpp --
// worth a pass on its own since a sibling identity/init bug is exactly the
// kind of thing likely to repeat in adjacent code. Reduces the middle axis
// of a 3D (outer, dim, inner) tensor, matching IsNotLastDim.
#include "common.hpp"

// 0 = OK, 1 = expected rejection (IsLargeReduceSize/etc "no solver found"),
// 2 = genuine wrong answer.
enum class ExtremeResult
{
    OK,
    SKIP,
    WRONG
};

static ExtremeResult run_one(miopenReduceExtremeOp_t op, int outer, int dim, int inner)
{
    bool want_value = (op == MIOPEN_REDUCE_EXTREME_MIN || op == MIOPEN_REDUCE_EXTREME_MAX);
    bool is_max      = (op == MIOPEN_REDUCE_EXTREME_MAX || op == MIOPEN_REDUCE_EXTREME_ARGMAX);

    size_t elems  = (size_t)outer * dim * inner;
    size_t yelems = (size_t)outer * inner;
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> h_x(elems);
    for(auto& v : h_x)
        v = dist(rng);

    std::vector<float> h_ref_val(yelems);
    std::vector<int32_t> h_ref_idx(yelems);
    for(int o = 0; o < outer; o++)
        for(int i = 0; i < inner; i++)
        {
            double best     = is_max ? -1e300 : 1e300;
            int32_t best_id = 0;
            for(int d = 0; d < dim; d++)
            {
                float v = h_x[((size_t)o * dim + d) * inner + i];
                if((is_max && v > best) || (!is_max && v < best))
                {
                    best    = v;
                    best_id = d;
                }
            }
            h_ref_val[(size_t)o * inner + i] = (float)best;
            h_ref_idx[(size_t)o * inner + i] = best_id;
        }

    void *d_x, *d_y = nullptr, *d_idx;
    CHECK_HIP(hipMalloc(&d_x, elems * 4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), elems * 4, hipMemcpyHostToDevice));
    if(want_value)
    {
        CHECK_HIP(hipMalloc(&d_y, yelems * 4));
        CHECK_HIP(hipMemset(d_y, 0, yelems * 4));
    }
    CHECK_HIP(hipMalloc(&d_idx, yelems * 4));
    CHECK_HIP(hipMemset(d_idx, 0, yelems * 4));

    miopenHandle_t handle;
    miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc, yDesc, idxDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);
    miopenCreateTensorDescriptor(&idxDesc);
    int xlens[3] = {outer, dim, inner};
    miopenSetTensorDescriptor(xDesc, miopenFloat, 3, xlens, nullptr);
    int ylens[2] = {outer, inner};
    miopenSetTensorDescriptor(yDesc, miopenFloat, 2, ylens, nullptr);
    miopenSetTensorDescriptor(idxDesc, miopenInt32, 2, ylens, nullptr);

    miopenStatus_t es =
        miopenReduceExtremeForward(handle, xDesc, d_x, 1, op, yDesc, d_y, idxDesc, d_idx);
    CHECK_HIP(hipDeviceSynchronize());

    const char* opn = op == MIOPEN_REDUCE_EXTREME_MIN
                          ? "MIN"
                          : op == MIOPEN_REDUCE_EXTREME_MAX
                                ? "MAX"
                                : op == MIOPEN_REDUCE_EXTREME_ARGMIN ? "ARGMIN" : "ARGMAX";
    ExtremeResult result = ExtremeResult::OK;
    if(es != miopenStatusSuccess)
    {
        // status 6 (miopenStatusNotImplemented, "No solver found") is the
        // expected shape here for MIN/MAX beyond IsLargeReduceSize()'s
        // limit -- ARGMIN/ARGMAX have no such limit, so if *those* ever
        // hit this branch it would be a real regression, not expected.
        bool expected = (es == miopenStatusNotImplemented) &&
                        (op == MIOPEN_REDUCE_EXTREME_MIN || op == MIOPEN_REDUCE_EXTREME_MAX);
        printf("op=%-6s outer=%d dim=%d inner=%d -> EXEC FAILED status=%d%s\n", opn, outer, dim,
               inner, (int)es, expected ? "  (expected: IsLargeReduceSize limit)" : "  <-- WRONG");
        result = expected ? ExtremeResult::SKIP : ExtremeResult::WRONG;
    }
    else
    {
        std::vector<int32_t> h_gpu_idx(yelems);
        CHECK_HIP(hipMemcpy(h_gpu_idx.data(), d_idx, yelems * 4, hipMemcpyDeviceToHost));
        size_t idx_mismatches = 0;
        for(size_t i = 0; i < yelems; i++)
            if(h_gpu_idx[i] != h_ref_idx[i])
                idx_mismatches++;

        double val_metric = 1.0;
        bool val_ok        = true;
        if(want_value)
        {
            std::vector<float> h_gpu_val(yelems);
            CHECK_HIP(hipMemcpy(h_gpu_val.data(), d_y, yelems * 4, hipMemcpyDeviceToHost));
            bool used_absdiff;
            val_ok = vectors_match(h_ref_val, h_gpu_val, &val_metric, &used_absdiff);
        }

        bool ok = (idx_mismatches == 0) && val_ok;
        printf("op=%-6s outer=%d dim=%d inner=%d -> idx_mismatches=%zu/%zu%s%s\n", opn, outer,
               dim, inner, idx_mismatches, yelems,
               want_value ? (val_ok ? "  val=OK" : "  val=WRONG") : "",
               ok ? "  OK" : "  <-- WRONG");
        result = ok ? ExtremeResult::OK : ExtremeResult::WRONG;
    }

    hipFree(d_x);
    if(d_y)
        hipFree(d_y);
    hipFree(d_idx);
    miopenDestroyTensorDescriptor(xDesc);
    miopenDestroyTensorDescriptor(yDesc);
    miopenDestroyTensorDescriptor(idxDesc);
    miopenDestroy(handle);
    return result;
}

int main()
{
    miopenReduceExtremeOp_t ops[] = {MIOPEN_REDUCE_EXTREME_MIN, MIOPEN_REDUCE_EXTREME_MAX,
                                     MIOPEN_REDUCE_EXTREME_ARGMIN, MIOPEN_REDUCE_EXTREME_ARGMAX};
    struct Case
    {
        int outer, dim, inner;
    };
    Case cases[] = {{4, 64, 10}, {4, 255, 10}, {4, 256, 10},
                    {4, 257, 10}, {1, 1000, 5}, {2, 24, 100},
                    {1, 1, 1},    {8, 7, 7}};
    int total = 0, wrong = 0, skipped = 0;
    for(auto op : ops)
        for(auto& c : cases)
        {
            total++;
            ExtremeResult r = run_one(op, c.outer, c.dim, c.inner);
            if(r == ExtremeResult::WRONG)
                wrong++;
            else if(r == ExtremeResult::SKIP)
                skipped++;
        }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG, %d SKIP (expected IsLargeReduceSize "
                     "rejections)\n",
            total, wrong, skipped);
    return wrong > 0 ? 1 : 0;
}
