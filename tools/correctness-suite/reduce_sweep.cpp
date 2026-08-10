// gfx803 ReduceCalculation (Sum/Prod) forward correctness sweep (bucket C,
// arch-blind SumForward/ProdForward). IsApplicable requires reduction dim
// is NOT the last dim -- test reducing the middle axis of a 3D tensor
// (outer, dim, inner), sweeping dim size across typical thresholds.
#include "common.hpp"

static bool run_one(miopenReduceCalculationOp_t op, int outer, int dim, int inner) {
    size_t elems=(size_t)outer*dim*inner;
    size_t yelems=(size_t)outer*inner;
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f); // avoid near-zero for prod stability
    std::vector<float> h_x(elems), h_ref(yelems), h_gpu(yelems);
    for (auto& v : h_x) v = dist(rng);
    for (int o=0;o<outer;o++) for (int i=0;i<inner;i++) {
        double acc = (op == MIOPEN_REDUCE_CALCULATION_SUM) ? 0.0 : 1.0;
        for (int d=0; d<dim; d++) {
            float v = h_x[((size_t)o*dim+d)*inner+i];
            if (op == MIOPEN_REDUCE_CALCULATION_SUM) acc += v; else acc *= v;
        }
        h_ref[(size_t)o*inner+i] = (float)acc;
    }

    void *d_x,*d_y;
    CHECK_HIP(hipMalloc(&d_x, elems*4));
    CHECK_HIP(hipMalloc(&d_y, yelems*4));
    CHECK_HIP(hipMemcpy(d_x, h_x.data(), elems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, yelems*4));

    miopenHandle_t handle; miopenCreate(&handle);
    miopenTensorDescriptor_t xDesc,yDesc;
    miopenCreateTensorDescriptor(&xDesc);
    miopenCreateTensorDescriptor(&yDesc);
    int xlens[3] = {outer, dim, inner};
    miopenSetTensorDescriptor(xDesc, miopenFloat, 3, xlens, nullptr);
    int ylens[2] = {outer, inner};
    miopenSetTensorDescriptor(yDesc, miopenFloat, 2, ylens, nullptr);

    size_t ws_size = 0;
    miopenGetReduceCalculationWorkspaceSize(handle, xDesc, 1, op, yDesc, &ws_size);
    void* d_ws = nullptr;
    if (ws_size > 0) CHECK_HIP(hipMalloc(&d_ws, ws_size));

    miopenStatus_t es = miopenReduceCalculationForward(handle, MIOPEN_REDUCE_CALCULATION_NOT_PROPAGATE_NAN,
        d_ws, ws_size, xDesc, d_x, 1, op, yDesc, d_y);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok=true;
    const char* opn = (op==MIOPEN_REDUCE_CALCULATION_SUM) ? "SUM" : "PROD";
    if (es != miopenStatusSuccess) {
        printf("op=%-4s outer=%d dim=%d inner=%d -> EXEC FAILED status=%d\n",opn,outer,dim,inner,(int)es);
        ok=false;
    } else {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_y, yelems*4, hipMemcpyDeviceToHost));
        double cs = cos_sim(h_ref, h_gpu);
        printf("op=%-4s outer=%d dim=%d inner=%d -> cos=%.5f%s\n",opn,outer,dim,inner,cs, cs<0.999?"  <-- WRONG":"  OK");
        if (cs<0.999) ok=false;
    }

    if (d_ws) hipFree(d_ws);
    hipFree(d_x); hipFree(d_y);
    miopenDestroyTensorDescriptor(xDesc); miopenDestroyTensorDescriptor(yDesc);
    miopenDestroy(handle);
    return ok;
}

int main() {
    miopenReduceCalculationOp_t ops[] = {MIOPEN_REDUCE_CALCULATION_SUM, MIOPEN_REDUCE_CALCULATION_PROD};
    struct Case { int outer,dim,inner; };
    Case cases[] = {
        {4, 64, 10}, {4, 255, 10}, {4, 256, 10}, {4, 257, 10}, {1, 1000, 5}, {2, 24, 100}, {1,1,1}, {8, 7, 7}
    };
    int total=0, wrong=0;
    for (auto op : ops) for (auto& c : cases) {
        total++;
        if (!run_one(op, c.outer,c.dim,c.inner)) wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong>0?1:0;
}
