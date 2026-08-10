// gfx803 tensorOp correctness sweep (bucket C, arch-blind Op4dTensorLite/
// Generic/OpTensorFwdBias -- dispatch between Lite/Generic/Squash/FwdBias
// solvers keyed on shape/broadcast pattern, never gfx803-verified). Covers
// full-tensor elementwise add/mul and broadcast bias-add (shape (1,C,1,1)
// against (N,C,H,W)), several sizes.
#include "common.hpp"

static bool run_one(miopenTensorOp_t op, int N,int C,int H,int W, bool broadcast_bias) {
    size_t elems=(size_t)N*C*H*W;
    size_t belems = broadcast_bias ? (size_t)C : elems;
    std::mt19937 rng(21);
    std::normal_distribution<float> dist(0.0f,1.5f);
    std::vector<float> h_a(elems), h_b(belems), h_ref(elems), h_gpu(elems);
    for (auto& v : h_a) v = dist(rng);
    for (auto& v : h_b) v = dist(rng);
    for (size_t n=0;n<(size_t)N;n++) for (size_t c=0;c<(size_t)C;c++) for (size_t hw=0; hw<(size_t)H*W; hw++) {
        size_t idx = (n*C+c)*H*W+hw;
        float bv = broadcast_bias ? h_b[c] : h_b[idx];
        float r;
        switch(op) {
            case miopenTensorOpAdd: r = h_a[idx]+bv; break;
            case miopenTensorOpMul: r = h_a[idx]*bv; break;
            case miopenTensorOpMin: r = std::min(h_a[idx],bv); break;
            case miopenTensorOpMax: r = std::max(h_a[idx],bv); break;
            default: r = h_a[idx];
        }
        h_ref[idx] = r;
    }

    void *d_a,*d_b,*d_c;
    CHECK_HIP(hipMalloc(&d_a, elems*4));
    CHECK_HIP(hipMalloc(&d_b, belems*4));
    CHECK_HIP(hipMalloc(&d_c, elems*4));
    CHECK_HIP(hipMemcpy(d_a, h_a.data(), elems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b, h_b.data(), belems*4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_c, 0, elems*4));

    miopenHandle_t handle; miopenCreate(&handle);
    miopenTensorDescriptor_t aDesc,bDesc,cDesc;
    miopenCreateTensorDescriptor(&aDesc);
    miopenCreateTensorDescriptor(&bDesc);
    miopenCreateTensorDescriptor(&cDesc);
    miopenSet4dTensorDescriptor(aDesc, miopenFloat, N,C,H,W);
    miopenSet4dTensorDescriptor(cDesc, miopenFloat, N,C,H,W);
    if (broadcast_bias) miopenSet4dTensorDescriptor(bDesc, miopenFloat, 1,C,1,1);
    else miopenSet4dTensorDescriptor(bDesc, miopenFloat, N,C,H,W);

    float alpha1=1.0f, alpha2=1.0f, beta=0.0f;
    miopenStatus_t es = miopenOpTensor(handle, op, &alpha1, aDesc, d_a, &alpha2, bDesc, d_b, &beta, cDesc, d_c);
    CHECK_HIP(hipDeviceSynchronize());

    bool ok=true;
    const char* opn = op==miopenTensorOpAdd?"ADD":op==miopenTensorOpMul?"MUL":op==miopenTensorOpMin?"MIN":"MAX";
    if (es != miopenStatusSuccess) {
        printf("op=%-4s N=%d C=%d H=%d W=%d bcast=%d -> EXEC FAILED status=%d\n", opn,N,C,H,W,broadcast_bias,(int)es);
        ok=false;
    } else {
        CHECK_HIP(hipMemcpy(h_gpu.data(), d_c, elems*4, hipMemcpyDeviceToHost));
        double cs = cos_sim(h_ref, h_gpu);
        printf("op=%-4s N=%d C=%d H=%d W=%d bcast=%d -> cos=%.5f%s\n", opn,N,C,H,W,broadcast_bias,cs, cs<0.999?"  <-- WRONG":"  OK");
        if (cs<0.999) ok=false;
    }

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    miopenDestroyTensorDescriptor(aDesc); miopenDestroyTensorDescriptor(bDesc); miopenDestroyTensorDescriptor(cDesc);
    miopenDestroy(handle);
    return ok;
}

int main() {
    miopenTensorOp_t ops[] = {miopenTensorOpAdd, miopenTensorOpMul, miopenTensorOpMin, miopenTensorOpMax};
    struct Case { int N,C,H,W; bool bcast; };
    Case cases[] = {
        {1, 24, 32, 100, true},
        {1, 24, 32, 100, false},
        {2, 64, 16, 16, true},
        {2, 64, 16, 16, false},
        {1, 1, 1, 1, true},
        {4, 8, 4, 4, true},
        {1, 512, 4, 4, true},
        {1, 3, 224, 224, true},
    };
    int total=0, wrong=0;
    for (auto op : ops) for (auto& c : cases) {
        total++;
        if (!run_one(op, c.N,c.C,c.H,c.W,c.bcast)) wrong++;
    }
    fflush(stdout);
    fprintf(stderr, "\nSUMMARY: %d cases tested, %d WRONG\n", total, wrong);
    return wrong>0?1:0;
}
