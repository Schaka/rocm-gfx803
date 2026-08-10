// LD_PRELOAD interceptor for miopenSetTensorDescriptor (the "non-packed
// tensor shape" API) and miopenConvolutionForward, to log the real
// dims/strides ORT passes for the crashing CLAP conv layer -- the raw
// AMD_LOG_LEVEL kernel-arg trace can't render the stride struct arguments
// (they show up blank), so read them at the API level instead, before
// they're compiled into kernel args.
#include <miopen/miopen.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

typedef miopenStatus_t (*miopenSetTensorDescriptor_t)(miopenTensorDescriptor_t, miopenDataType_t, int, const int*, const int*);
typedef miopenStatus_t (*miopenSet4dTensorDescriptor_t)(miopenTensorDescriptor_t, miopenDataType_t, int, int, int, int);
typedef miopenStatus_t (*miopenConvolutionForward_t)(miopenHandle_t, const void*, const miopenTensorDescriptor_t, const void*,
                                                      const miopenTensorDescriptor_t, const void*, const miopenConvolutionDescriptor_t,
                                                      miopenConvFwdAlgorithm_t, const void*, const miopenTensorDescriptor_t, void*,
                                                      void*, size_t);

typedef miopenStatus_t (*miopenExecuteFusionPlan_t)(const miopenHandle_t, const miopenFusionPlanDescriptor_t,
                                                     const miopenTensorDescriptor_t, const void*,
                                                     const miopenTensorDescriptor_t, void*, miopenOperatorArgs_t);
typedef miopenStatus_t (*miopenCompileFusionPlan_t)(miopenHandle_t, miopenFusionPlanDescriptor_t);

static miopenSetTensorDescriptor_t real_set = nullptr;
static miopenConvolutionForward_t real_conv_fwd = nullptr;
static miopenExecuteFusionPlan_t real_exec_fusion = nullptr;
static miopenCompileFusionPlan_t real_compile_fusion = nullptr;

static int call_num = 0;

extern "C" miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType,
                                                     int nbDims, const int* dimsA, const int* stridesA) {
    if (!real_set) real_set = (miopenSetTensorDescriptor_t)dlsym(RTLD_NEXT, "miopenSetTensorDescriptor");
    fprintf(stderr, "[stride_tracer] #%d miopenSetTensorDescriptor desc=%p nbDims=%d dims=[", call_num++, (void*)tensorDesc, nbDims);
    for (int i = 0; i < nbDims; i++) fprintf(stderr, "%d%s", dimsA[i], i+1<nbDims?",":"");
    fprintf(stderr, "] strides=[");
    for (int i = 0; i < nbDims; i++) fprintf(stderr, "%d%s", stridesA ? stridesA[i] : -1, i+1<nbDims?",":"");
    fprintf(stderr, "]\n");
    fflush(stderr);
    return real_set(tensorDesc, dataType, nbDims, dimsA, stridesA);
}

extern "C" miopenStatus_t miopenConvolutionForward(miopenHandle_t handle, const void* alpha,
                                                    const miopenTensorDescriptor_t xDesc, const void* x,
                                                    const miopenTensorDescriptor_t wDesc, const void* w,
                                                    const miopenConvolutionDescriptor_t convDesc,
                                                    miopenConvFwdAlgorithm_t algo, const void* beta,
                                                    const miopenTensorDescriptor_t yDesc, void* y,
                                                    void* workSpace, size_t workSpaceSize) {
    if (!real_conv_fwd) real_conv_fwd = (miopenConvolutionForward_t)dlsym(RTLD_NEXT, "miopenConvolutionForward");

    auto dump = [](const char* name, miopenTensorDescriptor_t d) {
        int n=0; miopenGetTensorDescriptorSize(d, &n);
        fprintf(stderr, "  %s: descSize=%d", name, n);
        if (n == 4) {
            miopenDataType_t dt; int nn,c,h,w,ns,cs,hs,ws;
            miopenGet4dTensorDescriptor(d, &dt, &nn,&c,&h,&w,&ns,&cs,&hs,&ws);
            fprintf(stderr, " dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]", nn,c,h,w,ns,cs,hs,ws);
        }
        fprintf(stderr, "\n");
    };
    fprintf(stderr, "[stride_tracer] #%d miopenConvolutionForward algo=%d ws=%p wsSize=%zu x=%p w=%p y=%p\n",
            call_num++, (int)algo, workSpace, workSpaceSize, x, w, y);
    dump("xDesc", xDesc);
    dump("wDesc", wDesc);
    dump("yDesc", yDesc);

    return real_conv_fwd(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, beta, yDesc, y, workSpace, workSpaceSize);
}

extern "C" miopenStatus_t miopenCompileFusionPlan(miopenHandle_t handle, miopenFusionPlanDescriptor_t fusePlanDesc) {
    if (!real_compile_fusion) real_compile_fusion = (miopenCompileFusionPlan_t)dlsym(RTLD_NEXT, "miopenCompileFusionPlan");
    fprintf(stderr, "[stride_tracer] #%d miopenCompileFusionPlan plan=%p\n", call_num++, (void*)fusePlanDesc);
    miopenStatus_t st = real_compile_fusion(handle, fusePlanDesc);
    fprintf(stderr, "[stride_tracer]   compile status=%d\n", (int)st);
    return st;
}

extern "C" miopenStatus_t miopenExecuteFusionPlan(const miopenHandle_t handle, const miopenFusionPlanDescriptor_t fusePlanDesc,
                                                   const miopenTensorDescriptor_t inputDesc, const void* input,
                                                   const miopenTensorDescriptor_t outputDesc, void* output,
                                                   miopenOperatorArgs_t args) {
    if (!real_exec_fusion) real_exec_fusion = (miopenExecuteFusionPlan_t)dlsym(RTLD_NEXT, "miopenExecuteFusionPlan");

    auto dump = [](const char* name, miopenTensorDescriptor_t d) {
        int n=0; miopenGetTensorDescriptorSize(d, &n);
        fprintf(stderr, "  %s: descSize=%d", name, n);
        if (n == 4) {
            miopenDataType_t dt; int nn,c,h,w,ns,cs,hs,ws;
            miopenGet4dTensorDescriptor(d, &dt, &nn,&c,&h,&w,&ns,&cs,&hs,&ws);
            fprintf(stderr, " dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]", nn,c,h,w,ns,cs,hs,ws);
        }
        fprintf(stderr, "\n");
    };
    fprintf(stderr, "[stride_tracer] #%d miopenExecuteFusionPlan plan=%p input=%p output=%p\n",
            call_num++, (void*)fusePlanDesc, input, output);
    dump("inputDesc", inputDesc);
    dump("outputDesc", outputDesc);
    fflush(stderr);

    miopenStatus_t st = real_exec_fusion(handle, fusePlanDesc, inputDesc, input, outputDesc, output, args);
    fprintf(stderr, "[stride_tracer]   execute status=%d\n", (int)st);
    return st;
}
