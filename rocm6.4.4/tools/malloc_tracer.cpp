// LD_PRELOAD hipMalloc/hipFree logger, to directly observe rocBLAS's
// internal GSU workspace allocation pattern (sizes, alloc/free order,
// pointer reuse) instead of inferring it from source reading alone.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

typedef hipError_t (*hipMalloc_t)(void**, size_t);
typedef hipError_t (*hipFree_t)(void*);

static hipMalloc_t real_hipMalloc = nullptr;
static hipFree_t real_hipFree = nullptr;
static int call_num = 0;

extern "C" hipError_t hipMalloc(void** ptr, size_t size) {
    if (!real_hipMalloc) real_hipMalloc = (hipMalloc_t)dlsym(RTLD_NEXT, "hipMalloc");
    hipError_t r = real_hipMalloc(ptr, size);
    fprintf(stderr, "[malloc_tracer] #%d hipMalloc(size=%zu) -> %p\n", call_num++, size, *ptr);
    return r;
}

extern "C" hipError_t hipFree(void* ptr) {
    if (!real_hipFree) real_hipFree = (hipFree_t)dlsym(RTLD_NEXT, "hipFree");
    fprintf(stderr, "[malloc_tracer] #%d hipFree(%p)\n", call_num++, ptr);
    return real_hipFree(ptr);
}
