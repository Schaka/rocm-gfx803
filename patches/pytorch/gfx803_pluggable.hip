// hipMalloc-backed pluggable allocator for torch on gfx803 (experimental).
//
// Loaded by torch when GFX803_PLUGGABLE_ALLOCATOR is set at runtime (see the
// hook in torch/cuda/__init__.py); off by default. Every raw allocation comes
// from plain hipMalloc -- the same path the repo's C++ kernels use -- instead of
// torch's default caching allocator, whose block layout on gfx803 can corrupt the
// next kernel's access after a GEMM.
//
// Signatures match what torch's CUDAPluggableAllocator binds via ctypes:
//   void* gfx803_alloc(size_t size, int device, hipStream_t stream);
//   void  gfx803_free(void* ptr, size_t size, hipStream_t stream);
// (torch passes the raw function pointers through _cuda_customAllocator; the
// device/stream arguments are ignored -- hipMalloc allocates on the current device.)
//
// Build: hipcc --offload-arch=gfx803 -O2 -fPIC -shared gfx803_pluggable.hip \
//            -o libgfx803_pluggable.so -lamdhip64
#include <hip/hip_runtime.h>

extern "C" void* gfx803_alloc(size_t size, int device, hipStream_t stream) {
  void* p = nullptr;
  if (hipMalloc(&p, size) != hipSuccess) {
    return nullptr;
  }
  return p;
}

extern "C" void gfx803_free(void* ptr, size_t size, hipStream_t stream) {
  if (ptr) {
    hipFree(ptr);
  }
}
