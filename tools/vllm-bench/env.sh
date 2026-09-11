# Environment for the native (no-container) gfx803 ROCm + vLLM install.
#
# Sources of the settings below:
#   - LD_PRELOAD/TORCH_BLAS_PREFER_HIPBLASLT/ROCM_PATH mirror
#     tools/host-setup/native-rocm-vllm-setup.sh's gfx803-env.sh. The sgemm
#     shim must stay preloaded or rocBLAS's default gfx803 dispatch returns
#     silently wrong results.
#   - PYTHONPATH must reach the ROCm stack's own amdsmi bindings. vLLM's ROCm
#     platform plugin is gated on `import amdsmi`, and without it vLLM
#     resolves to UnspecifiedPlatform and dies in DeviceConfig with "Device
#     string must not be empty" -- an error that names nothing related to the
#     cause. The bindings shipped in /opt/rocm (<core>/share/amd_smi) match
#     the installed libamd_smi.so, which the PyPI amdsmi wheel need not.
#   - Every cache directory is moved to /data. This box's root filesystem is
#     a 117GB NVMe with well under half of it free, while /data is a 458GB
#     disk that is nearly empty. One torch.compile generation of this model
#     writes on the order of a gigabyte of inductor and Triton artifacts.
export LD_PRELOAD=/opt/rocm/lib/libgfx803_sgemm_shim.so
export LD_LIBRARY_PATH=/opt/rocm/lib
export PYTHONPATH=/opt/rocm/lib:/opt/rocm/share/amd_smi
export TORCH_BLAS_PREFER_HIPBLASLT=0
export ROCM_PATH=/opt/rocm
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
export VLLM_GFX803_GEMM_CACHE_MB=${VLLM_GFX803_GEMM_CACHE_MB:-768}

export XDG_CACHE_HOME=/data/cache
export TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor
export TRITON_CACHE_DIR=/data/cache/triton
export VLLM_CACHE_ROOT=/data/cache/vllm
export TORCH_EXTENSIONS_DIR=/data/cache/torch
