#!/usr/bin/env python3
"""On-hardware smoke test for the therock-experimental gfx803 image.

Adapted from ../verify.py -- same checks (MIGraphX EP inference, rocBLAS
GEMM numerics, MIOpen convolution), against this line's own image instead.

WGM8 guard note: this line's rocm-libraries patches now carry
0004-rocblas-wgm-miscompute-source-fix.patch (a source-level replacement
for the sed-based wgm-miscompute.sh, see README.md's "wgm-miscompute"
section for the real-hardware verification behind it) -- gated on
ISA==gfx803 && KernelLanguage==Assembly, so it correctly leaves the
already-correct HIP-source "_WGM8"-named fallback kernels alone rather
than zeroing every "_WGM8" string in the library the way the sed did.
The "no WGM8 symbols" guard below is therefore scoped to the *_KLA_
(assembly) kernels specifically, not a library-wide grep, to avoid
flagging those known-fine fallback kernels as a regression.

Run inside a container started with --device=/dev/kfd --device=/dev/dri
--group-add video.
"""

import re
import sys


def step(name):
    print(f"\n=== {name} ===", flush=True)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    step("onnxruntime providers")
    import onnxruntime as ort

    providers = ort.get_available_providers()
    print("available:", providers)
    if "MIGraphXExecutionProvider" not in providers:
        fail("MIGraphXExecutionProvider not built into this wheel")

    step("MIGraphX EP inference")
    import numpy as np
    import onnx
    from onnx import helper, TensorProto

    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 10
    onnx.save(model, "/tmp/relu.onnx")

    sess = ort.InferenceSession("/tmp/relu.onnx", providers=["MIGraphXExecutionProvider"])
    active = sess.get_providers()
    print("session providers:", active)
    if active[0] != "MIGraphXExecutionProvider":
        fail(f"expected MIGraphXExecutionProvider first, got {active}")

    out = sess.run(None, {"x": np.array([-1, 2, -3, 4], dtype=np.float32)})[0]
    print("relu output:", out)
    expected = np.array([0, 2, 0, 4], dtype=np.float32)
    if not np.array_equal(out, expected):
        fail(f"expected {expected}, got {out}")

    step("torch / rocm visibility")
    import torch

    print("torch", torch.__version__, "HIP built:", torch.version.hip)
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False -- card not visible to torch")
    print("device:", torch.cuda.get_device_name(0))

    step("rocBLAS GEMM numerics (torch)")
    for m, n, k in ((512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (1024, 64, 1024)):
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        got = a @ b
        torch.cuda.synchronize()
        ref = (a.double().cpu() @ b.double().cpu())
        scale = ref.abs().max().item()
        rel = (got.double().cpu() - ref).abs().max().item() / max(scale, 1e-30)
        print(f"  {m}x{n}x{k}: max rel err {rel:.3g}")
        if rel > 1e-4:
            fail(f"GEMM {m}x{n}x{k} is numerically wrong (max rel err {rel:.3g}). "
                 "rocBLAS reports success regardless -- check for _KLA_..._WGM8 "
                 "kernels in /opt/rocm/**/rocblas/library.")

    step("no WorkGroupMapping!=1 assembly kernels in the rocBLAS library")
    # Scoped to _KLA_ (assembly) kernel names specifically -- the fallback
    # library's _KLS_ (HIP-source) kernels legitimately carry "_WGM8" in
    # their name too, and are correct regardless (see README.md's
    # wgm-miscompute section for the real-hardware verification of that).
    import glob
    import os

    offenders = []
    for root, _, files in os.walk("/opt/rocm"):
        if "rocblas" not in root or "library" not in root:
            continue
        for name in files:
            if "gfx803" not in name:
                continue
            path = os.path.join(root, name)
            with open(path, "rb") as fh:
                data = fh.read()
            n = len(re.findall(rb"Cijk_[A-Za-z0-9_]*_KLA_[A-Za-z0-9_]*_WGM8", data))
            if n:
                offenders.append((name, n))
    if offenders:
        for name, n in offenders[:5]:
            print(f"  {n:4d}  {name}")
        total = sum(n for _, n in offenders)
        fail(f"{total} assembly (_KLA_) WGM8 kernels across {len(offenders)} gfx803 "
             "library files -- these miscompute silently on this arch")
    print("checked gfx803 rocblas library files, no assembly WGM8 kernels")

    step("MIOpen convolution (torch)")
    import torch.nn as nn

    conv = nn.Conv2d(3, 16, 3).cuda()
    xi = torch.randn(1, 3, 64, 64, device="cuda")
    yo = conv(xi)
    torch.cuda.synchronize()
    ref = nn.functional.conv2d(xi.cpu(), conv.weight.detach().cpu(), conv.bias.detach().cpu())
    rel = (yo.cpu() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
    print("conv output shape:", tuple(yo.shape), "max rel err:", f"{rel:.3g}")
    if rel > 1e-3:
        fail(f"conv2d is numerically wrong (max rel err {rel:.3g})")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
