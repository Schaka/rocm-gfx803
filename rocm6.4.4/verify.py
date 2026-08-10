#!/usr/bin/env python3
"""On-hardware smoke test for the gfx803 (Polaris) image.

Run inside a container started with --device=/dev/kfd --device=/dev/dri
--group-add video. Exercises the three paths that only the base image's
build-time import check (gfx803/Dockerfile, final stage) doesn't cover:
MIGraphX EP inference, rocBLAS GEMM, MIOpen convolution -- each of which
can import fine and still fail (or silently fall back) on real hardware.

See README.gfx803.md#verifying-on-hardware.
"""

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
    # onnx's default IR/opset can outrun what ORT 1.21.1 supports
    # (max IR version 10, ai.onnx opset ceiling 22) -- pin both explicitly.
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
    # This used to print the sum and assert nothing, which is exactly how gfx803
    # shipped a rocBLAS whose sgemm returned garbage: every Tensile assembly
    # kernel built with WorkGroupMapping != 1 miscomputes on this arch, and
    # rocBLAS still reports success. See patches/rocblas/wgm-miscompute.sh.
    #
    # Shapes matter more than size here. Which Tensile solution gets picked is a
    # function of M/N/K, and the broken kernels were only selected from M>=768 --
    # a single square 2048 case would have caught this one, but a single 512 case
    # would not have. These cover both sides of that boundary and a non-square
    # case, since M and N select differently.
    for m, n, k in ((512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (1024, 64, 1024)):
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        got = a @ b
        torch.cuda.synchronize()
        # float64 on the CPU: a float32 reference accumulates in a different
        # order and would blur the difference between "fp32 rounding" and
        # "wrong answer", which here differ by ~6 orders of magnitude.
        ref = (a.double().cpu() @ b.double().cpu())
        scale = ref.abs().max().item()
        rel = (got.double().cpu() - ref).abs().max().item() / max(scale, 1e-30)
        print(f"  {m}x{n}x{k}: max rel err {rel:.3g}")
        if rel > 1e-4:
            fail(f"GEMM {m}x{n}x{k} is numerically wrong (max rel err {rel:.3g}). "
                 "rocBLAS reports success regardless -- check for WGM8 kernels in "
                 "/opt/rocm/lib/rocblas/library.")

    step("no WorkGroupMapping!=1 kernels in the rocBLAS library")
    # Direct regression guard for the above: kernel names embed the parameters
    # they were generated with, so a correctly patched library contains no
    # ..._WGM8 symbol at all. Cheap, needs no GPU, and names the defect exactly.
    import glob
    import os

    offenders = []
    libdir = "/opt/rocm/lib/rocblas/library"
    for path in glob.glob(os.path.join(libdir, "*gfx803*")):
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            n = fh.read().count(b"_WGM8")
        if n:
            offenders.append((os.path.basename(path), n))
    if offenders:
        for name, n in offenders[:5]:
            print(f"  {n:4d}  {name}")
        total = sum(n for _, n in offenders)
        fail(f"{total} WGM8 kernels across {len(offenders)} gfx803 library files -- "
             "these miscompute silently on this arch")
    print(f"checked {len(glob.glob(os.path.join(libdir, '*gfx803*')))} gfx803 library files, no WGM8 kernels")

    step("MIOpen convolution (torch)")
    # gfx803 has no .kdb tuning DB here -- expect slow, not wrong or crashing.
    import torch.nn as nn

    conv = nn.Conv2d(3, 16, 3).cuda()
    xi = torch.randn(1, 3, 64, 64, device="cuda")
    yo = conv(xi)
    torch.cuda.synchronize()
    # Same reasoning as the GEMM above: shape alone proves nothing.
    ref = nn.functional.conv2d(xi.cpu(), conv.weight.detach().cpu(), conv.bias.detach().cpu())
    rel = (yo.cpu() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
    print("conv output shape:", tuple(yo.shape), "max rel err:", f"{rel:.3g}")
    if rel > 1e-3:
        fail(f"conv2d is numerically wrong (max rel err {rel:.3g})")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
