#!/usr/bin/env python3
"""On-hardware smoke test for the gfx803 (Polaris) image.

Run inside a container started with --device=/dev/kfd --device=/dev/dri
--group-add video. Exercises the three paths that only the base image's
build-time import check (Dockerfile.gfx803, final stage) doesn't cover:
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

    step("rocBLAS GEMM (torch)")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("gemm sum:", y.sum().item())

    step("MIOpen convolution (torch)")
    # gfx803 has no .kdb tuning DB here -- expect slow, not wrong or crashing.
    import torch.nn as nn

    conv = nn.Conv2d(3, 16, 3).cuda()
    xi = torch.randn(1, 3, 64, 64, device="cuda")
    yo = conv(xi)
    torch.cuda.synchronize()
    print("conv output shape:", tuple(yo.shape), "abs sum:", yo.abs().sum().item())

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
