#!/usr/bin/env python3
"""On-hardware smoke test for the gfx803 (Polaris) image.

Run inside a container started with --device=/dev/kfd --device=/dev/dri
--group-add video. Each step covers a path that the base image's build-time
import check does not: the MIGraphX EP, rocBLAS GEMM, MIOpen convolution, the
rocSOLVER device code, torch.linalg, and cross-dispatch coherence. Every one of
them can import cleanly and still fail, or fall back silently, on real hardware.

See README.gfx803.md#verifying-on-hardware.
"""

import sys


def step(name):
    print(f"\n=== {name} ===", flush=True)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def hip_fatbin_state(path):
    """(has_device_code, size) for an ELF64 shared library's .hip_fatbin section.

    binutils is not in the runtime image, so the section header table is read
    directly. SHT_PROGBITS (1) carries file bytes; SHT_NOBITS (8) is a
    reservation with none, which is precisely how a HIP library that linked
    without any offload target ships.
    """
    import struct

    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:4] != b"\x7fELF" or blob[4] != 2:
        return None, 0
    e_shoff = struct.unpack_from("<Q", blob, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    strtab_off = struct.unpack_from("<Q", blob, e_shoff + e_shstrndx * e_shentsize + 0x18)[0]
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        name_off, sh_type = struct.unpack_from("<II", blob, sh)
        sh_size = struct.unpack_from("<Q", blob, sh + 0x20)[0]
        end = blob.index(b"\0", strtab_off + name_off)
        if blob[strtab_off + name_off:end] == b".hip_fatbin":
            return sh_type == 1, sh_size
    return None, 0


def main():
    step("onnxruntime providers")
    import onnxruntime as ort

    providers = ort.get_available_providers()
    print("available:", providers)
    if "MIGraphXExecutionProvider" not in providers:
        fail("MIGraphXExecutionProvider not built into this wheel")

    step("MIGraphX EP inference")
    # onnx's default IR/opset can outrun what ORT 1.21.1 supports
    # (max IR version 10, ai.onnx opset ceiling 22). Pin both explicitly.
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
    # Shape matters more than size here. The picked Tensile solution depends on
    # M/N/K, and the broken kernels were selected only from M>=768. One square
    # one square 2048 case exposes it and one 512 case does not. These cases cover
    # both sides of that boundary, plus a non-square case, because M and N select
    # differently.
    for m, n, k in ((512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (1024, 64, 1024)):
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        got = a @ b
        torch.cuda.synchronize()
        # float64 on the CPU: a float32 reference accumulates in a different
        # order and blurs the difference between "fp32 rounding" and
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
    # gfx803 has no .kdb tuning DB here. Expect slow, and not wrong or crashing.
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

    step("rocSOLVER embeds device code")
    # The stock 10.0 and 7.14 rocSOLVER ships host stubs, a kernel-registration
    # table (.hipFatBinSegment), and an empty .hip_fatbin. That section is
    # SHT_NOBITS with zero file bytes, so the library holds no code object for
    # any architecture. Every hipSOLVER-backed torch.linalg entry point then dies
    # inside hipLaunchKernel on a bogus kernel handle. That reads like a ROCclr
    # crash, but it is a build that linked successfully and shipped nothing. The
    # section header table is read directly here, because binutils is not in the
    # runtime image. The same check is a build-time gate in the Dockerfile's
    # rocsolver-builder stage.
    solver = None
    for cand in ("/opt/rocm/lib/librocsolver.so", "/opt/rocm/lib/rocsolver/librocsolver.so"):
        if os.path.exists(cand):
            solver = cand
            break
    if solver is None:
        fail("no librocsolver.so under /opt/rocm/lib -- torch.linalg cannot work")
    real, fatbin_size = hip_fatbin_state(os.path.realpath(solver))
    print(f"  {os.path.realpath(solver)}: .hip_fatbin "
          + (f"{fatbin_size} bytes of device code" if real else f"absent or SHT_NOBITS ({fatbin_size} reserved)"))
    if not real or fatbin_size < 100000:
        fail("librocsolver.so carries no device code for this architecture")

    step("torch.linalg numerics")
    # This runs in a subprocess on purpose. The failure mode above is a SIGSEGV,
    # and a smoke test that dies with the thing it tests reports nothing.
    import subprocess

    linalg_code = """
import torch
torch.manual_seed(0)
dev = "cuda"
a = torch.randn(64, 64, device=dev)
spd = a @ a.T + 64 * torch.eye(64, device=dev)
rv = {}
q, r = torch.linalg.qr(a)
rv["qr"] = torch.allclose(q @ r, a, atol=1e-3)
rv["cholesky"] = torch.allclose(torch.linalg.cholesky(spd) @ torch.linalg.cholesky(spd).T.mT, spd, atol=1e-2)
rv["solve"] = torch.allclose(torch.linalg.solve(spd, a), spd.inverse() @ a, atol=1e-2)
u, s, vh = torch.linalg.svd(a)
rv["svd"] = torch.allclose(u @ torch.diag(s) @ vh, a, atol=1e-2)
rv["inv"] = torch.allclose(a.inverse() @ a, torch.eye(64, device=dev), atol=1e-2)
bad = [k for k, v in rv.items() if not v]
print("LINALIGN_RESULT", rv)
raise SystemExit(1 if bad else 0)
"""
    proc = subprocess.run([sys.executable, "-c", linalg_code], capture_output=True, text=True, timeout=900)
    tail = [l for l in proc.stdout.splitlines() if l.startswith("LINALIGN_RESULT")]
    if proc.returncode != 0 and not tail:
        fail("torch.linalg crashed (rc=%d). An empty .hip_fatbin looks exactly like "
             "this: %s" % (proc.returncode, proc.stderr.strip().splitlines()[-3:]))
    print("  " + (tail[0] if tail else "no result line"))
    if proc.returncode != 0:
        fail("torch.linalg returned wrong results; see the per-op flags above")

    step("cross-dispatch coherence (recycled VAs, host copy)")
    # Guards patches/rocm-systems/gfx803-tc-invalidate-acquire-mem.patch. Both
    # directions of the defect are silent. A kernel can be served stale TC lines
    # for a VA that an earlier kernel overwrote. And a copy engine reads DRAM and
    # never the shader TC, so it can copy out bytes that the writer kernel has
    # not written back yet. In that second case the GPU result is right and its
    # host copy is not. The shapes below are the ones that reproduced the fault
    # under a caching allocator that recycles a handful of VAs. One round is
    # enough to catch it, because the copy-engine direction is deterministic and
    # not a race.
    conv = nn.Conv2d(320, 320, 3, padding=1).cuda().double()
    xn = torch.randn(1, 320, 16, 16, device="cuda")
    for _ in range(6):
        y = conv(xn.float())
        got = y.cpu().double()
        ref = nn.functional.conv2d(xn.double(), conv.weight.detach().cpu(), conv.bias.detach().cpu())
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
        if rel > 1e-3:
            fail(f"host copy of a conv disagrees with the CPU reference (max rel err {rel:.3g}) "
                 "-- stale-read/coherence regression")
        stale = y.clone()
        torch.cuda.synchronize()
        if not torch.equal(stale, y):
            fail("a GPU->GPU copy of a tensor disagrees with the tensor itself")
    print(f"  conv + host copy + frozen copy consistent, max rel err {rel:.3g}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
