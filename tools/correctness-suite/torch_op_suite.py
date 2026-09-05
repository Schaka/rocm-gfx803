#!/usr/bin/env python3

import torch
import traceback

torch.manual_seed(1234)

DEVICE = torch.device("cuda")

print("=" * 90)
print("gfx803 TORCH OP CORRECTNESS BENCHMARK")
print("CPU REFERENCE vs GPU EAGER vs GPU INDUCTOR")
print("=" * 90)

print("Torch :", torch.__version__)
print("HIP   :", torch.version.hip)
print("GPU   :", torch.cuda.get_device_name(0))
print("Device:", DEVICE)

results = []


# =====================================================================
# HELPERS
# =====================================================================

def finite(x):
    return torch.isfinite(x).all().item()


def stats(x):
    xf = x.detach().float().cpu()

    if not torch.isfinite(xf).all():
        return {
            "finite": False,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }

    return {
        "finite": True,
        "min": xf.min().item(),
        "max": xf.max().item(),
        "mean": xf.mean().item(),
    }


def compare(name, result, reference, atol, rtol):

    try:
        result_cpu = result.detach().float().cpu()
        reference_cpu = reference.detach().float().cpu()

        if result_cpu.shape != reference_cpu.shape:
            print(f"[ERROR] {name}")
            print("  shape result :", tuple(result_cpu.shape))
            print("  shape ref    :", tuple(reference_cpu.shape))

            results.append((name, "ERROR"))
            return

        if not torch.isfinite(result_cpu).all():
            bad = (~torch.isfinite(result_cpu)).sum().item()

            print(f"[BAD] {name}")
            print("  NONFINITE:", bad)

            results.append((name, "NONFINITE"))
            return

        diff = (result_cpu - reference_cpu).abs()

        maxerr = diff.max().item()
        meanerr = diff.mean().item()

        ok = torch.allclose(
            result_cpu,
            reference_cpu,
            atol=atol,
            rtol=rtol,
        )

        status = "OK" if ok else "BAD"

        print(
            f"[{status}] {name:<50} "
            f"max={maxerr:.6g} mean={meanerr:.6g}"
        )

        if not ok:
            print(
                f"      tolerance: atol={atol} rtol={rtol}"
            )

        results.append((name, status))

    except Exception as e:

        print(f"[ERROR] {name}")
        print(" ", type(e).__name__ + ":", e)

        results.append((name, "ERROR"))


def run_test(
    name,
    eager_fn,
    cpu_reference_fn,
    inductor_fn=None,
    atol=1e-3,
    rtol=1e-3,
):

    print("\n" + "-" * 90)
    print(name)
    print("-" * 90)

    # -------------------------------------------------------------
    # CPU reference
    # -------------------------------------------------------------

    try:
        with torch.no_grad():
            reference = cpu_reference_fn()

        print("CPU reference:", stats(reference))

    except Exception as e:

        print("[ERROR] CPU reference failed:")
        print(type(e).__name__ + ":", e)

        results.append((name + " CPU", "ERROR"))
        return

    # -------------------------------------------------------------
    # GPU EAGER
    # -------------------------------------------------------------

    try:

        with torch.no_grad():
            eager = eager_fn()

        s = stats(eager)

        print("GPU Eager:", s)

        compare(
            name + " / EAGER",
            eager,
            reference,
            atol,
            rtol,
        )

    except Exception as e:

        print("[ERROR] GPU Eager failed:")
        print(type(e).__name__ + ":", e)

        results.append((name + " / EAGER", "ERROR"))

        eager = None

    # -------------------------------------------------------------
    # GPU INDUCTOR
    # -------------------------------------------------------------

    if inductor_fn is not None:

        try:

            compiled = torch.compile(
                inductor_fn,
                backend="inductor",
            )

            with torch.no_grad():
                ind = compiled()

            s = stats(ind)

            print("GPU Inductor:", s)

            compare(
                name + " / INDUCTOR",
                ind,
                reference,
                atol,
                rtol,
            )

        except Exception as e:

            print("[ERROR] Inductor failed:")
            print(type(e).__name__ + ":", e)

            results.append((name + " / INDUCTOR", "ERROR"))

        finally:

            try:
                torch.compiler.reset()
            except Exception:
                pass

            torch.cuda.empty_cache()


def rand(shape, dtype):
    return torch.randn(
        *shape,
        dtype=dtype,
        device=DEVICE,
    )


# =====================================================================
# ELEMENTWISE
# =====================================================================

for dtype in [torch.float32, torch.float16]:

    x_cpu = torch.randn(4096, dtype=torch.float32)

    x_gpu = x_cpu.to(DEVICE, dtype=dtype)

    for opname, op in [
        ("add", lambda x: x + 1.25),
        ("sub", lambda x: x - 1.25),
        ("mul", lambda x: x * 1.25),
        ("div", lambda x: x / 1.25),
        ("sqrt", lambda x: torch.sqrt(x.abs() + 0.01)),
        ("exp", lambda x: torch.exp(x.clamp(-5, 5))),
        ("log", lambda x: torch.log(x.abs() + 0.01)),
        ("sin", torch.sin),
        ("cos", torch.cos),
        ("tanh", torch.tanh),
        ("sigmoid", torch.sigmoid),
        ("relu", torch.relu),
        ("silu", torch.nn.functional.silu),
        ("gelu", torch.nn.functional.gelu),
    ]:

        cpu_op = lambda op=op: op(x_cpu)

        gpu_op = lambda op=op: op(x_gpu)

        run_test(
            f"{opname} [{dtype}]",
            gpu_op,
            cpu_op,
            atol=0.1 if dtype == torch.float16 else 1e-5,
            rtol=0.02 if dtype == torch.float16 else 1e-5,
        )


# =====================================================================
# REDUCTIONS
# =====================================================================

x_cpu = torch.randn(8, 320, 32, 32)
x_gpu32 = x_cpu.cuda()
x_gpu16 = x_cpu.half().cuda()

for dtype, x_gpu in [
    (torch.float32, x_gpu32),
    (torch.float16, x_gpu16),
]:

    # IMPORTANT:
    # Explicitly compare against CPU reference.
    # No GPU-vs-GPU comparison here.

    run_test(
        f"sum [{dtype}]",
        lambda x=x_gpu: x.sum(),
        lambda dtype=dtype: x_cpu.to(dtype).sum().float(),
        atol=0.5,
        rtol=0.02,
    )

    run_test(
        f"mean [{dtype}]",
        lambda x=x_gpu: x.mean(),
        lambda dtype=dtype: x_cpu.to(dtype).mean().float(),
        atol=0.01,
        rtol=0.02,
    )

    run_test(
        f"amax [{dtype}]",
        lambda x=x_gpu: x.amax(),
        lambda dtype=dtype: x_cpu.to(dtype).amax().float(),
        atol=0.01,
        rtol=0.02,
    )


# =====================================================================
# MATMUL
# =====================================================================

matmul_shapes = [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (512, 512, 640),
    (640, 640, 512),
    (1024, 1024, 1024),
    (1280, 1280, 1280),
    (1920, 1920, 1920),
    (320, 1280, 640),
    (640, 640, 1280),
]


for M, K, N in matmul_shapes:

    for dtype in [torch.float32, torch.float16]:

        a_cpu = torch.randn(M, K)
        b_cpu = torch.randn(K, N)

        a_gpu = a_cpu.to(DEVICE, dtype=dtype)
        b_gpu = b_cpu.to(DEVICE, dtype=dtype)

        def cpu_ref(a=a_cpu, b=b_cpu, dtype=dtype):
            return torch.mm(
                a.to(dtype),
                b.to(dtype),
            ).float()

        def gpu_op(a=a_gpu, b=b_gpu):
            return torch.mm(a, b)

        run_test(
            f"matmul_{M}x{K}x{N} [{dtype}]",
            gpu_op,
            cpu_ref,
            gpu_op,
            atol=0.5 if dtype == torch.float16 else 0.01,
            rtol=0.02 if dtype == torch.float16 else 0.01,
        )


# =====================================================================
# BMM
# =====================================================================

bmm_shapes = [
    (2, 64, 64, 64),
    (2, 128, 64, 128),
    (4, 256, 64, 256),
    (8, 256, 80, 256),
    (2, 1024, 64, 1024),
]


for B, M, K, N in bmm_shapes:

    for dtype in [torch.float32, torch.float16]:

        a_cpu = torch.randn(B, M, K)
        b_cpu = torch.randn(B, K, N)

        a_gpu = a_cpu.to(DEVICE, dtype=dtype)
        b_gpu = b_cpu.to(DEVICE, dtype=dtype)

        def cpu_ref(a=a_cpu, b=b_cpu, dtype=dtype):
            return torch.bmm(
                a.to(dtype).float(),
                b.to(dtype).float(),
            )

        def gpu_op(a=a_gpu, b=b_gpu):
            return torch.bmm(a, b)

        run_test(
            f"bmm_{B}x{M}x{K}x{N} [{dtype}]",
            gpu_op,
            cpu_ref,
            gpu_op,
            atol=0.5 if dtype == torch.float16 else 0.1,
            rtol=0.02,
        )


# =====================================================================
# CONV2D
# =====================================================================

conv_tests = [
    (3, 64, 64, 64),
    (64, 64, 64, 64),
    (64, 128, 32, 32),
    (128, 128, 32, 32),
    (128, 256, 16, 16),
    (320, 320, 16, 16),
    (320, 640, 16, 16),
    (640, 640, 8, 8),
]


for Cin, Cout, H, W in conv_tests:

    for dtype in [torch.float32, torch.float16]:

        x_cpu = torch.randn(1, Cin, H, W)
        weight_cpu = torch.randn(Cout, Cin, 3, 3)
        bias_cpu = torch.randn(Cout)

        x_gpu = x_cpu.to(DEVICE, dtype=dtype)
        weight_gpu = weight_cpu.to(DEVICE, dtype=dtype)
        bias_gpu = bias_cpu.to(DEVICE, dtype=dtype)

        def cpu_ref(
            x=x_cpu,
            w=weight_cpu,
            b=bias_cpu,
            dtype=dtype,
        ):
            return torch.nn.functional.conv2d(
                x.to(dtype).float(),
                w.to(dtype).float(),
                b.to(dtype).float(),
                padding=1,
            )

        def gpu_op(
            x=x_gpu,
            w=weight_gpu,
            b=bias_gpu,
        ):
            return torch.nn.functional.conv2d(
                x,
                w,
                b,
                padding=1,
            )

        run_test(
            f"conv2d_{Cin}->{Cout}_{H}x{W} [{dtype}]",
            gpu_op,
            cpu_ref,
            gpu_op,
            atol=0.5 if dtype == torch.float16 else 0.1,
            rtol=0.02,
        )


# =====================================================================
# GROUPNORM
#
# FIX: weight AND bias are explicitly placed on the same device.
# =====================================================================

for C in [32, 64, 128, 320, 640, 1280]:

    groups = 32

    x_cpu = torch.randn(2, C, 16, 16)
    weight_cpu = torch.randn(C)
    bias_cpu = torch.randn(C)

    for dtype in [torch.float32, torch.float16]:

        x_gpu = x_cpu.to(DEVICE, dtype=dtype)
        weight_gpu = weight_cpu.to(DEVICE, dtype=dtype)
        bias_gpu = bias_cpu.to(DEVICE, dtype=dtype)

        def cpu_ref(
            x=x_cpu,
            w=weight_cpu,
            b=bias_cpu,
            dtype=dtype,
        ):
            return torch.nn.functional.group_norm(
                x.to(dtype).float(),
                groups,
                w.to(dtype).float(),
                b.to(dtype).float(),
            )

        def gpu_op(
            x=x_gpu,
            w=weight_gpu,
            b=bias_gpu,
        ):
            return torch.nn.functional.group_norm(
                x,
                groups,
                w,
                b,
            )

        run_test(
            f"groupnorm_C{C} [{dtype}]",
            gpu_op,
            cpu_ref,
            gpu_op,
            atol=0.1 if dtype == torch.float16 else 1e-3,
            rtol=0.02,
        )


# =====================================================================
# LAYERNORM
#
# FIX: weight AND bias explicitly moved to GPU.
# =====================================================================

for C in [320, 640, 768, 1280]:

    x_cpu = torch.randn(4, 128, C)
    weight_cpu = torch.randn(C)
    bias_cpu = torch.randn(C)

    for dtype in [torch.float32, torch.float16]:

        x_gpu = x_cpu.to(DEVICE, dtype=dtype)
        weight_gpu = weight_cpu.to(DEVICE, dtype=dtype)
        bias_gpu = bias_cpu.to(DEVICE, dtype=dtype)

        def cpu_ref(
            x=x_cpu,
            w=weight_cpu,
            b=bias_cpu,
            dtype=dtype,
        ):
            return torch.nn.functional.layer_norm(
                x.to(dtype).float(),
                (C,),
                w.to(dtype).float(),
                b.to(dtype).float(),
            )

        def gpu_op(
            x=x_gpu,
            w=weight_gpu,
            b=bias_gpu,
        ):
            return torch.nn.functional.layer_norm(
                x,
                (C,),
                w,
                b,
            )

        run_test(
            f"layernorm_{C} [{dtype}]",
            gpu_op,
            cpu_ref,
            gpu_op,
            atol=0.1 if dtype == torch.float16 else 1e-3,
            rtol=0.02,
        )


# =====================================================================
# SOFTMAX
# =====================================================================

for C in [64, 320, 640, 1280]:

    x_cpu = torch.randn(2, 128, C)

    for dtype in [torch.float32, torch.float16]:

        x_gpu = x_cpu.to(DEVICE, dtype=dtype)

        run_test(
            f"softmax_{C} [{dtype}]",
            lambda x=x_gpu: torch.softmax(x, dim=-1),
            lambda x=x_cpu, dtype=dtype:
                torch.softmax(x.to(dtype), dim=-1).float(),
            lambda x=x_gpu: torch.softmax(x, dim=-1),
            atol=0.01,
            rtol=0.02,
        )


# =====================================================================
# ATTENTION
# =====================================================================

attention_tests = [
    (1, 8, 64, 40),
    (1, 8, 128, 40),
    (1, 8, 256, 40),
    (1, 8, 4096, 40),
]


for B, H, S, D in attention_tests:

    for dtype in [torch.float32, torch.float16]:

        q_cpu = torch.randn(B, H, S, D)
        k_cpu = torch.randn(B, H, S, D)
        v_cpu = torch.randn(B, H, S, D)

        q_gpu = q_cpu.to(DEVICE, dtype=dtype)
        k_gpu = k_cpu.to(DEVICE, dtype=dtype)
        v_gpu = v_cpu.to(DEVICE, dtype=dtype)

        scale = D ** -0.5

        def cpu_ref(
            q=q_cpu,
            k=k_cpu,
            v=v_cpu,
            dtype=dtype,
        ):

            q = q.to(dtype).float()
            k = k.to(dtype).float()
            v = v.to(dtype).float()

            scores = torch.matmul(
                q,
                k.transpose(-2, -1),
            ) * scale

            probs = torch.softmax(scores, dim=-1)

            return torch.matmul(probs, v)

        def gpu_op(
            q=q_gpu,
            k=k_gpu,
            v=v_gpu,
        ):

            scores = torch.matmul(
                q,
                k.transpose(-2, -1),
            ) * scale

            probs = torch.softmax(
                scores,
                dim=-1,
            )

            return torch.matmul(
                probs,
                v,
            )

        run_test(
            f"attention_B{B}_H{H}_S{S}_D{D} [{dtype}]",
            gpu_op,
            cpu_ref,
            gpu_op,
            atol=0.5 if dtype == torch.float16 else 0.05,
            rtol=0.02,
        )


# =====================================================================
# SUMMARY
# =====================================================================

print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)

total = len(results)
bad = sum(x[1] in ("BAD", "NONFINITE", "ERROR") for x in results)
errors = sum(x[1] == "ERROR" for x in results)
nonfinite = sum(x[1] == "NONFINITE" for x in results)
incorrect = sum(x[1] == "BAD" for x in results)

print("Total results :", total)
print("BAD           :", incorrect)
print("NONFINITE     :", nonfinite)
print("ERROR         :", errors)
print("PASS          :", total - bad)

print("\nFailures:")
for name, status in results:
    if status != "OK":
        print(f"{status:<12} {name}")

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
