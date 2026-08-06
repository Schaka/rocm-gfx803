# Repro of the documented gfx803 CLAP crash, WITHOUT the plugin's
# ConvActivationFusion workaround -- i.e. the exact "revert" the user asked for:
# ROCMExecutionProvider gets the fused Conv+Bias+Activation nodes ORT's default
# optimizer produces, same as any other arch.
#
# docs/ARCH_NOTES.md: "Session churn is the amplifier, not the cause: a churn
# loop usually dies within a handful of iterations, but a crash on the very
# first session of a fresh process was also observed. One long-lived session
# survived 180 consecutive inferences." -- so this churns sessions, not just
# inferences within one, to match the reproduction that was actually used.
import sys
import time
import numpy as np
import onnxruntime as ort

MODEL = "/model/model_epoch_36.onnx"
N_SESSIONS = 40

print("ORT", ort.__version__, "providers avail:", ort.get_available_providers(), flush=True)

so = ort.SessionOptions()
# Deliberately NOT setting optimization.disable_specified_optimizers here --
# that is exactly the line being reverted.

rng = np.random.default_rng(0)

for i in range(N_SESSIONS):
    t0 = time.time()
    sess = ort.InferenceSession(
        MODEL, sess_options=so,
        providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
    )
    active = sess.get_providers()
    x = rng.standard_normal((1, 1, 128, 400), dtype=np.float32)
    out = sess.run(None, {"mel_spectrogram": x})[0]
    dt = time.time() - t0
    print(f"session {i:3d}: providers={active[0]:24s} out_shape={out.shape} "
          f"out[0,:3]={out[0,:3]} nan={np.isnan(out).any()} {dt:.3f}s", flush=True)
    del sess

print("ALL SESSIONS COMPLETED WITHOUT CRASH", flush=True)
