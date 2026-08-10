#!/usr/bin/env python3
"""Layer 4: real-model op-coverage bisection.

Generalizes the CLAP-specific scan scripts used to find the Winograd
fused-CBA bug into a model-agnostic tool: point it at any ONNX model, it
extracts a real per-node subgraph for every node with a numeric tensor
output (via onnx.utils.extract_model -- NOT tapping an extra output on the
full graph, which suppresses graph-optimizer fusion and hides exactly the
class of bug this whole suite exists to find), runs it through CPUExecutionProvider
(reference) vs ROCMExecutionProvider with and without ConvActivationFusion,
and reports cosine similarity per node. A node whose fused cos is much worse
than its unfused cos is the signature of the bug class already found twice
in this project (fusion-plan-wrapped kernel miscomputing a shape its
non-fused sibling gets right).

Meant to run unattended overnight against a fleet of models, same as the
MIOpen op sweeps in this directory -- structured output for later analysis,
not just a println tool.

Usage:
    MODEL_PATH=/path/to/model.onnx python3 model_bisect.py \
        --input-shape 1,1,128,400 --results-dir /tmp/bisect-results

Output: <results-dir>/<model-basename>.csv with columns
    node_idx,node_name,op_type,cos_fused,cos_unfused,flag
plus a live progress line per node to stdout.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import onnx
import onnx.utils
import onnxruntime as ort


def cos_sim(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    denom = na * nb
    # An additive epsilon here (the previous `+ 1e-12`) silently corrupts the
    # ratio whenever both vectors have a legitimately tiny-but-nonzero
    # magnitude -- 1e-12 can dwarf denom and crush a correct near-1.0 cosine
    # down to ~0, misreported as a divergence. float64 division is safe far
    # below that; only guard the actual 0/0 case. (Same bug, same fix, as
    # common.hpp's cos_sim in this same directory.)
    if denom == 0.0:
        return 1.0  # both vectors exactly zero -- legitimate match
    return float(a @ b / denom)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH"), help="ONNX model path (or set MODEL_PATH)")
    ap.add_argument("--input-shape", required=True,
                     help="comma-separated shape applied to every graph input, e.g. 1,3,224,224 "
                          "(covers both single-input models like MobileNet/Whisper and "
                          "multi-input-but-same-shape models like BERT's input_ids+attention_mask)")
    ap.add_argument("--results-dir", default="./bisect-results")
    ap.add_argument("--threshold", type=float, default=0.9, help="cos below this on the fused path flags a node")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only scan the first N eligible nodes (for a quick pass)")
    ap.add_argument("--tmp-dir", default="/tmp", help="scratch dir for extracted subgraphs -- use a real-disk path, not a quota-limited tmpfs")
    ap.add_argument("--input-file", default=None,
                     help="load a .npy array for the first float input instead of raw N(0,1) noise "
                          "-- use when raw noise causes generic compounding-FP-drift divergence in "
                          "deep networks unrelated to any real kernel bug (in-distribution input "
                          "keeps activations in the range the model was actually trained/tuned for)")
    args = ap.parse_args()

    if not args.model:
        print("FATAL: --model or MODEL_PATH required", file=sys.stderr)
        sys.exit(1)

    in_shape = [int(x) for x in args.input_shape.split(",")]
    os.makedirs(args.results_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    csv_path = os.path.join(args.results_dir, f"{model_name}.csv")

    m = onnx.load(args.model)
    graph = m.graph
    in_names = [i.name for i in graph.input]

    # Every node with at least one output is a bisection candidate --
    # not just Conv. Real correctness bugs have been found in Conv so far,
    # but activ/pool/norm ops are equally arch-blind and untested at the
    # real-model level (only synthetically, via the op sweeps).
    candidates = [n for n in graph.node if len(n.output) > 0 and n.output[0]]
    if args.limit:
        candidates = candidates[: args.limit]

    rng = np.random.default_rng(args.seed)
    # ONNX elem_type 7 == INT64, 1 == FLOAT -- generate per-input data
    # matching each declared dtype so multi-input models (e.g. BERT's
    # input_ids + attention_mask, both int64) don't get fed float noise.
    loaded_file_input = np.load(args.input_file) if args.input_file else None
    used_file_for = None
    inputs = {}
    for inp in graph.input:
        elem_type = inp.type.tensor_type.elem_type
        if elem_type == onnx.TensorProto.INT64:
            if "mask" in inp.name.lower():
                # Attention/padding masks are 0/1-valued in real usage; random
                # ints across a wide range (e.g. 0-99) feed garbage into
                # (1-mask)*-inf-style masking arithmetic and can drive softmax
                # inputs to extreme magnitudes unrelated to any real kernel
                # bug -- keep this dtype-correct but semantically realistic.
                inputs[inp.name] = rng.integers(0, 2, size=in_shape, dtype=np.int64)
            else:
                inputs[inp.name] = rng.integers(0, 100, size=in_shape, dtype=np.int64)
        elif loaded_file_input is not None and used_file_for is None:
            inputs[inp.name] = loaded_file_input.astype(np.float32)
            used_file_for = inp.name
        else:
            inputs[inp.name] = rng.standard_normal(in_shape, dtype=np.float32)

    so_unfused = ort.SessionOptions()
    so_unfused.add_session_config_entry("optimization.disable_specified_optimizers", "ConvActivationFusion")

    rows = []
    t0 = time.time()
    for i, node in enumerate(candidates):
        tap = node.output[0]
        sub_path = os.path.join(args.tmp_dir, f"bisect_{model_name}_{i}.onnx")
        try:
            onnx.utils.extract_model(args.model, sub_path, input_names=in_names, output_names=[tap])
        except Exception as e:
            rows.append([i, node.name, node.op_type, "", "", f"EXTRACT_FAILED: {e}"])
            print(f"{i:4d}/{len(candidates)} {node.op_type:20s} {node.name:30s} EXTRACT_FAILED", flush=True)
            continue

        try:
            cpu_sess = ort.InferenceSession(sub_path, providers=["CPUExecutionProvider"])
            fused_sess = ort.InferenceSession(sub_path, providers=["ROCMExecutionProvider", "CPUExecutionProvider"])
            unfused_sess = ort.InferenceSession(
                sub_path, sess_options=so_unfused, providers=["ROCMExecutionProvider", "CPUExecutionProvider"]
            )
            cpu_out = cpu_sess.run([tap], inputs)[0]
            fused_out = fused_sess.run([tap], inputs)[0]
            unfused_out = unfused_sess.run([tap], inputs)[0]
            c_fused = cos_sim(cpu_out, fused_out)
            c_unfused = cos_sim(cpu_out, unfused_out)
            flag = "BAD" if c_fused < args.threshold else ""
            rows.append([i, node.name, node.op_type, f"{c_fused:.4f}", f"{c_unfused:.4f}", flag])
            marker = "  <-- BAD" if flag else ""
            print(
                f"{i:4d}/{len(candidates)} {node.op_type:20s} {node.name:30s} "
                f"cos_fused={c_fused:.4f} cos_unfused={c_unfused:.4f}{marker}",
                flush=True,
            )
        except Exception as e:
            rows.append([i, node.name, node.op_type, "", "", f"RUN_FAILED: {e}"])
            print(f"{i:4d}/{len(candidates)} {node.op_type:20s} {node.name:30s} RUN_FAILED: {e}", flush=True)
        finally:
            if os.path.exists(sub_path):
                os.remove(sub_path)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_idx", "node_name", "op_type", "cos_fused", "cos_unfused", "flag"])
        w.writerows(rows)

    bad = [r for r in rows if r[5] == "BAD"]
    elapsed = time.time() - t0
    print(f"\nSUMMARY: {len(rows)} nodes scanned in {elapsed:.0f}s, {len(bad)} flagged BAD, results: {csv_path}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
