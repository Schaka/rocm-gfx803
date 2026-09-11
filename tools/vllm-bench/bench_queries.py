#!/usr/bin/env python3
"""Multi-query serving benchmark for the gfx803 vLLM stack.

Measures steady-state prefill and decode throughput over a set of queries,
after warm-up, across a range of request concurrency. Time-to-first-token is
reported but is not the target metric.

Method: every configuration is run twice, once with `--decode 1` and once with
the real decode length, and decode throughput is the difference of the two.
Prefill and engine overhead therefore cancel, so a change that only speeds up
prefill cannot be reported as a decode win (and the reverse). Both runs use
greedy sampling with `ignore_eos` and a forced token count, so each request
generates exactly `--decode` tokens and run-to-run variance does not come from
sampled output length. Prefix caching is off: the request set reuses prompt
bodies, and a prefix cache hit would turn prefill into a no-op.

Prompts are distinct per request and covered in prose/code/table bodies, so the
workload has realistic variety instead of one repeated string.

Usage:
    python bench_queries.py --concurrency 1,4,8,16 --prompt-len 128 --decode 128
"""

import argparse
import json
import statistics
import time

from vllm import LLM, SamplingParams

_BODIES = [
    "Explain how a transformer language model turns a prompt into the next "
    "token. Cover tokenisation, the attention operation, and why the decode "
    "step re-reads every weight in the model even when it produces a single "
    "token. Be concrete about where the time goes on a memory-bandwidth-bound "
    "device with no matrix cores.",
    "def merge_intervals(intervals):\n"
    "    intervals.sort()\n"
    "    out = []\n"
    "    for start, end in intervals:\n"
    "        if out and start <= out[-1][1]:\n"
    "            out[-1][1] = max(out[-1][1], end)\n"
    "        else:\n"
    "            out.append([start, end])\n"
    "    return out\n\n"
    "Explain what this function does, then rewrite it so it does not mutate "
    "the list it is given and does not depend on the input being sorted.",
    "The following table lists measured throughput for a small language model "
    "on an older GPU. Column one is the batch size, column two is decode "
    "tokens per second, column three is prefill tokens per second:\n"
    "1 42 310\n2 71 402\n4 118 505\n8 166 585\n"
    "Summarise the trend and explain what limits the batch size column.",
    "Summarise the trade-offs between speculative decoding, quantisation, and "
    "batching for serving a small model on a device with roughly 200 GB/s of "
    "memory bandwidth and no matrix cores. Which of them is lossless, and "
    "which changes the output distribution?",
    "A storage array writes 4 KiB blocks; the workload issues 70 percent "
    "random reads and 30 percent sequential writes. Explain which device "
    "metric decides the throughput ceiling, and how write amplification "
    "changes the answer.",
    "Contexte : un service d'inference doit traiter des requetes courtes avec "
    "une latence faible. Decrivez comment le batching continu change le debit "
    "agrege, et pourquoi le temps jusqu'au premier token reste inchange.",
]


def _prompt_of_length(body: str, target_tokens: int, salt: int) -> str:
    """Grow `body` to roughly target_tokens (4 chars/token heuristic)."""
    target_chars = max(8, target_tokens * 4)
    text = f"[request {salt}] " + body
    while len(text) < target_chars:
        text = text + "\n\n" + body
    return text[:target_chars]


def build_prompts(concurrency: int, prompt_len: int) -> list[str]:
    return [
        _prompt_of_length(
            _BODIES[i % len(_BODIES)], prompt_len, salt=1000 + i
        )
        for i in range(concurrency)
    ]


def _timed_generate(llm: LLM, prompts: list[str], max_tokens: int):
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=True,
    )
    start = time.perf_counter()
    outs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return elapsed, out_tokens


def measure(llm: LLM, prompts: list[str], decode: int, repeats: int, warmup: int):
    for _ in range(warmup):
        _timed_generate(llm, prompts, 32)
    prefill_times, full_times = [], []
    for _ in range(repeats):
        t1, _ = _timed_generate(llm, prompts, 1)
        prefill_times.append(t1)
        t2, _ = _timed_generate(llm, prompts, decode)
        full_times.append(t2)
    return statistics.median(prefill_times), statistics.median(full_times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-0.6B")
    ap.add_argument("--concurrency", default="1,4,8,16")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--prefix-caching", action="store_true")
    args = ap.parse_args()

    lvls = [int(v) for v in args.concurrency.split(",")]
    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=max(args.max_num_seqs, max(lvls)),
        enable_prefix_caching=args.prefix_caching,
        disable_log_stats=True,
        enforce_eager=args.enforce_eager,
    )

    results = []
    for conc in lvls:
        prompts = build_prompts(conc, args.prompt_len)
        prefill, full = measure(
            llm, prompts, args.decode, args.repeats, args.warmup
        )
        decode_seconds = max(full - prefill, 1e-9)
        decode_tps = (conc * args.decode) / decode_seconds
        prefill_tps = (conc * args.prompt_len) / max(prefill, 1e-9)
        row = {
            "label": args.label,
            "concurrency": conc,
            "prompt_len": args.prompt_len,
            "decode": args.decode,
            "prefill_s_median": round(prefill, 4),
            "full_s_median": round(full, 4),
            "decode_tokens_per_s": round(decode_tps, 1),
            "prefill_tokens_per_s": round(prefill_tps, 1),
            "per_request_decode_tps": round(decode_tps / conc, 1),
            "max_mem_allocated_gb": round(
                __import__("torch").cuda.max_memory_allocated() / 1e9, 2
            ),
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    print("\n=== summary ===")
    print(f"{'conc':>5} {'prefill tok/s':>14} {'decode tok/s':>13} {'per-req tok/s':>14}")
    for row in results:
        print(
            f"{row['concurrency']:>5} {row['prefill_tokens_per_s']:>14.1f} "
            f"{row['decode_tokens_per_s']:>13.1f} "
            f"{row['per_request_decode_tps']:>14.1f}"
        )
    if args.json_out:
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps({"args": vars(args), "results": results}) + "\n")


if __name__ == "__main__":
    main()
