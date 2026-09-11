#!/usr/bin/env python3
"""Cross-check decode at M > 1 by generating greedily with and without the
gfx803 multi-token GEMV path, at several concurrencies.

The switch is read when vLLM imports the loader, so the two configurations
necessarily run in separate processes: run this once with
VLLM_GFX803_GEMV_M=0 and once with VLLM_GFX803_GEMV_M=1, then compare the
token id lines. Greedy sampling with ignore_eos and a forced length keeps both
runs at the same token count, so a difference is a numerical difference in the
path under test rather than sampling noise.

A concurrency of 1 exercises nothing: decode is M=1 there and takes the
single-token GEMV. It is in the list as a control, because if the two
configurations disagree at concurrency 1 then something other than this
kernel differs between them.

Usage:
    VLLM_GFX803_GEMV_M=0 python verify_gemv_m.py --concurrency 1,8 --decode 32
    VLLM_GFX803_GEMV_M=1 python verify_gemv_m.py --concurrency 1,8 --decode 32
"""

import argparse
import os

from vllm import LLM, SamplingParams

_PROMPTS = [
    "List the first eight prime numbers, one per line, and nothing else.",
    "Write a haiku about memory bandwidth. Then explain each line.",
    "What is 17 * 23? Show the arithmetic, then state the answer alone.",
    "Name three causes of a wrong answer from a matrix multiply on a GPU.",
    "Summarise what a decode step costs on a device without matrix cores.",
    "Convert 0.375 to a fraction in lowest terms and show the steps.",
    "Explain why a 64-row tile wastes work when the batch is 8 rows.",
    "Give the exact command that compiles a HIP kernel for gfx803.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--concurrency", default="1,8")
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        max_num_batched_tokens=2048,
        max_num_seqs=32,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
    )
    print(f"VLLM_GFX803_GEMV_M={os.environ.get('VLLM_GFX803_GEMV_M', '1')}")

    for conc in (int(c) for c in args.concurrency.split(",")):
        sp = SamplingParams(
            temperature=0.0, max_tokens=args.decode, ignore_eos=True
        )
        outs = llm.generate(_PROMPTS[:conc], sp)
        ids = [list(o.outputs[0].token_ids) for o in outs]
        print(f"conc={conc} first_ids={[i[:8] for i in ids]}")
        print(f"conc={conc} joined={[i for row in ids for i in row]}")


if __name__ == "__main__":
    main()
