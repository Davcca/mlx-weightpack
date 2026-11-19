"""Compare greedy token IDs through MLX-LM, loading one model at a time."""

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from weightpack.inference import load


def sample(model_dir, sidecar, prompt, count):
    model, tok = load(model_dir, sidecar)
    ids = (
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
        if tok.has_chat_template
        else tok.encode(prompt)
    )
    tokens = [
        part.token
        for part in stream_generate(model, tok, ids, max_tokens=count, sampler=make_sampler(temp=0))
    ]
    return tokens


def compare(model_dir, sidecar, prompt, count):
    if count < 1:
        raise ValueError("--max-tokens must be positive")
    stock = sample(model_dir, None, prompt, count)
    gc.collect()
    mx.clear_cache()
    packed = sample(model_dir, sidecar, prompt, count)
    if not stock or stock != packed:
        raise ValueError(f"Greedy continuation differs: stock={stock}, packed={packed}")
    return {
        "prompt": prompt,
        "max_tokens": count,
        "compared_tokens": len(stock),
        "greedy_token_ids": stock,
        "identical": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Write a Python function that squares a number.")
    args = parser.parse_args()
    print(json.dumps(compare(args.model, args.sidecar, args.prompt, args.max_tokens), indent=2))
