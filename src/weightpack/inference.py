"""Local MLX-LM generation and paired projection timings on Apple Silicon."""

import json
import math
import random
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from .archive import Archive
from .checkpoint import Checkpoint
from .runtime import PackedEmbedding, PackedLinear


def source_bits(array):
    return np.asarray(array.view(mx.uint16)).astype("<u2", copy=False)


def module_slot(model, path):
    parts = path.split(".")
    owner = model
    for part in parts[:-1]:
        owner = owner[int(part)] if isinstance(owner, list) else owner[part]
    key = int(parts[-1]) if isinstance(owner, list) else parts[-1]
    return owner, key


def apply_archive(model, sidecar):
    with Archive(sidecar) as archive:
        paths = {name for name, _ in model.named_modules()}
        for entry in archive.entries.values():
            name = entry.name
            if name not in paths and getattr(model, "model_type", "") == "qwen3_5":
                name = (
                    "language_model.model." + name.removeprefix("model.language_model.")
                    if name.startswith("model.language_model.")
                    else "language_model." + name
                )
            if name not in paths:
                raise ValueError(f"{entry.name}: model loader has no matching module")
            owner, key = module_slot(model, name)
            original = owner[key]
            if (
                type(original) not in (nn.QuantizedLinear, nn.QuantizedEmbedding)
                or original.mode != "affine"
            ):
                raise ValueError(f"{entry.name}: expected an MLX affine linear or embedding module")
            dtype = {"F16": mx.float16, "BF16": mx.bfloat16}[entry.dtype]
            if (
                original.scales.dtype != dtype
                or original.biases.dtype != dtype
                or original.weight.shape != entry.weight_shape
            ):
                raise ValueError(f"{entry.name}: the loaded matrix has a different layout")
            table, words = archive.checked_arrays(
                entry, source_bits(original.scales), source_bits(original.biases)
            )
            layer = PackedEmbedding if type(original) is nn.QuantizedEmbedding else PackedLinear
            packed = layer(
                original.weight,
                table,
                words,
                entry.shape,
                dtype,
                original.group_size,
                original.bits,
                original.get("bias"),
            )
            if layer is PackedEmbedding:
                packed.num_embeddings, packed.dims = original.num_embeddings, original.dims
            owner[key] = packed
    mx.clear_cache()


def load(model_path, sidecar=None):
    """Load a local text checkpoint and optionally install its verified companion."""
    from mlx_lm import load as load_mlx

    root = Path(model_path).resolve()
    if not (root / "config.json").is_file():
        raise ValueError(f"Expected a local MLX-LM model with config.json: {root}")
    model, tokenizer = load_mlx(str(root), lazy=True)
    if sidecar is not None:
        apply_archive(model, sidecar)
    return model, tokenizer


def generate_text(model_path, sidecar, prompt, max_tokens, raw):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    if max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    model, tokenizer = load(model_path, sidecar)
    if not raw and tokenizer.has_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
    for output in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0)
    ):
        print(output.text, end="", flush=True)
    print()


def read_tensor(tensor, rows=None):
    dtype = {"U32": "<u4", "F16": "<f2", "BF16": "<u2", "F32": "<f4"}[tensor.dtype]
    array = np.memmap(tensor.path, dtype=dtype, mode="r", offset=tensor.offset, shape=tensor.shape)
    if rows is not None:
        array = array[rows]
    result = mx.array(np.array(array))
    return result.view(mx.bfloat16) if tensor.dtype == "BF16" else result


def benchmark(source, sidecar, name, rows, repeats):
    if min(rows) <= 0 or repeats <= 0 or len(set(rows)) != len(rows):
        raise ValueError("Use distinct positive row counts and a positive repeat count")
    source = Path(source)
    config = (source if source.is_dir() else source.parent) / "config.json"
    quant = json.loads(config.read_text()).get("quantization")
    if not isinstance(quant, dict):
        raise TypeError("bench requires an MLX affine quantization configuration in config.json")
    checkpoint = Checkpoint(source)
    with Archive(sidecar) as archive:
        if name is None:
            name = next(
                (n for n in archive.entries if n.endswith("mlp.gate_proj")),
                next(iter(archive.entries)),
            )
        if name not in archive.entries:
            raise ValueError(f"Matrix is absent from this pack: {name}")
        entry = archive.entries[name]
        matrix = next((m for m in checkpoint.matrices() if m.name == name), None)
        if (
            matrix is None
            or matrix.scales.dtype != entry.dtype
            or matrix.weight.shape != entry.weight_shape
        ):
            raise ValueError(f"{name}: matrix layout differs")
        table, words = archive.checked_arrays(entry, matrix.scales.raw16(), matrix.biases.raw16())
    quant = {**quant, **quant.get(name, {})}
    if quant.get("mode", "affine") != "affine":
        raise ValueError("Expected an affine matrix")
    group, bits = quant["group_size"], quant["bits"]
    w, s, b = [read_tensor(t) for t in (matrix.weight, matrix.scales, matrix.biases)]
    packed = PackedLinear(w, table, words, entry.shape, s.dtype, group, bits)
    mx.eval(w, s, b, packed.ids, packed.table)
    rng = random.Random(27)
    mx.random.seed(27)
    timings = []
    for count in rows:
        x = mx.random.normal((count, entry.shape[1] * group)).astype(s.dtype)
        mx.eval(x)

        def stock(x):
            return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group, bits=bits)

        arms = [("stock", stock), ("packed", packed)]
        times = {key: [] for key, _ in arms}
        for repeat in range(repeats + 2):
            rng.shuffle(arms)
            outputs = {}
            for key, run in arms:
                start = time.perf_counter_ns()
                outputs[key] = run(x)
                mx.eval(outputs[key])
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if repeat >= 2:
                    times[key].append(elapsed)
            if not np.array_equal(source_bits(outputs["stock"]), source_bits(outputs["packed"])):
                raise ValueError(f"{name}: output bits differ at {count} rows")
        timings.append(
            {"rows": count, **{k + "_ms": statistics.median(v) for k, v in times.items()}}
        )
    return {
        "matrix": name,
        "shape": [w.shape[0], math.prod(entry.shape[1:]) * group],
        "repeats": repeats,
        "bit_identical": True,
        "timings": timings,
    }
