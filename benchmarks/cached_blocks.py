"""Compare cached target blocks layer by layer, with checkpoint I/O outside the timer."""

import argparse
import json
import random
import statistics
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
from support import Shards, compare, packed_layer

from weightpack.archive import Archive
from weightpack.inference import source_bits
from weightpack.runtime import PackedEmbedding, PackedLinear


def copies_for(layer, weights, path, archive):
    replacements = []
    for name, _ in weights:
        if not name.endswith(".scales"):
            continue
        *parts, field = name.removesuffix(".scales").split(".")
        owner = layer
        for part in parts:
            owner = getattr(owner, part)
        stock = getattr(owner, field)
        packed = packed_layer(archive, path + name.removesuffix(".scales"), stock)
        mx.eval(stock.weight, stock.scales, stock.biases, packed.ids, packed.table)
        replacements.append((owner, field, stock, packed))
    return replacements


def measure_pair(make_run, repeats, rng):
    order = ["stock", "packed"]
    timings = {key: [] for key in order}
    outputs = {}
    for repeat in range(repeats + 2):
        rng.shuffle(order)
        for key in order:
            run = make_run(key)
            start = time.perf_counter()
            results = run()
            mx.eval(*results)
            elapsed = 1000 * (time.perf_counter() - start)
            if repeat >= 2:
                timings[key].append(elapsed)
            outputs[key] = results
    medians = {key: statistics.median(values) for key, values in timings.items()}
    return outputs, medians, timings


def layer_run(mode, *, layer, replacements, cache, mask_for, x):
    for owner, field, original, packed in replacements:
        setattr(owner, field, original if mode == "stock" else packed)
    clone = type(cache).from_state(list(cache.state), cache.meta_state)
    mask = mask_for(x, clone)

    def forward():
        y = layer(x, mask=mask, cache=clone)
        return [y, *clone.state]

    return forward


def head_run(mode, *, x, norm, w, s, b, head):
    def forward():
        normalized = norm(x)
        if mode == "packed":
            return [head(normalized)]
        return [mx.quantized_matmul(normalized, w, s, b, group_size=64, bits=4, transpose=True)]

    return forward


def measure(root, archive, fixture, widths, repeats):
    ids, greedy = fixture["prefix_ids"], fixture["greedy_ids"]
    if repeats < 1 or not widths or min(widths) < 1 or len(set(widths)) != len(widths):
        raise ValueError("Use positive repetitions and distinct positive widths")
    if max(widths) > len(greedy):
        raise ValueError("The fixture has too few continuation tokens")
    config = json.loads((root / "config.json").read_text())
    args = TextModelArgs.from_dict(config["text_config"])
    shards = Shards(root)
    rng = random.Random(285)
    embedding_ids = ids + greedy[: max(widths)]
    embed = "language_model.model.embed_tokens."
    w, s, b = [
        shards.read(embed + field, rows=embedding_ids) for field in ("weight", "scales", "biases")
    ]
    hidden = mx.dequantize(w, s, b, group_size=64, bits=4)[None]
    entry = archive.entries[embed.rstrip(".")]
    table, words = archive.arrays(entry)
    packed_embed = PackedEmbedding(
        shards.read(embed + "weight"), table, words, entry.shape, s.dtype, 64, 4
    )
    compare(hidden, packed_embed(mx.array(embedding_ids))[None], "input embeddings")
    del packed_embed, w, s, b, table, words
    prefix = mx.contiguous(hidden[:, : len(ids)])
    blocks = {m: mx.contiguous(hidden[:, len(ids) : len(ids) + m]) for m in widths}
    mx.eval(prefix, *blocks.values())

    stages = []
    for i in range(args.num_hidden_layers):
        path = f"language_model.model.layers.{i}."
        layer = DecoderLayer(args, i)
        nn.quantize(
            layer,
            group_size=64,
            bits=4,
            class_predicate=lambda key, module, path=path: path + key + ".scales" in shards.fields,
        )
        weights = [
            (key.removeprefix(path), shards.read(key))
            for key in shards.fields
            if key.startswith(path)
        ]
        layer.load_weights(weights)
        cache = ArraysCache(size=2) if layer.is_linear else KVCache()
        mask_for = create_ssm_mask if layer.is_linear else create_attention_mask
        prefix = layer(prefix, mask=mask_for(prefix, cache), cache=cache)
        mx.eval(prefix, *cache.state)
        replacements = copies_for(layer, weights, path, archive)
        stage = {"layer": i, "block_ms": {}, "samples_ms": {}}
        for width in widths:
            x = blocks[width]
            prepare = partial(
                layer_run,
                layer=layer,
                replacements=replacements,
                cache=cache,
                mask_for=mask_for,
                x=x,
            )
            outputs, stage["block_ms"][str(width)], stage["samples_ms"][str(width)] = measure_pair(
                prepare, repeats, rng
            )
            for j, (stock, packed) in enumerate(
                zip(outputs["stock"], outputs["packed"], strict=True)
            ):
                compare(stock, packed, f"layer {i}, width {width}, tensor {j}")
            blocks[width] = outputs["stock"][0]
        stages.append(stage)
        if (i + 1) % 16 == 0:
            print(f"Matched {i + 1} layers", flush=True)
        del layer, weights, replacements, cache
        mx.clear_cache()

    norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    norm.weight = shards.read("language_model.model.norm.weight")
    w, s, b = [
        shards.read("language_model.lm_head." + field) for field in ("weight", "scales", "biases")
    ]
    entry = archive.entries["language_model.lm_head"]
    table, words = archive.checked_arrays(entry, source_bits(s), source_bits(b))
    head = PackedLinear(w, table, words, entry.shape, s.dtype, 64, 4)
    mx.eval(w, s, b, head.ids, head.table, norm.weight)
    hashes = {}
    stage = {"layer": "final_norm_and_head", "block_ms": {}, "samples_ms": {}}
    for width in widths:
        x = blocks[width]
        prepare_head = partial(head_run, x=x, norm=norm, w=w, s=s, b=b, head=head)
        outputs, stage["block_ms"][str(width)], stage["samples_ms"][str(width)] = measure_pair(
            prepare_head, repeats, rng
        )
        hashes[str(width)] = compare(
            *[outputs[mode][0] for mode in ("stock", "packed")],
            f"all logits at width {width}",
        )
    stages.append(stage)
    totals = {
        str(width): {
            mode: sum(stage["block_ms"][str(width)][mode] for stage in stages)
            for mode in ("stock", "packed")
        }
        for width in widths
    }
    return {
        "prefix_tokens": len(ids),
        "widths": widths,
        "timed_pairs_per_stage": repeats,
        "method": "Sum of per-stage paired median wall times, decoder layers plus final norm and head. Each layer starts from the same stock-prefilled cache. The timer includes the forward and evaluation of the hidden state and both updated cache arrays. Checkpoint I/O, cache cloning, model conversion, token selection and HTTP are outside the timer.",
        "layers": args.num_hidden_layers,
        "output_scores_compared": sum(widths) * args.vocab_size,
        "all_layer_outputs_caches_and_logits_bit_identical": True,
        "logits_sha256": hashes,
        "sum_of_stage_medians_ms": totals,
        "stages": stages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).parent / "fixtures/edit_tokens.json",
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    with Archive(opts.sidecar) as archive:
        report = measure(
            opts.model, archive, json.loads(opts.fixture.read_text()), opts.widths, opts.repeats
        )
    print(json.dumps({k: v for k, v in report.items() if k != "stages"}, indent=2))
    if opts.output:
        opts.output.write_text(json.dumps(report, indent=2) + "\n")
