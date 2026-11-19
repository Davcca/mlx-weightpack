"""Common readers for the layer-at-a-time Qwen benchmark."""

import hashlib

import mlx.core as mx
import numpy as np

from weightpack.checkpoint import Checkpoint
from weightpack.inference import read_tensor, source_bits
from weightpack.runtime import PackedLinear


class Shards:
    def __init__(self, path):
        self.fields = Checkpoint(path).tensors

    def read(self, name, rows=None):
        return read_tensor(self.fields[name], rows)


def packed_layer(archive, name, stock):
    entry = archive.entries[name]
    table, words = archive.checked_arrays(
        entry, source_bits(stock.scales), source_bits(stock.biases)
    )
    return PackedLinear(
        stock.weight,
        table,
        words,
        entry.shape,
        stock.scales.dtype,
        stock.group_size,
        stock.bits,
        stock.get("bias"),
    )


def compare(stock, packed, label):
    mx.eval(stock, packed)
    first = np.asarray(stock.view(mx.uint8))
    second = np.asarray(packed.view(mx.uint8))
    if (
        stock.shape != packed.shape
        or stock.dtype != packed.dtype
        or not np.array_equal(first, second)
    ):
        raise ValueError(f"{label}: the two outputs differ")
    return hashlib.sha256(first.tobytes()).hexdigest()
