"""A tiny deterministic checkpoint for trying the complete workflow offline."""

import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .archive import pack, verify


def write_example(directory):
    directory = Path(directory)
    rng = np.random.default_rng(27)
    rows, groups = 128, 32
    key = rng.integers(0, 16, size=(rows, groups))
    scale = np.linspace(0.01, 0.16, 16, dtype=np.float16)
    bias = np.linspace(-0.15, 0.0, 16, dtype=np.float16)
    weights = rng.integers(0, 2**32, size=(rows, groups * 8), dtype=np.uint32)
    save_file(
        {"layer.weight": weights, "layer.scales": scale[key], "layer.biases": bias[key]},
        directory / "model.safetensors",
    )


def demo():
    with tempfile.TemporaryDirectory(prefix="weightpack-demo-") as directory:
        root = Path(directory)
        write_example(root)
        archive = root / "demo.wpack"
        return {**pack(root, archive), **verify(root, archive)}
