"""Measure live MLX allocations for all affine metadata in a fresh process."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from weightpack.archive import Archive
from weightpack.checkpoint import Checkpoint
from weightpack.inference import read_tensor


def measure(root, mode, sidecar):
    checkpoint = Checkpoint(root)
    retained = []
    baseline = mx.get_active_memory()
    # Unpacked matrices still count against the packed configuration.
    with Archive(sidecar) as archive:
        for matrix in checkpoint.matrices():
            if mode == "packed" and (entry := archive.entries.get(matrix.name)):
                table, words = archive.checked_arrays(
                    entry, matrix.scales.raw16(), matrix.biases.raw16()
                )
                arrays = (mx.array(table), mx.array(words))
            else:
                arrays = (read_tensor(matrix.scales), read_tensor(matrix.biases))
            mx.eval(*arrays)
            retained.extend(arrays)
    mx.clear_cache()
    return {
        "mode": mode,
        "matrices": len(retained) // 2,
        "gpu_arrays": len(retained),
        "retained_tensor_bytes": sum(x.nbytes for x in retained),
        "active_mlx_bytes_above_start": mx.get_active_memory() - baseline,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("mode", choices=["stock", "packed"])
    args = parser.parse_args()
    print(json.dumps(measure(args.model, args.mode, args.sidecar), indent=2))
