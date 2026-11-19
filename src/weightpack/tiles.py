"""Run a group-64 packed Q4 projection with small row tiles on Apple GPUs."""

import re
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def measured_runtime():
    from importlib.metadata import version

    return version("mlx") == "0.32.2" and mx.device_info()["device_name"] == "Apple M1 Max"


def tile_height(rows, output_dim, input_dim):
    # Stock MLX chooses a different reduction on smaller projections. Restrict
    # the packed path to the shapes that matched the stock reduction bitwise.
    if not (7 <= rows <= 16) or input_dim != 5120 or output_dim not in (10240, 12288, 17408):
        return None
    if not measured_runtime():
        return None
    return 8 if rows <= 8 else 16


@lru_cache(maxsize=1)
def steel_header():
    # Use the Steel helpers supplied by the installed, pinned MLX wheel. Its
    # inline Metal compiler does not resolve filesystem include directories.
    include = Path(mx.__file__).parent / "include"
    emitted = set()

    def inline(name):
        if name in emitted:
            return ""
        emitted.add(name)
        source = (include / name).read_text().replace("#pragma once", "")
        return re.sub(
            r'^#include "([^"]+)"',
            lambda match: inline(match[1]),
            source,
            flags=re.MULTILINE,
        )

    prefix = "mlx/backend/metal/kernels/steel/gemm/"
    return inline(prefix + "loader.h") + "\n" + inline(prefix + "mma.h")


@lru_cache(maxsize=1)
def packed_kernel():
    return mx.fast.metal_kernel(
        name="packed_affine_q4_steel_tile",
        input_names=["x", "w", "ids", "table"],
        output_names=["y"],
        header=steel_header(),
        source=files("weightpack").joinpath("metal", "tile.metal").read_text(),
    )


def project(x, packed, tile_rows):
    m, k = x.shape
    n = packed.weight.shape[0]
    if k % 64 or n % 32 or x.dtype != mx.float16:
        raise ValueError(
            "This kernel requires FP16 input, K divisible by 64, and N divisible by 32"
        )
    if tile_rows != tile_height(m, n, k) or tile_rows not in (8, 16):
        raise ValueError("No exact packed tile was selected for this shape and runtime")
    threads = 128
    (result,) = packed_kernel()(
        inputs=[x, packed.weight, packed.ids, packed.table],
        template=[("BM", tile_rows), ("ID_BITS", packed.width)],
        grid=(threads * (n // 32), (m + tile_rows - 1) // tile_rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )
    return result
