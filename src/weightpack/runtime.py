"""Affine MLX layers that keep their tables and packed indices on the device."""

import math
from functools import lru_cache
from importlib.resources import files

import mlx.core as mx
from mlx import nn

from .codec import width_for
from .tiles import measured_runtime, project, tile_height


def metal_file(name):
    return files("weightpack").joinpath("metal", name).read_text()


@lru_cache(maxsize=1)
def expand_kernel():
    return mx.fast.metal_kernel(
        name="weightpack_expand_metadata",
        input_names=["ids", "table"],
        output_names=["scales", "biases"],
        header=metal_file("read_id.h"),
        source="""
            const uint i = thread_position_in_grid.x;
            const uint pair = table[read_id<BITS>(ids, i)];
            scales[i] = ushort(pair >> 16);
            biases[i] = ushort(pair);
        """,
    )


@lru_cache(maxsize=1)
def qmv_kernel():
    return mx.fast.metal_kernel(
        name="weightpack_single_row_q4",
        input_names=["x", "weight", "ids", "table"],
        output_names=["out"],
        header=metal_file("q4_dot.h") + metal_file("read_id.h"),
        source=metal_file("qmv.metal"),
    )


class PackedLinear(nn.Module):
    def __init__(self, weight, table, words, shape, dtype, group_size, bits, bias=None):
        super().__init__()
        self.weight, self.table, self.ids = weight, mx.array(table), mx.array(words)
        self.width = width_for(self.table.size)
        self.shape, self.meta_dtype = tuple(shape), dtype
        self.group_size, self.bits, self.mode = group_size, bits, "affine"
        if bias is not None:
            self.bias = bias
        if (
            math.prod(self.shape) > 2**32 - 1
            or self.shape[0] != weight.shape[0]
            or self.shape[1] * group_size * bits != weight.shape[1] * 32
        ):
            raise ValueError("Matrix dimensions do not match its quantization configuration")
        self.freeze()

    def metadata(self):
        scales, biases = expand_kernel()(
            inputs=[self.ids, self.table],
            template=[("BITS", self.width)],
            grid=(math.prod(self.shape), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[self.shape] * 2,
            output_dtypes=[mx.uint16] * 2,
        )
        return scales.view(self.meta_dtype), biases.view(self.meta_dtype)

    def as_linear(self, x):
        input_dim = self.shape[1] * self.group_size
        if x.ndim == 0 or x.shape[-1] != input_dim:
            raise ValueError(f"Expected a last input dimension of {input_dim}, received {x.shape}")
        rows = math.prod(x.shape[:-1])
        n, k = self.shape[0], x.shape[-1]
        if self.bits == 4 and self.group_size == 64 and self.meta_dtype == x.dtype == mx.float16:
            tile = tile_height(rows, n, k)
            if tile:
                y = project(x.reshape(rows, k), self, tile)
                return y.reshape(*x.shape[:-1], n)
            if rows == 1 and (n, k) == (17408, 5120) and measured_runtime():
                (y,) = qmv_kernel()(
                    inputs=[x.reshape(rows, k), self.weight, self.ids, self.table],
                    template=[("BITS", self.width)],
                    grid=(32, 2 * (n // 8), 1),
                    threadgroup=(32, 2, 1),
                    output_shapes=[(1, n)],
                    output_dtypes=[mx.float16],
                )
                return y.reshape(*x.shape[:-1], n)
        scales, biases = self.metadata()
        return mx.quantized_matmul(
            x,
            self.weight,
            scales,
            biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def __call__(self, x):
        y = self.as_linear(x)
        return y + self.bias if "bias" in self else y


class PackedEmbedding(PackedLinear):
    def __call__(self, x):
        groups = self.shape[1]
        # Only fetch metadata for requested vocabulary rows.
        tokens = x.astype(mx.uint32)
        if mx.issubdtype(x.dtype, mx.signedinteger):
            tokens = mx.where(x < 0, tokens + self.shape[0], tokens)
        rows = tokens[..., None] * groups + mx.arange(groups, dtype=mx.uint32)
        bit = (rows & 31) * self.width
        word = (rows >> 5) * self.width + (bit >> 5)
        shift = bit & 31
        lo = self.ids[word] >> shift
        hi = (self.ids[word + 1] << 1) << (31 - shift)
        pairs = self.table[(lo | hi) & ((1 << self.width) - 1)]
        scales = (pairs >> 16).astype(mx.uint16).view(self.meta_dtype)
        biases = pairs.astype(mx.uint16).view(self.meta_dtype)
        return mx.dequantize(
            self.weight[x], scales, biases, group_size=self.group_size, bits=self.bits
        )
