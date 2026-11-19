import unittest

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipUnless(mx is not None and mx.metal.is_available(), "requires an Apple Metal GPU")
class MetalTests(unittest.TestCase):
    def make_layer(self, layer_type, dtype, add_bias=False):
        from weightpack.codec import encode, table_for

        rng = np.random.default_rng(13)
        groups, rows = 3, 64
        codes = mx.array(rng.integers(0, 2**32, (rows, groups * 8), dtype=np.uint32))
        # Seventeen distinct pairs give a five-bit index; rows span word boundaries.
        indices = mx.array(rng.integers(0, 17, (rows, groups)))
        s = ((mx.arange(17) + 1) * 0.03125).astype(dtype)[indices]
        b = (-mx.arange(17) * 0.0625).astype(dtype)[indices]
        rs, rb = np.asarray(s.view(mx.uint16)), np.asarray(b.view(mx.uint16))
        table = table_for(rs, rb)
        bias = mx.arange(rows).astype(dtype) if add_bias else None
        layer = layer_type(codes, table, encode(rs, rb, table), s.shape, dtype, 64, 4, bias)
        return layer, codes, s, b

    def assertSameBits(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        np.testing.assert_array_equal(np.asarray(a.view(mx.uint8)), np.asarray(b.view(mx.uint8)))

    def test_linear_fallback_and_bias(self):
        from weightpack.runtime import PackedLinear

        for dtype in (mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                layer, w, s, b = self.make_layer(PackedLinear, dtype, True)
                unpacked = layer.metadata()
                self.assertSameBits(unpacked[0], s)
                self.assertSameBits(unpacked[1], b)
                x = mx.ones((2, 3, 192), dtype=dtype)
                self.assertSameBits(
                    layer(x),
                    mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
                    + layer.bias,
                )

    def test_embedding_gather_and_tied_projection(self):
        from weightpack.runtime import PackedEmbedding

        for dtype in (mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                layer, w, s, b = self.make_layer(PackedEmbedding, dtype)
                ids = mx.array([[63, 1], [0, 63]])
                self.assertSameBits(
                    layer(ids), mx.dequantize(w[ids], s[ids], b[ids], group_size=64, bits=4)
                )
                x = mx.ones((5, 192), dtype=dtype)
                self.assertSameBits(
                    layer.as_linear(x),
                    mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4),
                )
