import unittest

import numpy as np

from weightpack.codec import (
    CHUNK,
    check_pairs,
    decode_ids,
    encode,
    stream_words,
    table_for,
    width_for,
)


class CodecTests(unittest.TestCase):
    def test_every_width_crosses_words_and_chunks(self):
        for width in range(1, 17):
            with self.subTest(width=width):
                count = 2**width
                rng = np.random.default_rng(width)
                indices = np.concatenate(
                    (np.arange(count), rng.integers(0, count, CHUNK + 3))
                ).astype("<u2")
                scales, biases = indices, indices ^ 0x8000
                table = table_for(scales, biases)
                self.assertEqual(width_for(table.size), width)
                packed = encode(scales, biases, table)
                self.assertEqual(packed.size, stream_words(indices.size, width))
                check_pairs(scales, biases, table, packed)
                np.testing.assert_array_equal(
                    decode_ids(packed, width, CHUNK - 2, 5), indices[CHUNK - 2 : CHUNK + 3]
                )

    def test_float_payloads_stay_distinct(self):
        raw = np.array([0, 0x8000, 0x7E01, 0x7E02, 0xFFFF, 0x7F80], dtype="<u2")
        scales = np.resize(raw, (2, 63))
        biases = scales[:, ::-1].copy()
        table = table_for(scales, biases)
        words = encode(scales, biases, table)
        check_pairs(scales, biases, table, words)
        words[-1] = 1
        with self.assertRaisesRegex(ValueError, "sentinel"):
            check_pairs(scales, biases, table, words)

    def test_large_dictionary_falls_back(self):
        pairs = np.arange(CHUNK + 1, dtype="<u4")
        self.assertIsNone(table_for((pairs >> 16).astype("<u2"), pairs.astype("<u2")))

    def test_single_entry_and_invalid_index(self):
        a = np.zeros(33, dtype="<u2")
        table = table_for(a, a)
        words = encode(a, a, table)
        self.assertEqual(width_for(1), 1)
        words[0] = 1
        with self.assertRaisesRegex(ValueError, "outside"):
            check_pairs(a, a, table, words)
