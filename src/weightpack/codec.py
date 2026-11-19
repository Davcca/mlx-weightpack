"""The on-disk format uses little-endian words and a zero sentinel per index stream."""

import hashlib

import numpy as np

CHUNK = 65536  # A multiple of 32 groups keeps every full chunk on a word boundary.
MAX_ENTRIES = 1 << 16


def make_pairs(scales, biases):
    return (scales.astype("<u4") << 16) | biases


def table_for(scales, biases):
    """Build a sorted palette using bounded temporary storage."""
    flat_s, flat_b = scales.reshape(-1), biases.reshape(-1)
    table = np.empty(0, dtype="<u4")
    for start in range(0, flat_s.size, CHUNK):
        stop = start + CHUNK
        pairs = make_pairs(flat_s[start:stop], flat_b[start:stop])
        table = np.union1d(table, np.unique(pairs))
        if table.size > MAX_ENTRIES:
            return None
    return table.astype("<u4", copy=False)


def width_for(entries):
    return max(1, (int(entries) - 1).bit_length())


def stream_words(groups, width):
    return (int(groups) * width + 31) // 32 + 1


def encode(scales, biases, table):
    scales, biases = scales.reshape(-1), biases.reshape(-1)
    width = width_for(table.size)
    words = np.zeros(stream_words(scales.size, width), dtype="<u4")
    raw = words.view("u1")
    shifts = np.arange(width, dtype=np.uint16)
    for start in range(0, scales.size, CHUNK):
        pairs = make_pairs(scales[start : start + CHUNK], biases[start : start + CHUNK])
        indices = np.searchsorted(table, pairs)
        if np.any(indices == table.size) or not np.array_equal(table[indices], pairs):
            raise ValueError("Metadata changed while building its palette")
        indices = indices.astype(np.uint16)
        chunk = np.packbits((indices[:, None] >> shifts) & 1, bitorder="little")
        byte = start * width // 8
        raw[byte : byte + chunk.size] = chunk
    return words


def decode_ids(words, width, start, count):
    pos = np.arange(start, start + count, dtype=np.uint64) * width
    idx = (pos >> 5).astype(np.intp)
    shift = (pos & 31).astype(np.uint32)
    lo = words[idx] >> shift
    hi = (words[idx + 1] << 1) << (31 - shift)
    return (lo | hi) & ((1 << width) - 1)


def check_pairs(scales, biases, table, words):
    scales, biases = scales.reshape(-1), biases.reshape(-1)
    width = width_for(table.size)
    if words.size != stream_words(scales.size, width) or words[-1] != 0:
        raise ValueError("Invalid index stream length or nonzero sentinel")
    for start in range(0, scales.size, CHUNK):
        stop = min(scales.size, start + CHUNK)
        ids = decode_ids(words, width, start, stop - start)
        if ids.max() >= table.size:
            raise ValueError(f"Index outside the palette at groups {start}..{stop - 1}")
        expected = make_pairs(scales[start:stop], biases[start:stop])
        if not np.array_equal(table[ids], expected):
            raise ValueError(f"Reconstructed scale or bias differs at groups {start}..{stop - 1}")


def digest(*arrays):
    result = hashlib.sha256()
    for a in arrays:
        result.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return result.hexdigest()
