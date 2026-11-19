template <int BITS>
inline uint read_id(const device uint* words, uint group) {
    const uint bit = (group & 31u) * BITS;
    const uint word = (group >> 5) * BITS + (bit >> 5);
    const uint shift = bit & 31u;
    const uint lo = words[word] >> shift;
    const uint hi = (words[word + 1] << 1) << (31u - shift);
    return (lo | hi) & ((1u << BITS) - 1);
}
