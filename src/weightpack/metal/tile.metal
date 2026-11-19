// Each 64-wide K tile spans exactly one affine group per output row.
constexpr short BK = 64, BN = 32, WM = 1, WN = 4, LD = BK + 8;
constexpr short threads = WM * WN * 32;
constexpr short reads = BN * (BK / 2) / threads;
using Loader = mlx::steel::BlockLoader<half, BM, BK, LD, 1, threads>;
using MMA = mlx::steel::BlockMMA<half, half, BM, BN, BK, WM, WN, false, true, LD, LD>;

const uint K = x_shape[1], N = w_shape[0], M = x_shape[0];
const uint col0 = threadgroup_position_in_grid.x * BN;
const uint row0 = threadgroup_position_in_grid.y * BM;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint first_byte = reads * thread_index_in_threadgroup;
const uint w_row = first_byte / (BK / 2);
const uint w_byte = first_byte % (BK / 2);
const uint groups = K / BK;
const short valid_rows = min(uint(BM), M - row0);

threadgroup half xs[BM * LD];
threadgroup half ws[BN * LD];
Loader xload(x + row0 * K, K, xs, simd, lane);
MMA accum(simd, lane);

const device uchar* weights = (const device uchar*) w +
    (col0 + w_row) * (K / 2) + w_byte;
threadgroup half* dest = ws + w_row * LD + w_byte * 2;
uint group = (col0 + w_row) * groups;

for (uint k = 0; k < groups; ++k, ++group, weights += BK / 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid_rows == BM) xload.load_unsafe();
    else xload.load_safe(short2(BK, valid_rows));
    const uint bit = group * ID_BITS;
    const uint word = bit >> 5;
    const uint shift = bit & 31;
    const uint lo = ids[word] >> shift;
    const uint hi = (ids[word + 1] << 1) << (31 - shift);
    const uint idx = (lo | hi) & ((1u << ID_BITS) - 1);
    const uint pair = table[idx];
    const float scale = float(as_type<half>(ushort(pair >> 16)));
    const float bias = float(as_type<half>(ushort(pair)));

    for (short i = 0; i < reads; ++i) {
        uchar q = weights[i];
        dest[2 * i] = half(scale * (q & 0x0f) + bias);
        dest[2 * i + 1] = half((scale / 16.0f) * (q & 0xf0) + bias);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    accum.mma(xs, ws);
    xload.next();
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (valid_rows == BM) accum.store_result(y + row0 * N + col0, N);
else accum.store_result_safe(y + row0 * N + col0, N, short2(BN, valid_rows));
