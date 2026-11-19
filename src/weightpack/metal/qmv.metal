const uint K = x_shape[1], N = weight_shape[0], groups = K / 64;
const uint lane = thread_index_in_simdgroup;
const uint row_start = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup * 4;
const uint m = threadgroup_position_in_grid.x;
const device uchar* weights = (const device uchar*) weight;
float accum[4] = {0.0};
for (uint k = 0; k < K; k += 512) {
    float values[16];
    float sum = load_vector4_exact<half>(x + m * K + k + lane * 16, values);
    for (uint r = 0; r < 4; ++r) {
        uint row = row_start + r;
        uint group = row * groups + k / 64 + lane / 4;
        uint pair = table[read_id<BITS>(ids, group)];
        float scale = float(as_type<half>(ushort(pair >> 16)));
        float bias = float(as_type<half>(ushort(pair)));
        accum[r] += qdot4_exact(weights + row * (K / 2) + k / 2 + lane * 8,
                                values, scale, bias, sum);
    }
}
for (uint r = 0; r < 4; ++r) {
    float val = simd_sum(accum[r]);
    if (lane == 0) out[m * N + row_start + r] = half(val);
}
