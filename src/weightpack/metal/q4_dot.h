// Copyright 2026 Youssof Altoukhi. SPDX-License-Identifier: Apache-2.0
// Extracted from MTPLX 2.11.3, mtplx/kernels/qmv4_matmul.py.
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int PACK_FACTOR = 8;
constant constexpr int PACKS_PER_THREAD = 2;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BYTES_PER_PACK = 4;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int NUM_SIMDGROUPS = 2;
constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
constant constexpr int MAX_M = 6;

template <typename T>
inline float load_vector4_exact(const device T* x, thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

inline float qdot4_exact(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
    uint16_t packed = ws[i];
    accum +=
      x_thread[4 * i] * float(packed & 0x000f) +
      x_thread[4 * i + 1] * float(packed & 0x00f0) +
      x_thread[4 * i + 2] * float(packed & 0x0f00) +
      x_thread[4 * i + 3] * float(packed & 0xf000);
  }
  return scale * accum + sum * bias;
}
