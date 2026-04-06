#ifndef PITORCH_GEMM_RECT_H
#define PITORCH_GEMM_RECT_H

#include <stdint.h>

static inline uint32_t gemm_f2u(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}

/*
 * Rectangular GEMM on GPU (12 QPUs, arena-based).
 *
 *   C[M][N] += A[M][K] * B[K][N]   (if accumulate=1)
 *   C[M][N]  = A[M][K] * B[K][N]   (if accumulate=0)
 *
 * Constraints:
 *   - N must be a multiple of 16
 *   - B must be in ARM-accessible memory (read via TMU)
 *   - num_qpus: 1..12
 *   - Requires gpu_arena_init() before first call
 */
void sgemm_rect_tmu(const float *A, const float *B, float *C,
                    int M, int K, int N,
                    int num_qpus, int accumulate);

#endif
