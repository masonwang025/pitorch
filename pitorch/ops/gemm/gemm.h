#ifndef PITORCH_GEMM_H
#define PITORCH_GEMM_H

#include <stdint.h>
#include "profiler.h"

static inline uint32_t f2u(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}

static inline float u2f(uint32_t u) {
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

void sgemm_cpu(const float *A, const float *B, float *C,
               int M, int K, int N);

/* TMU GEMM. Square (dim limited by uniform stream size).
 * num_qpus = 1..12. Host tiles N in 16-col chunks. */
void sgemm_tmu(const float *A, const float *B, float *C,
               int M, int K, int N, int num_qpus, perf_t *perf);

#endif
