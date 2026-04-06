#ifndef PITORCH_MATVEC_H
#define PITORCH_MATVEC_H

#include <stdint.h>
#include "profiler.h"

static inline uint32_t matvec_f2u(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}

/*
 * GPU matvec: y = W @ x.
 * W is [out_dim, in_dim] row-major, in ARM memory (TMU reads via bus alias).
 * out_dim must be a multiple of 16.
 * num_qpus: 1..12.
 * Requires gpu_arena_init() before first call; caller should gpu_arena_reset()
 * between calls to reclaim arena space.
 */
void smatvec_tmu(const float *W, const float *x, float *y,
                 int out_dim, int in_dim, int num_qpus, perf_t *perf);

#endif
