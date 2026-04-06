#ifndef PITORCH_MATVEC_NO_PACKING_H
#define PITORCH_MATVEC_NO_PACKING_H

#include "profiler.h"
#include "matvec.h"   /* matvec_f2u, smatvec_fn_t */

/*
 * GPU matvec without the uniform-packing optimization (pre-wr08 version).
 * Repacks the full x vector into every QPU's uniform buffer on every dispatch.
 * Same signature as smatvec_tmu — drop-in replacement for benchmarking.
 */
void smatvec_tmu_no_packing(const float *W, const float *x, float *y,
                             int out_dim, int in_dim, int num_qpus, perf_t *perf);

#endif
