#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "arena.h"
#include "matvec.h"
#include "matvec_tmu_shader.h"

void cache_flush_all(void);

#define ALIGN16(x) (((x) + 15u) & ~15u)
#define MAX_QPUS 12

void smatvec_tmu(const float *W, const float *x, float *y,
                 int out_dim, int in_dim, int num_qpus, perf_t *perf) {
    demand(out_dim > 0 && (out_dim % 16) == 0,
           "out_dim must be positive multiple of 16");
    demand(in_dim >= 1, "in_dim must be >= 1");
    demand(num_qpus >= 1 && num_qpus <= MAX_QPUS, "num_qpus 1..12");

    uint32_t W_bus = CPU_TO_GPU((uint32_t)W);

    unsigned unif_words = 4 + (unsigned)in_dim;
    unsigned y_bytes    = (unsigned)out_dim * 4;
    unsigned code_bytes = ALIGN16(sizeof(matvec_tmu_shader));
    unsigned unif_bytes = ALIGN16((unsigned)num_qpus * unif_words * 4);
    unsigned ptr_bytes  = ALIGN16((unsigned)num_qpus * 4);

    volatile float    *y_gpu = gpu_arena_alloc(y_bytes);
    void              *code  = gpu_arena_alloc(code_bytes);
    volatile uint32_t *unifs = gpu_arena_alloc(unif_bytes);
    volatile uint32_t *ptrs  = gpu_arena_alloc(ptr_bytes);

    uint32_t y_bus    = CPU_TO_GPU((uint32_t)y_gpu);
    uint32_t code_bus = CPU_TO_GPU((uint32_t)code);

    memset((void *)y_gpu, 0, y_bytes);
    memcpy((void *)code, matvec_tmu_shader, sizeof(matvec_tmu_shader));

    if (perf) perf_start();

    int rows_per_dispatch = num_qpus * 16;

    /*
     * Pack uniforms once: the x vector (in_dim floats) is identical for every
     * QPU in every dispatch. Pack it fully for the first dispatch, then only
     * update the 2 per-QPU header words (W_row_base, y_row_base) on subsequent
     * dispatches. This eliminates ~99% of uniform writes for large matrices
     * (e.g., classifier: 167 dispatches × 12 QPUs × 288 floats → saved).
     */
    int first_dispatch = 1;

    for (int row = 0; row < out_dim; row += rows_per_dispatch) {
        int actual_qpus = 0;

        for (int q = 0; q < num_qpus; q++) {
            int qpu_row = row + q * 16;
            if (qpu_row + 16 > out_dim) break;

            volatile uint32_t *u = unifs + (unsigned)q * unif_words;

            u[0] = W_bus + (uint32_t)qpu_row * (uint32_t)in_dim * 4;
            u[1] = y_bus + (uint32_t)qpu_row * 4;

            if (first_dispatch) {
                u[2] = (uint32_t)in_dim;
                u[3] = (uint32_t)q;
                for (int k = 0; k < in_dim; k++)
                    u[4 + k] = matvec_f2u(x[k]);
                ptrs[q] = CPU_TO_GPU((uint32_t)u);
            }

            actual_qpus++;
        }

        if (actual_qpus > 0)
            qpu_launch(code_bus, (uint32_t *)ptrs, actual_qpus);

        first_dispatch = 0;
    }

    if (perf) *perf = perf_stop();

    cache_flush_all(); /* invalidate D-cache before reading GPU-written y */
    for (int i = 0; i < out_dim; i++)
        y[i] = y_gpu[i];
}
