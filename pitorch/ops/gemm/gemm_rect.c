#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "arena.h"
#include "gemm_rect.h"
#include "gemm_rect_tmu_shader.h"

void cache_flush_all(void);

#define ALIGN16(x) (((x) + 15u) & ~15u)
#define MAX_QPUS 12

/* ================================================================
 * Rectangular GEMM:  C[M][N] += A[M][K] * B[K][N]
 *
 * Arena-based, same pattern as smatvec_tmu.
 * - A packed into uniforms (rows assigned round-robin to QPUs)
 * - B read from ARM memory via TMU (CPU_TO_GPU addressing)
 * - C allocated from GPU arena, copied back after completion
 * - N tiled in 16-column chunks (one QPU dispatch per tile)
 * - N must be a multiple of 16
 *
 * For weight gradients:  dW[out×in]  += dY^T[out×T] * X[T×in]   (K=T)
 * For batched forward:   Y[M×N]      = X[M×K]      * W^T[K×N]   (K=in_dim)
 * For batched input grad: dX[M×N]    = dY[M×K]     * W[K×N]     (K=out_dim)
 * ================================================================ */

void sgemm_rect_tmu(const float *A, const float *B, float *C,
                    int M, int K, int N,
                    int num_qpus, int accumulate) {
    demand(M > 0, "M must be positive");
    demand(K >= 1, "K must be >= 1");
    demand(N > 0 && (N % 16) == 0, "N must be positive multiple of 16");
    demand(num_qpus >= 1 && num_qpus <= MAX_QPUS, "num_qpus 1..12");

    uint32_t B_bus = CPU_TO_GPU((uint32_t)B);
    unsigned N_stride = (unsigned)N * 4;  /* row stride for B and C in bytes */

    unsigned max_rows_per_q = (unsigned)((M + num_qpus - 1) / num_qpus);
    unsigned unif_words = 7 + max_rows_per_q * (unsigned)K;
    unsigned C_bytes    = (unsigned)M * (unsigned)N * 4;
    unsigned code_bytes = ALIGN16(sizeof(gemm_rect_tmu_shader));
    unsigned unif_bytes = ALIGN16((unsigned)num_qpus * unif_words * 4);
    unsigned ptr_bytes  = ALIGN16((unsigned)num_qpus * 4);

    volatile float    *C_gpu = gpu_arena_alloc(C_bytes);
    void              *code  = gpu_arena_alloc(code_bytes);
    volatile uint32_t *unifs = gpu_arena_alloc(unif_bytes);
    volatile uint32_t *ptrs  = gpu_arena_alloc(ptr_bytes);

    uint32_t C_bus    = CPU_TO_GPU((uint32_t)C_gpu);
    uint32_t code_bus = CPU_TO_GPU((uint32_t)code);

    /* If accumulating, copy existing C into GPU buffer; otherwise zero it */
    if (accumulate) {
        for (unsigned i = 0; i < (unsigned)(M * N); i++)
            ((volatile float *)C_gpu)[i] = C[i];
    } else {
        memset((void *)C_gpu, 0, C_bytes);
    }
    memcpy((void *)code, gemm_rect_tmu_shader, sizeof(gemm_rect_tmu_shader));

    /* Pack uniforms: A values packed once (first tile), header updated per tile.
     * The QPU re-reads from ptrs[q] on each launch, so A values persist. */
    for (int q = 0; q < num_qpus; q++) {
        volatile uint32_t *u = unifs + (unsigned)q * unif_words;

        int num_rows = 0;
        for (int row = q; row < M; row += num_qpus) num_rows++;

        /* Header (words 0-6): filled with tile 0 values initially */
        u[0] = B_bus;                         /* B_base (updated per tile) */
        u[1] = C_bus;                         /* C_base (updated per tile) */
        u[2] = (uint32_t)K;
        u[3] = N_stride;
        u[4] = (uint32_t)num_qpus;
        u[5] = (uint32_t)q;
        u[6] = (uint32_t)num_rows;

        /* A values (word 7+): packed once, reused for all tiles */
        int ui = 7;
        for (int row = q; row < M; row += num_qpus)
            for (int kk = 0; kk < K; kk++)
                u[ui++] = gemm_f2u(A[row * K + kk]);

        ptrs[q] = CPU_TO_GPU((uint32_t)u);
    }

    /* Tile N in 16-column chunks */
    for (int ct = 0; ct < N; ct += 16) {
        /* Update B_base and C_base for this column tile */
        for (int q = 0; q < num_qpus; q++) {
            volatile uint32_t *u = unifs + (unsigned)q * unif_words;
            u[0] = B_bus + (unsigned)ct * 4;
            u[1] = C_bus + (unsigned)ct * 4;
        }
        qpu_launch(code_bus, (uint32_t *)ptrs, num_qpus);
    }

    /* Read back results */
    cache_flush_all();
    for (unsigned i = 0; i < (unsigned)(M * N); i++)
        C[i] = C_gpu[i];
}
