#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "gemm.h"
#include "gemm_tmu_shader.h"

#define ALIGN16(x) (((x) + 15u) & ~15u)
#define MAX_QPUS 12

/* ================================================================
 * CPU reference
 * ================================================================ */

void sgemm_cpu(const float *A, const float *B, float *C,
               int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int kk = 0; kk < K; kk++)
                acc += A[i * K + kk] * B[kk * N + j];
            C[i * N + j] = acc;
        }
}

/* ================================================================
 * Multi-QPU TMU GEMM  (A via uniforms, B via TMU, square)
 * ================================================================ */

void sgemm_tmu(const float *A, const float *B, float *C,
               int M, int K, int N, int num_qpus, perf_t *perf) {
    int dim = M;
    unsigned n = (unsigned)(dim * dim);

    unsigned max_rows_per_q = (unsigned)((dim + num_qpus - 1) / num_qpus);
    unsigned unif_per_q = (6 + max_rows_per_q * (unsigned)dim) * 4;

    unsigned B_sz    = n * 4;
    unsigned C_sz    = n * 4;
    unsigned code_sz = sizeof(gemm_tmu_shader);
    unsigned unif_sz = unif_per_q * (unsigned)num_qpus;
    unsigned ptr_sz  = (unsigned)num_qpus * 4;

    unsigned off_B    = 0;
    unsigned off_C    = ALIGN16(off_B + B_sz);
    unsigned off_code = ALIGN16(off_C + C_sz);
    unsigned off_unif = ALIGN16(off_code + code_sz);
    unsigned off_ptr  = ALIGN16(off_unif + unif_sz);
    unsigned total    = off_ptr + ptr_sz;

    gpu_mem_t mem = gpu_alloc(total);
    volatile uint8_t *base = (volatile uint8_t *)mem.cpu_ptr;
    uint32_t bus = mem.bus_addr;

    for (unsigned i = 0; i < n; i++)
        ((volatile float *)(base + off_B))[i] = B[i];
    memset((void *)(base + off_C), 0, C_sz);
    memcpy((void *)(base + off_code), gemm_tmu_shader, code_sz);

    if (perf) perf_start();

    for (int ct = 0; ct < N; ct += 16) {
        for (int q = 0; q < num_qpus; q++) {
            volatile uint32_t *u = (volatile uint32_t *)
                (base + off_unif + (unsigned)q * unif_per_q);

            int ui = 0;
            u[ui++] = bus + off_B + (unsigned)ct * 4;
            u[ui++] = bus + off_C + (unsigned)ct * 4;
            u[ui++] = (uint32_t)dim;
            u[ui++] = (uint32_t)num_qpus;
            u[ui++] = (uint32_t)q;

            int num_rows = 0;
            for (int row = q; row < dim; row += num_qpus) num_rows++;
            u[ui++] = (uint32_t)num_rows;

            for (int row = q; row < dim; row += num_qpus)
                for (int k = 0; k < dim; k++)
                    u[ui++] = f2u(A[row * dim + k]);

            volatile uint32_t *p = (volatile uint32_t *)(base + off_ptr);
            p[q] = bus + off_unif + (unsigned)q * unif_per_q;
        }

        qpu_launch(bus + off_code, (uint32_t *)(base + off_ptr), num_qpus);
    }

    if (perf) *perf = perf_stop();

    for (unsigned i = 0; i < n; i++)
        C[i] = ((volatile float *)(base + off_C))[i];

    gpu_free(&mem);
}
