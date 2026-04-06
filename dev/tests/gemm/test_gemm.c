#include "rpi.h"
#include "gpu.h"
#include "profiler.h"
#include "gemm.h"

#define NUM_QPUS  12
#define MAX_DIM   32

static float A_buf[MAX_DIM * MAX_DIM];
static float B_buf[MAX_DIM * MAX_DIM];
static float C_ref[MAX_DIM * MAX_DIM];
static float C_gpu[MAX_DIM * MAX_DIM];

static uint32_t rng_state;
static float randf(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (float)(int)((rng_state >> 16) & 0xF) + 1.0f;
}

static void print_float(float f) {
    int whole = (int)f;
    int frac  = (int)((f - (float)whole) * 100.0f);
    if (frac < 0) frac = -frac;
    printk("%d.%d", whole, frac);
}

static int verify(const char *tag, const float *ref, const float *got,
                  int M, int N) {
    int errs = 0;
    for (int i = 0; i < M * N; i++) {
        float d = ref[i] - got[i];
        if (d < 0) d = -d;
        if (d > 0.01f) {
            if (errs < 4) {
                printk("  MISMATCH [%d]: expected ", i);
                print_float(ref[i]);
                printk("  got ");
                print_float(got[i]);
                printk("\n");
            }
            errs++;
        }
    }
    if (errs)
        printk("  %s FAIL: %d mismatches\n", tag, errs);
    else
        printk("  %s PASS\n", tag);
    return errs;
}

static void fill(float *buf, int n, uint32_t seed) {
    rng_state = seed;
    for (int i = 0; i < n; i++)
        buf[i] = randf();
}

static void bench(int dim) {
    int M = dim, K = dim, N = dim;
    printk("=== GEMM benchmark (%dx%d) ===\n\n", dim, dim);

    fill(A_buf, M * K, 42);
    fill(B_buf, K * N, 137);

    perf_t p;

    perf_start();
    sgemm_cpu(A_buf, B_buf, C_ref, M, K, N);
    p = perf_stop();
    perf_print("cpu_ref", M, K, N, &p);

    sgemm_tmu(A_buf, B_buf, C_gpu, M, K, N, NUM_QPUS, &p);
    perf_print("gpu_tmu_12", M, K, N, &p);
    verify("gpu_tmu_12", C_ref, C_gpu, M, N);

    printk("\n");
}

void notmain(void) {
    qpu_enable();
    perf_init();

    bench(32);
    bench(16);

    qpu_disable();
    printk("=== done ===\n");
}
