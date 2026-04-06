#include "rpi.h"
#include "gpu.h"
#include "arena.h"
#include "profiler.h"
#include "pt_ops.h"
#include "matvec.h"

#define NUM_QPUS    12
#define ARENA_SIZE  (2 * 1024 * 1024)
#define TOL         0.05f

/*
 * Scratch memory for large test buffers (32000x288 W matrix = ~35 MB).
 * kmalloc heap is ~700 KB, far too small. Use physical RAM at 32 MB mark
 * instead — well above program/heap/stack, below GPU firmware region.
 */
#define SCRATCH_BASE 0x02000000u
static unsigned scratch_off;
static float *salloc(int n) {
    float *p = (float *)(SCRATCH_BASE + scratch_off);
    scratch_off += (unsigned)n * sizeof(float);
    return p;
}

static uint32_t rng_state;
static float randf(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (float)(int)((rng_state >> 16) & 0x7) + 1.0f;
}

static void fill(float *buf, int n, uint32_t seed) {
    rng_state = seed;
    for (int i = 0; i < n; i++)
        buf[i] = randf();
}

static void print_float(float f) {
    int neg = (f < 0);
    if (neg) f = -f;
    int whole = (int)f;
    int frac  = (int)((f - (float)whole) * 1000.0f);
    if (neg) printk("-");
    printk("%d.%03d", whole, frac);
}

static int verify(const float *ref, const float *got, int n) {
    int errs = 0;
    for (int i = 0; i < n; i++) {
        float d = ref[i] - got[i];
        if (d < 0) d = -d;
        if (d > TOL) {
            if (errs < 3) {
                printk("  [%d] ref=", i);
                print_float(ref[i]);
                printk(" got=");
                print_float(got[i]);
                printk(" diff=");
                print_float(d);
                printk("\n");
            }
            errs++;
        }
    }
    return errs;
}

static void test_size(int M, int K) {
    int n_w = M * K;
    scratch_off = 0;
    float *W     = salloc(n_w);
    float *x     = salloc(K);
    float *y_cpu = salloc(M);
    float *y_gpu = salloc(M);

    fill(W, n_w, 42);
    fill(x, K,   137);

    perf_t p;
    perf_start();
    smatvec_cpu(W, x, y_cpu, M, K);
    p = perf_stop();
    uint32_t cpu_us = p.wall_us;

    gpu_arena_reset();
    smatvec_tmu(W, x, y_gpu, M, K, NUM_QPUS, &p);
    uint32_t gpu_us = p.wall_us;

    int errs = verify(y_cpu, y_gpu, M);

    printk("smatvec %dx%d:\tcpu=%dus\tgpu=%dus\t%s",
           M, K, cpu_us, gpu_us, errs == 0 ? "MATCH" : "MISMATCH");
    if (errs) printk(" (%d errs)", errs);
    printk("\n");
}

void notmain(void) {
    qpu_enable();
    perf_init();
    gpu_arena_init(ARENA_SIZE);

    printk("\n=== smatvec_tmu correctness + perf ===\n\n");

    test_size(48,    48);
    test_size(288,   288);
    test_size(768,   288);
    test_size(288,   768);
    test_size(32000, 288);

    gpu_arena_free();
    qpu_disable();
    printk("\n=== done ===\n");
}
