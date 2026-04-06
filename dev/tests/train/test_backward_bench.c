/*
 * Backward pass benchmark: CPU-only vs GPU, with and without D-cache.
 * Runs one training step each way and prints per-phase timing.
 */
#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "arena.h"
#include "mmu.h"
#include "profiler.h"
#include "pt_ops.h"
#include "pt_math.h"
#include "matvec.h"
#include "llama2.h"
#include "pt_train.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u
#define T_SEQ        8
#define LR           0.001f

static int target[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

static unsigned state_base;
static unsigned scratch_off;
static void *scratch_alloc(unsigned bytes) {
    void *p = (void *)(state_base + scratch_off);
    scratch_off += (bytes + 15u) & ~15u;
    return p;
}

static void matvec_gpu(const float *W, const float *x, float *y,
                       int out_dim, int in_dim) {
    gpu_arena_reset();
    smatvec_tmu(W, x, y, out_dim, in_dim, NUM_QPUS, NULL);
}

static void run_step(const char *label,
                     pt_config_t *cfg, pt_weights_t *w,
                     pt_activations_t *acts, pt_grads_t *grads,
                     pt_backward_buf_t *bb,
                     pt_matvec_fn fwd_mv, pt_matvec_fn bwd_mv) {
    printk("\n--- %s ---\n", label);

    unsigned t0 = timer_get_usec();
    pt_zero_grads(grads);
    unsigned t1 = timer_get_usec();

    float loss = pt_forward_train(cfg, w, acts, target, T_SEQ, fwd_mv, NULL);
    unsigned t2 = timer_get_usec();

    pt_backward(cfg, w, grads, acts, bb, target, T_SEQ, bwd_mv, NULL);
    unsigned t3 = timer_get_usec();

    pt_sgd_update(w, grads, LR, cfg);
    unsigned t4 = timer_get_usec();

    printk("  zero=%dms fwd=%dms bwd=%dms sgd=%dms total=%ds\n",
           (t1-t0)/1000, (t2-t1)/1000, (t3-t2)/1000, (t4-t3)/1000,
           (t4-t0)/1000000);
    printk("  loss="); pt_pf(loss, 4); printk("\n");
}

void notmain(void) {
    mmu_init_and_enable();
    qpu_enable();
    perf_init();
    gpu_arena_init(ARENA_SIZE);

    void *data = (void *)WEIGHT_BASE;
    if (*(volatile int *)data <= 0 || *(volatile int *)data > 65536)
        panic("no weights at 0x%x\n", WEIGHT_BASE);

    pt_config_t cfg;
    pt_load_config(&cfg, data);
    int shared = ((const int *)data)[5] > 0;
    pt_weights_t w;
    pt_load_weights(&w, &cfg, data);
    unsigned model_bytes = pt_file_size(data);
    state_base = (WEIGHT_BASE + model_bytes + 0x100000) & ~0xFFFFF;

    scratch_off = 0;
    pt_activations_t acts;
    pt_scratch_alloc_activations(&acts, &cfg, T_SEQ, scratch_alloc);
    unsigned grad_base = (state_base + scratch_off + 0xFFFFF) & ~0xFFFFF;
    pt_grads_t grads;
    pt_scratch_alloc_grads(&grads, &cfg, shared, grad_base);
    pt_backward_buf_t bb;
    pt_scratch_alloc_backward_buf(&bb, &cfg, T_SEQ, scratch_alloc);
    unsigned wt_size = cfg.vocab_size * cfg.dim;
    unsigned ht_size = cfg.hidden_dim * cfg.dim;
    unsigned tb_base = grad_base + (unsigned)grads._n_params * 4;
    tb_base = (tb_base + 15u) & ~15u;
    bb.w_transpose = (float *)tb_base;
    int tmp_sz = cfg.dim > cfg.hidden_dim ? cfg.dim : cfg.hidden_dim;
    bb.d_temp = scratch_alloc(tmp_sz * 4);

    printk("\n=== backward pass benchmark ===\n");
    printk("%dd %dL %dKv | T=%d\n", cfg.dim, cfg.n_layers, cfg.vocab_size/1000, T_SEQ);

    /* Step 1: GPU forward, CPU-only backward (matvec=NULL) */
    run_step("GPU fwd + CPU bwd (D-cache ON)",
             &cfg, &w, &acts, &grads, &bb, matvec_gpu, NULL);

    /* Step 2: GPU forward, GPU backward */
    run_step("GPU fwd + GPU bwd (D-cache ON)",
             &cfg, &w, &acts, &grads, &bb, matvec_gpu, matvec_gpu);

    /* Step 3: CPU forward, CPU backward */
    run_step("CPU fwd + CPU bwd (D-cache ON)",
             &cfg, &w, &acts, &grads, &bb, smatvec_cpu, NULL);
}
