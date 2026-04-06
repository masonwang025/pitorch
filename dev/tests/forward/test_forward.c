/*
 * Forward pass test: CPU vs GPU inference comparison.
 *
 * Runs N_STEPS of greedy decode with both CPU and GPU matvec,
 * verifies they produce the same token sequence.
 *
 * SD card: initramfs weights/<model>.bin 0x2000000
 */
#include "rpi.h"
#include "pt.h"
#include "pt_ops.h"
#include "profiler.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (1 * 1024 * 1024)
#define N_STEPS      5
#define WEIGHT_BASE  0x02000000u

static int run_steps(const char *label, pt_context_t *ctx, int *tokens_out) {
    pt_reset_kv(ctx);
    int token = 1;  /* BOS */
    for (int step = 0; step < N_STEPS; step++) {
        perf_t p;
        perf_start();
        int next = pt_forward_step(ctx, token);
        p = perf_stop();
        tokens_out[step] = next;
        printk("%s step %d: %d -> %d  (%d us)\n",
               label, step, token, next, p.wall_us);
        token = next;
    }
    return 0;
}

void notmain(void) {
    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, 0, ARENA_SIZE);

    /* save GPU matvec, run CPU first */
    pt_matvec_fn gpu_matvec = ctx.matvec;

    int cpu_tokens[N_STEPS];
    printk("\n--- CPU-only ---\n");
    ctx.matvec = smatvec_cpu;
    run_steps("cpu", &ctx, cpu_tokens);

    /* check GPU alignment */
    int head_dim = ctx.cfg.dim / ctx.cfg.n_heads;
    int kv_dim   = ctx.cfg.n_kv_heads * head_dim;
    int gpu_ok = (ctx.cfg.dim % 16 == 0) && (ctx.cfg.hidden_dim % 16 == 0) &&
                 (ctx.cfg.vocab_size % 16 == 0) && (kv_dim % 16 == 0);

    if (gpu_ok) {
        int gpu_tokens[N_STEPS];
        printk("\n--- GPU matvec ---\n");
        ctx.matvec = gpu_matvec;
        run_steps("gpu", &ctx, gpu_tokens);

        printk("\n--- comparison ---\n");
        int all_match = 1;
        for (int i = 0; i < N_STEPS; i++) {
            int match = (gpu_tokens[i] == cpu_tokens[i]);
            if (!match) all_match = 0;
            printk("step %d: cpu=%d gpu=%d %s\n",
                   i, cpu_tokens[i], gpu_tokens[i],
                   match ? "MATCH" : "MISMATCH");
        }
        printk(all_match ? "\nALL MATCH\n" : "\nFAILED\n");
    } else {
        printk("\nGPU: skipped (dims not aligned to 16)\n");
    }

    printk("\n=== done ===\n");
}
