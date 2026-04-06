/*
 * Single-Pi baseline for comparison with pipeline inference.
 * Runs the same 5-step greedy decode, timing each phase.
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

void notmain(void) {
    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, 0, ARENA_SIZE);
    pt_print_config(&ctx);

    int dim = ctx.cfg.dim;
    int n_layers = ctx.cfg.n_layers;
    int split = n_layers / 2;

    printk("=== single-Pi baseline (GPU matvec) ===\n");
    printk("dim=%d layers=%d split_point=%d\n", dim, n_layers, split);

    pt_reset_kv(&ctx);
    int token = 1;  /* BOS */

    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0, t1, t2, t3, t4;

        t0 = timer_get_usec();

        /* embed */
        pt_forward_embed(&ctx.w, ctx.state.x, dim, token);

        t1 = timer_get_usec();

        /* layers [0, split) — what rank 0 would do */
        pt_forward_layers_range(&ctx.cfg, &ctx.w, &ctx.state, ctx.pos,
                                0, split, ctx.matvec);

        t2 = timer_get_usec();

        /* layers [split, n_layers) — what rank 1 would do */
        pt_forward_layers_range(&ctx.cfg, &ctx.w, &ctx.state, ctx.pos,
                                split, n_layers, ctx.matvec);

        t3 = timer_get_usec();

        /* head */
        pt_forward_head(&ctx.cfg, &ctx.w, &ctx.state, ctx.matvec);

        t4 = timer_get_usec();

        int next = argmax(ctx.state.logits, ctx.cfg.vocab_size);

        printk("step %d: %d -> %d | embed=%d layers0_%d=%d layers%d_%d=%d head=%d total=%d us\n",
               step, token, next,
               t1 - t0,
               split, t2 - t1,
               split, n_layers, t3 - t2,
               t4 - t3,
               t4 - t0);

        token = next;
        ctx.pos++;
    }

    printk("=== DONE ===\n");
    clean_reboot();
}
