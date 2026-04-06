/*
 * On-device training: SGD overfit on a fixed 8-token target.
 * Forward: GPU-accelerated matvec (12 QPUs).
 * Backward: CPU backward ops (GPU-accelerated transposed matvec).
 *
 * SD card: initramfs weights/stories15M.bin 0x2000000
 */
#include "rpi.h"
#include "mmu.h"
#include "pt.h"
#include "pt_ops.h"
#include "profiler.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u

#define T_SEQ        8
#define N_STEPS      20
#define LR           0.001f
#define LOSS_TARGET  0.1f

static int target[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

void notmain(void) {
    mmu_init_and_enable();

    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, T_SEQ, ARENA_SIZE);

    /* ── training loop ── */
    printk("\nT=%d lr=", T_SEQ); pt_pf(LR, 4); printk(" %d QPUs\n\n", NUM_QPUS);

    for (int step = 0; step < N_STEPS; step++) {
        unsigned t0 = timer_get_usec();
        float loss = pt_train_step(&ctx, target, T_SEQ, LR);
        unsigned elapsed = (timer_get_usec() - t0) / 1000;

        printk("step %d: loss=", step); pt_pf(loss, 4);
        printk(" (%dms)\n", elapsed);

        if (loss < LOSS_TARGET && step >= 3) {
            printk("\nconverged at step %d\n", step);
            break;
        }
    }

    /* ── verify: greedy decode should reproduce the target ── */
    printk("\n--- verification ---\n");
    pt_reset_kv(&ctx);
    int token = target[0];
    int match = 1;
    printk("  pos 0: target=%d got=%d OK\n", target[0], token);

    for (int t = 0; t < T_SEQ - 1; t++) {
        int next = pt_forward_step(&ctx, token);
        int ok = (next == target[t + 1]);
        if (!ok) match = 0;
        printk("  pos %d: target=%d got=%d %s\n",
               t + 1, target[t + 1], next, ok ? "OK" : "FAIL");
        token = next;
    }

    printk("\n%s\n", match ? "MATCH" : "MISMATCH");
}
