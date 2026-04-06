/*
 * train.c — On-device training on a single Pi Zero.
 *
 * Overfits a 15M-parameter LLaMA-2 model to an 8-token target sequence
 * using SGD with GPU-accelerated forward and backward passes (12 QPUs).
 * After convergence, verifies the model reproduces the target via greedy decode.
 *
 * ── How to run ──────────────────────────────────────────────────────
 *
 *   cd examples && ./run.sh train              # deploy to Pi 0 (default)
 *   cd examples && PI_DEVICE=2 ./run.sh train  # deploy to Pi 2
 *
 * ── SD card ─────────────────────────────────────────────────────────
 *
 *   initramfs weights/stories15M.bin 0x2000000
 *
 * ── Expected output ─────────────────────────────────────────────────
 *
 *   step 0: loss=7.4225 (23500ms)
 *   step 1: loss=6.8113 (23400ms)
 *   ...
 *   step 17: loss=0.0296 (23400ms)
 *   converged at step 17
 *
 *   --- verification ---
 *   pos 0: target=1    got=1    OK
 *   pos 1: target=365  got=365  OK
 *   ...
 *   MATCH
 *
 * ════════════════════════════════════════════════════════════════════
 */

#include "rpi.h"
#include "mmu.h"
#include "pt.h"
#include "pt_ops.h"

#define WEIGHT_ADDR  ((void *)0x02000000)
#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)   /* 100 MB — fits classifier GEMM without M-tiling */

#define SEQ_LEN      8
#define N_STEPS      20
#define LR           0.001f
#define LOSS_TARGET  0.1f

static int target[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

void notmain(void) {
    mmu_init_and_enable();

    /* Initialize model + GPU (max_T > 0 enables training buffers) */
    pt_context_t ctx;
    pt_pi_init(&ctx, WEIGHT_ADDR, NUM_QPUS, SEQ_LEN, ARENA_SIZE);

    /* ── Training loop ── */
    printk("\nT=%d lr=", SEQ_LEN);
    pt_pf(LR, 4);
    printk(" %d QPUs\n\n", NUM_QPUS);

    for (int step = 0; step < N_STEPS; step++) {
        unsigned t0 = timer_get_usec();
        float loss = pt_train_step(&ctx, target, SEQ_LEN, LR);
        unsigned elapsed = (timer_get_usec() - t0) / 1000;

        printk("step %d: loss=", step);
        pt_pf(loss, 4);
        printk(" (%dms)\n", elapsed);

        if (loss < LOSS_TARGET && step >= 3) {
            printk("\nconverged at step %d\n", step);
            break;
        }
    }

    /* ── Verification: greedy decode should reproduce the target ── */
    printk("\n--- verification ---\n");
    pt_reset_kv(&ctx);

    int token = target[0];
    int match = 1;
    printk("  pos 0: target=%d got=%d OK\n", target[0], token);

    for (int t = 0; t < SEQ_LEN - 1; t++) {
        int next = pt_forward_step(&ctx, token);
        int ok = (next == target[t + 1]);
        if (!ok) match = 0;
        printk("  pos %d: target=%d got=%d %s\n",
               t + 1, target[t + 1], next, ok ? "OK" : "FAIL");
        token = next;
    }

    printk("\n%s\n", match ? "MATCH" : "MISMATCH");
}
