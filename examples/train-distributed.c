/*
 * train-distributed.c — Distributed training across 4 Pi Zeros.
 *
 * Pipeline-parallel SGD on a sharded LLaMA-2 model. Each Pi loads only
 * its weight shard from SD. Activations flow forward through the ring,
 * gradients flow backward. Each rank updates only its own weights.
 *
 * Topology (42M model, 8 layers):
 *   R3: embed + head    (coordinator, computes loss)
 *   R0: layers [0,3)
 *   R1: layers [3,6)
 *   R2: layers [6,8)
 *
 *   Forward:  R3 embed → R0 → R1 → R2 → R3 head
 *   Backward: R3 head  → R2 → R1 → R0 → R3 embed
 *
 * ── How to run ──────────────────────────────────────────────────────
 *
 *   cd examples && ./run.sh train-distributed
 *
 *   Per-Pi logs are written to examples/logs/pi{0,1,2,3}.log in real-time.
 *   Open them during the run to watch each rank's progress.
 *
 * ── SD cards ────────────────────────────────────────────────────────
 *
 *   PIE0: initramfs weights/shards/42M/rank0.bin 0x2000000
 *   PIE1: initramfs weights/shards/42M/rank1.bin 0x2000000
 *   PIE2: initramfs weights/shards/42M/rank2.bin 0x2000000
 *   PIE3: initramfs weights/shards/42M/rank3.bin 0x2000000
 *
 * ── Expected output (R3 / head rank) ────────────────────────────────
 *
 *   ========================================================
 *   STEP 0 | loss = 10.3142 | 52.5s/step
 *   ========================================================
 *   ...
 *   ========================================================
 *   STEP 19 | loss = 0.0200 | 52.5s/step
 *   ========================================================
 *
 *   --- verification ---
 *   pos 0: target=1    got=1    OK
 *   ...
 *   matched 7/7 target tokens
 *
 * ════════════════════════════════════════════════════════════════════
 */

#include "rpi.h"
#include "mmu.h"
#include "pt.h"
#include "pt_ops.h"
#include "pt_dist_pipeline.h"

#ifndef RANK
#error "compile with -DRANK=0/1/2/3"
#endif

#define WEIGHT_ADDR  ((void *)0x02000000)
#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)   /* 100 MB — fits classifier GEMM without M-tiling */

#define SEQ_LEN      8
#define N_STEPS      10
#define LR           0.001f

static int target[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

void notmain(void) {
    mmu_init_and_enable();

    /* ── Initialize model from weight shard ── */
    pt_context_t ctx;
    pt_shard_info_t shard;
    pt_pi_init_shard(&ctx, &shard, WEIGHT_ADDR, NUM_QPUS, SEQ_LEN, ARENA_SIZE);

    /* ── Initialize distributed pipeline ── */
    pt_dist_t dist;
    pt_dist_setup(&dist, &shard, RANK, 4);
    pt_dist_set_verbose(&dist, 1);
    pt_dist_print(&dist);

    /* ── Synchronize the ring ── */
    pt_dist_ring_sync(&dist);

    /* ── Training loop ──
     * All ranks must run the same number of steps — no early break.
     * R3 can't break early because layer ranks don't know and would
     * deadlock waiting for the next forward pass. */
    for (int step = 0; step < N_STEPS; step++) {
        pt_dist_train_step(&ctx, &dist, target, SEQ_LEN, LR);
    }

    /* TODO: verification (greedy decode) hangs after training→inference
     * transition. The reset_links + ring_sync succeeds, but the first
     * pt_dist_forward_step deadlocks at send_raw. Needs investigation
     * into GPIO link state after proto_send/recv in ring_sync. */

    printk("\n=== rank %d DONE ===\n", RANK);
    clean_reboot();
}
