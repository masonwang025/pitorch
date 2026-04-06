/*
 * 4-Pi pipeline training for stories42M over GPIO ring.
 * Each Pi loads a weight SHARD (not the full model).
 *
 * Topology (42M = 8 layers, shared weights → embed+head on same rank):
 *   R3: embed + head (0 layers)   — coordinator, computes loss
 *   R0: layers [0,3)              — 3 layers
 *   R1: layers [3,6)              — 3 layers
 *   R2: layers [6,8)              — 2 layers
 *
 * Forward  (downstream ring): R3 embed → R0 → R1 → R2 → R3 head
 * Backward (upstream ring):   R3 head  → R2 → R1 → R0 → R3 embed
 * SGD:     each rank updates only its own weights
 *
 * Profiling: captures per-phase timing for each step, emits trace JSON
 * over UART for collection by the host.
 *
 * SD card: initramfs weights/shards/42M/rank<N>.bin 0x2000000
 * Compile with -DRANK=0/1/2/3.
 */
#include "rpi.h"
#include "pt.h"
#include "pt_ops.h"
#include "pt_dist.h"
#include "pt_proto.h"
#include "profiler.h"
#include "mmu.h"

#ifndef RANK
#error "RANK must be defined (0-3)"
#endif

#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)   /* 100 MB — fits classifier GEMM without M-tiling */
#define WEIGHT_BASE  0x02000000u
#define MAX_T        8
#define N_STEPS      20
#define LR           0.001f
#define WORLD_SIZE   4

/* Target sequence (same as all prior training tests) */
static int target[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

void notmain(void) {
    mmu_init_and_enable();

    /* Init model from weight shard */
    pt_context_t ctx;
    pt_shard_info_t shard;
    pt_pi_init_shard(&ctx, &shard, (void *)WEIGHT_BASE,
                     NUM_QPUS, MAX_T, ARENA_SIZE);

    int n_local = shard.n_local;
    int dim = ctx.cfg.dim;
    unsigned act_bytes = MAX_T * dim * sizeof(float);

    /* Init distributed context */
    pt_dist_t dist = pt_dist_init_gpio(RANK, WORLD_SIZE);

    /* Layer assignment uses LOCAL indices [0, n_local) */
    dist.l_start   = 0;
    dist.l_end     = n_local;
    dist.has_embed = shard.has_embed;
    dist.has_head  = shard.has_head;

    pt_dist_print(&dist);
    printk("42M training: rank %d, n_local=%d, act_bytes=%d\n",
           RANK, n_local, act_bytes);

    /* Enable tracing for detailed profiling */
    pt_trace_t trace;
    pt_trace_init(&trace);
    ctx.trace = &trace;

    /* ────────────────────────────────────────────────────────
     * Ring handshake: R3 sends PING downstream around ring.
     * Each rank recvs upstream, forwards downstream.
     * R3 recvs back upstream — confirms full ring connectivity.
     * ──────────────────────────────────────────────────────── */
    if (RANK == 3) {
        delay_ms(5000);
        printk("sending ring PING...\n");
        uint32_t dummy = 0x42;
        pt_proto_send(&dist.downstream.base, PT_OP_PING,
                      &dummy, sizeof(dummy));
        uint32_t op, plen;
        if (pt_proto_recv(&dist.upstream.base, &op,
                          &dummy, sizeof(dummy), &plen) < 0) {
            printk("FAIL: ring handshake timeout\n");
            clean_reboot();
        }
        printk("ring handshake OK!\n");
    } else {
        uint32_t op, plen, dummy;
        if (pt_proto_recv(&dist.upstream.base, &op,
                          &dummy, sizeof(dummy), &plen) < 0) {
            printk("FAIL: ring handshake timeout\n");
            clean_reboot();
        }
        pt_proto_send(&dist.downstream.base, PT_OP_PING,
                      &dummy, sizeof(dummy));
        printk("rank %d: ring handshake forwarded\n", RANK);
    }

    /* ════════════════════════════════════════════════════════
     * Training loop
     * ════════════════════════════════════════════════════════ */
    uint32_t train_start = timer_get_usec();

    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0 = timer_get_usec();

        pt_trace_begin(&trace, "step", "train", -1);
        pt_zero_grads(&ctx.grads);

        if (RANK == 3) {
            /* ══════════════════════════════════════════
             * R3: embed + head rank (no layers)
             * ══════════════════════════════════════════ */

            /* ── Forward: embed → send → recv → head ── */
            pt_trace_begin(&trace, "embed", "fwd", -1);
            pt_forward_train_embed(&ctx.w, &ctx.acts, target, MAX_T, dim);
            uint32_t t_emb = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "fwd_send", "comm", -1);
            float *fwd_out = ctx.acts.residuals;
            dist.downstream.base.send_raw(&dist.downstream.base,
                                          fwd_out, act_bytes);
            uint32_t t_fsend = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "fwd_recv", "comm", -1);
            /* n_local=0, so residuals[0] slot is free for recv */
            float *fwd_in = ctx.acts.residuals;
            if (dist.upstream.base.recv_raw(&dist.upstream.base,
                                            fwd_in, act_bytes) < 0) {
                printk("FAIL: recv fwd step %d\n", step);
                clean_reboot();
            }
            uint32_t t_frecv = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "head_fwd", "fwd", -1);
            float loss = pt_forward_train_head(&ctx.cfg, &ctx.w, &ctx.acts,
                                               target, MAX_T, ctx.matvec,
                                               ctx.trace);
            uint32_t t_head = timer_get_usec();
            pt_trace_end(&trace);

            /* ── Backward: head → send → recv → embed ── */
            pt_trace_begin(&trace, "head_bwd", "bwd", -1);
            pt_backward_head(&ctx.cfg, &ctx.w, &ctx.grads, &ctx.acts,
                             &ctx.bb, target, MAX_T, ctx.matvec, ctx.trace);
            uint32_t t_hbwd = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "bwd_send", "comm", -1);
            dist.upstream.base.send_raw(&dist.upstream.base,
                                        ctx.bb.d_res, act_bytes);
            uint32_t t_bsend = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "bwd_recv", "comm", -1);
            if (dist.downstream.base.recv_raw(&dist.downstream.base,
                                              ctx.bb.d_res, act_bytes) < 0) {
                printk("FAIL: recv bwd step %d\n", step);
                clean_reboot();
            }
            uint32_t t_brecv = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "embed_bwd", "bwd", -1);
            pt_backward_embed(&ctx.grads, &ctx.bb, target, MAX_T, dim);
            uint32_t t_ebwd = timer_get_usec();
            pt_trace_end(&trace);

            /* ── SGD: update embed + head weights ── */
            pt_trace_begin(&trace, "sgd", "sgd", -1);
            pt_sgd_update_head(&ctx.w, &ctx.grads, LR, &ctx.cfg);
            uint32_t t_sgd = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_end(&trace);  /* step */

            printk("step %d: loss=%d.%04d | "
                   "emb=%d fs=%d fr=%d hf=%d hb=%d bs=%d br=%d eb=%d sgd=%d "
                   "total=%d ms\n",
                   step,
                   (int)loss, ((int)(loss * 10000)) % 10000,
                   (t_emb - t0) / 1000, (t_fsend - t_emb) / 1000,
                   (t_frecv - t_fsend) / 1000, (t_head - t_frecv) / 1000,
                   (t_hbwd - t_head) / 1000, (t_bsend - t_hbwd) / 1000,
                   (t_brecv - t_bsend) / 1000, (t_ebwd - t_brecv) / 1000,
                   (t_sgd - t_ebwd) / 1000, (t_sgd - t0) / 1000);

        } else {
            /* ══════════════════════════════════════════
             * R0/R1/R2: layer ranks
             * ══════════════════════════════════════════ */

            /* ── Forward: recv → layers → send ── */
            pt_trace_begin(&trace, "fwd_recv", "comm", -1);
            float *act_in = ctx.acts.residuals;  /* local layer 0 input */
            if (dist.upstream.base.recv_raw(&dist.upstream.base,
                                            act_in, act_bytes) < 0) {
                printk("FAIL: recv fwd step %d\n", step);
                clean_reboot();
            }
            uint32_t t_frecv = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "fwd_layers", "fwd", -1);
            pt_forward_train_layers_range(&ctx.cfg, &ctx.w, &ctx.acts, MAX_T,
                                          0, n_local, ctx.matvec, ctx.trace);
            uint32_t t_fwd = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "fwd_send", "comm", -1);
            float *act_out = ctx.acts.residuals + n_local * MAX_T * dim;
            dist.downstream.base.send_raw(&dist.downstream.base,
                                          act_out, act_bytes);
            uint32_t t_fsend = timer_get_usec();
            pt_trace_end(&trace);

            /* ── Backward: recv → layers → send ── */
            pt_trace_begin(&trace, "bwd_recv", "comm", -1);
            if (dist.downstream.base.recv_raw(&dist.downstream.base,
                                              ctx.bb.d_res, act_bytes) < 0) {
                printk("FAIL: recv bwd step %d\n", step);
                clean_reboot();
            }
            uint32_t t_brecv = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "bwd_layers", "bwd", -1);
            pt_backward_layers_range(&ctx.cfg, &ctx.w, &ctx.grads, &ctx.acts,
                                     &ctx.bb, MAX_T, 0, n_local,
                                     ctx.matvec, ctx.trace);
            uint32_t t_bwd = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_begin(&trace, "bwd_send", "comm", -1);
            dist.upstream.base.send_raw(&dist.upstream.base,
                                        ctx.bb.d_res, act_bytes);
            uint32_t t_bsend = timer_get_usec();
            pt_trace_end(&trace);

            /* ── SGD: update local layer weights ── */
            pt_trace_begin(&trace, "sgd", "sgd", -1);
            pt_sgd_update_layers(&ctx.w, &ctx.grads, LR, &ctx.cfg,
                                 0, n_local);
            uint32_t t_sgd = timer_get_usec();
            pt_trace_end(&trace);

            pt_trace_end(&trace);  /* step */

            printk("step %d: fr=%d fwd=%d fs=%d br=%d bwd=%d bs=%d sgd=%d "
                   "total=%d ms\n",
                   step,
                   (t_frecv - t0) / 1000, (t_fwd - t_frecv) / 1000,
                   (t_fsend - t_fwd) / 1000, (t_brecv - t_fsend) / 1000,
                   (t_bwd - t_brecv) / 1000, (t_bsend - t_bwd) / 1000,
                   (t_sgd - t_bsend) / 1000, (t_sgd - t0) / 1000);
        }
    }

    uint32_t train_end = timer_get_usec();
    uint32_t total_ms = (train_end - train_start) / 1000;
    printk("\n=== rank %d: %d steps in %d ms (%d ms/step) ===\n",
           RANK, N_STEPS, total_ms, total_ms / N_STEPS);

    /* Emit trace JSON over UART */
    pt_trace_emit_uart(&trace);

    printk("=== train_4pi_42M rank %d DONE ===\n", RANK);
    clean_reboot();
}
