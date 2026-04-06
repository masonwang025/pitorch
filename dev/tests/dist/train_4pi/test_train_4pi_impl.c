/*
 * 4-Pi pipeline training over GPIO ring.
 *
 * Topology (embed+head on same rank to avoid shared weight sync):
 *   R3: embed + head (no layers) — coordinator
 *   R0: layers [0,2)
 *   R1: layers [2,4)
 *   R2: layers [4,6)
 *
 * Forward  (downstream): R3 embed → R0 → R1 → R2 → R3 head
 * Backward (upstream):   R3 head_bwd → R2 → R1 → R0 → R3 embed_bwd
 * SGD:     each rank updates only its own weights (no communication)
 *
 * Forward uses upstream-recv / downstream-send (normal link direction).
 * Backward uses downstream-recv / upstream-send (reversed link direction).
 *
 * Greedy decode after training uses 4-Pi pipeline inference to verify
 * the trained weights across all ranks.
 *
 * Expected: step 0 loss ≈ 7.4225, converges to < 0.05 in 20 steps,
 * greedy decode matches 7/7 target tokens.
 *
 * Compile with -DRANK=0/1/2/3.
 *
 * SD card: initramfs weights/stories15M_full.bin 0x2000000
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
#define ARENA_SIZE   (100 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u
#define MAX_T        8
#define N_STEPS      20
#define LR           0.001f
#define WORLD_SIZE   4

/* Target sequence (same as all prior training tests) */
static int target[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

void notmain(void) {
    mmu_init_and_enable();

    /* Init model + GPU */
    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, MAX_T, ARENA_SIZE);

    int n_layers = ctx.cfg.n_layers;
    int dim = ctx.cfg.dim;
    unsigned act_bytes = MAX_T * dim * sizeof(float);

    /* Init distributed context */
    pt_dist_t dist = pt_dist_init_gpio(RANK, WORLD_SIZE);

    /* Layer assignment: R0=[0,2), R1=[2,4), R2=[4,6), R3=embed+head */
    int compute_ranks = WORLD_SIZE - 1;  /* R0, R1, R2 */
    int layers_per = n_layers / compute_ranks;  /* 2 for stories15M */

    if (RANK == 3) {
        /* R3: embed + head, no layers */
        dist.l_start = n_layers;
        dist.l_end   = n_layers;
        dist.has_embed = 1;
        dist.has_head  = 1;
    } else {
        /* R0/R1/R2: layers only */
        dist.l_start = RANK * layers_per;
        dist.l_end   = (RANK == compute_ranks - 1) ? n_layers
                                                    : (RANK + 1) * layers_per;
        dist.has_embed = 0;
        dist.has_head  = 0;
    }

    pt_print_config(&ctx);
    pt_dist_print(&dist);

    printk("4pi training: rank %d, act_bytes=%d\n", RANK, act_bytes);
    if (RANK == 3)
        printk("  embed+head rank, %d steps, lr=%d.%03d\n",
               N_STEPS, (int)LR, (int)(LR * 1000) % 1000);
    else
        printk("  layers [%d,%d), %d steps\n",
               dist.l_start, dist.l_end, N_STEPS);

    /*
     * Ring handshake: R3 sends PING downstream around the ring.
     * Each rank recvs on upstream, forwards on downstream.
     * R3 recvs the PING back on upstream — confirms full ring works.
     * This also "warms up" all links in the forward direction.
     */
    if (RANK == 3) {
        delay_ms(5000);
        printk("sending ring PING...\n");
        uint32_t dummy = 0x42;
        pt_proto_send(&dist.downstream.base, PT_OP_PING, &dummy, sizeof(dummy));
        uint32_t op, plen;
        if (pt_proto_recv(&dist.upstream.base, &op, &dummy, sizeof(dummy), &plen) < 0) {
            printk("FAIL: ring handshake timeout\n");
            clean_reboot();
        }
        printk("ring handshake OK!\n");
    } else {
        uint32_t op, plen, dummy;
        if (pt_proto_recv(&dist.upstream.base, &op, &dummy, sizeof(dummy), &plen) < 0) {
            printk("FAIL: ring handshake timeout\n");
            clean_reboot();
        }
        pt_proto_send(&dist.downstream.base, PT_OP_PING, &dummy, sizeof(dummy));
        printk("rank %d: ring handshake forwarded\n", RANK);
    }

    /* ================================================================
     * Training loop
     *
     * Forward uses pt_proto (framed) for reliability on the ring.
     * The overhead is small vs compute (~16 bytes header per transfer).
     * ================================================================ */
    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0 = timer_get_usec();

        pt_zero_grads(&ctx.grads);

        if (RANK == 3) {
            /* ════════════════════════════════════════════════
             * R3: embed + head rank
             * ════════════════════════════════════════════════ */

            /* ── Forward ── */

            /* Embed tokens into residuals[0] */
            pt_forward_train_embed(&ctx.w, &ctx.acts, target, MAX_T, dim);

            /* Send residuals[0] downstream to R0 */
            float *fwd_out = ctx.acts.residuals;
            dist.downstream.base.send_raw(&dist.downstream.base, fwd_out, act_bytes);

            uint32_t t_fwd_send = timer_get_usec();

            /* Recv residuals[n_layers] from R2 (upstream) */
            float *fwd_in = ctx.acts.residuals + n_layers * MAX_T * dim;
            if (dist.upstream.base.recv_raw(&dist.upstream.base, fwd_in, act_bytes) < 0) {
                printk("FAIL: recv fwd act step %d\n", step);
                clean_reboot();
            }

            uint32_t t_fwd_recv = timer_get_usec();

            /* Head: final rmsnorm + classifier + loss */
            float loss = pt_forward_train_head(&ctx.cfg, &ctx.w, &ctx.acts,
                                                target, MAX_T, ctx.matvec, ctx.trace);

            uint32_t t_head_fwd = timer_get_usec();

            /* ── Backward ── */

            /* Head backward: classifier + final rmsnorm → d_res */
            pt_backward_head(&ctx.cfg, &ctx.w, &ctx.grads, &ctx.acts,
                             &ctx.bb, target, MAX_T, ctx.matvec, ctx.trace);

            uint32_t t_head_bwd = timer_get_usec();

            /* Send d_res upstream to R2 (reverse direction on R2→R3 link) */
            dist.upstream.base.send_raw(&dist.upstream.base, ctx.bb.d_res, act_bytes);

            uint32_t t_bwd_send = timer_get_usec();

            /* Recv d_res from R0 (downstream link, ring closure, reversed) */
            if (dist.downstream.base.recv_raw(&dist.downstream.base,
                                               ctx.bb.d_res, act_bytes) < 0) {
                printk("FAIL: recv bwd d_res step %d\n", step);
                clean_reboot();
            }

            uint32_t t_bwd_recv = timer_get_usec();

            /* Embed backward */
            pt_backward_embed(&ctx.grads, &ctx.bb, target, MAX_T, dim);

            /* ── SGD: update embed + head weights ── */
            pt_sgd_update_head(&ctx.w, &ctx.grads, LR, &ctx.cfg);

            uint32_t t_sgd = timer_get_usec();

            printk("step %d: loss=%d.%04d | embed_send=%d fwd_recv=%d head_fwd=%d "
                   "head_bwd=%d bwd_send=%d bwd_recv=%d sgd=%d total=%d us\n",
                   step,
                   (int)loss, ((int)(loss * 10000)) % 10000,
                   t_fwd_send - t0, t_fwd_recv - t_fwd_send,
                   t_head_fwd - t_fwd_recv, t_head_bwd - t_head_fwd,
                   t_bwd_send - t_head_bwd, t_bwd_recv - t_bwd_send,
                   t_sgd - t_bwd_recv, t_sgd - t0);

        } else {
            /* ════════════════════════════════════════════════
             * R0/R1/R2: layer ranks
             * ════════════════════════════════════════════════ */

            /* ── Forward: recv upstream → layers → send downstream ── */
            float *act_in = ctx.acts.residuals + dist.l_start * MAX_T * dim;
            if (dist.upstream.base.recv_raw(&dist.upstream.base, act_in, act_bytes) < 0) {
                printk("FAIL: recv fwd act step %d\n", step);
                clean_reboot();
            }

            uint32_t t_recv_fwd = timer_get_usec();

            pt_forward_train_layers_range(&ctx.cfg, &ctx.w, &ctx.acts, MAX_T,
                                           dist.l_start, dist.l_end,
                                           ctx.matvec, ctx.trace);

            uint32_t t_fwd = timer_get_usec();

            float *act_out = ctx.acts.residuals + dist.l_end * MAX_T * dim;
            dist.downstream.base.send_raw(&dist.downstream.base, act_out, act_bytes);

            uint32_t t_send_fwd = timer_get_usec();

            /* ── Backward: recv downstream (reversed) → layers_bwd → send upstream (reversed) ── */
            if (dist.downstream.base.recv_raw(&dist.downstream.base,
                                               ctx.bb.d_res, act_bytes) < 0) {
                printk("FAIL: recv bwd d_res step %d\n", step);
                clean_reboot();
            }

            uint32_t t_recv_bwd = timer_get_usec();

            pt_backward_layers_range(&ctx.cfg, &ctx.w, &ctx.grads, &ctx.acts,
                                      &ctx.bb, MAX_T, dist.l_start, dist.l_end,
                                      ctx.matvec, ctx.trace);

            uint32_t t_bwd = timer_get_usec();

            dist.upstream.base.send_raw(&dist.upstream.base, ctx.bb.d_res, act_bytes);

            uint32_t t_send_bwd = timer_get_usec();

            /* ── SGD: update local layer weights ── */
            pt_sgd_update_layers(&ctx.w, &ctx.grads, LR, &ctx.cfg,
                                 dist.l_start, dist.l_end);

            uint32_t t_sgd = timer_get_usec();

            printk("step %d: recv_fwd=%d fwd=%d send_fwd=%d "
                   "recv_bwd=%d bwd=%d send_bwd=%d sgd=%d total=%d us\n",
                   step,
                   t_recv_fwd - t0, t_fwd - t_recv_fwd, t_send_fwd - t_fwd,
                   t_recv_bwd - t_send_fwd, t_bwd - t_recv_bwd,
                   t_send_bwd - t_bwd, t_sgd - t_send_bwd, t_sgd - t0);
        }
    }

    /* ================================================================
     * Greedy decode verification via 4-Pi pipeline inference.
     *
     * Uses the trained weights on each rank. Same topology as inference:
     * R3 embeds → R0 layers → R1 layers → R2 layers → R3 head → argmax.
     * Token returns via ring: R3 sends downstream to R0, R0 recvs upstream.
     *
     * But wait — in training the topology is:
     *   forward: R3(embed) → downstream → R0 → R1 → R2 → downstream → R3(head)
     * For inference we do the same, just with inference functions.
     * ================================================================ */
    if (RANK == 3) {
        printk("verifying greedy decode...\n");
    }

    /* Re-synchronize before decode: reset links to idle and do another ring PING.
     * After training's last backward pass, links are in reversed mode. The decode
     * phase uses forward direction. A ring PING re-exercises all links in the
     * forward direction and confirms all ranks are ready. */
    dist.downstream.mode = 0;
    dist.upstream.mode = 0;
    for (int i = 0; i < 8; i++) {
        gpio_set_input(dist.downstream.d_base + i);
        gpio_set_input(dist.upstream.d_base + i);
    }
    gpio_set_input(dist.downstream.clk_pin);
    gpio_set_input(dist.downstream.ack_pin);
    gpio_set_input(dist.upstream.clk_pin);
    gpio_set_input(dist.upstream.ack_pin);
    dev_barrier();
    delay_ms(100);

    if (RANK == 3) {
        delay_ms(500);
        uint32_t dummy2 = 0x43;
        pt_proto_send(&dist.downstream.base, PT_OP_PING, &dummy2, sizeof(dummy2));
        uint32_t op2, plen2;
        pt_proto_recv(&dist.upstream.base, &op2, &dummy2, sizeof(dummy2), &plen2);
        printk("decode handshake OK\n");
    } else {
        uint32_t op2, plen2, dummy2;
        pt_proto_recv(&dist.upstream.base, &op2, &dummy2, sizeof(dummy2), &plen2);
        pt_proto_send(&dist.downstream.base, PT_OP_PING, &dummy2, sizeof(dummy2));
    }

    pt_reset_kv(&ctx);
    int token = target[0];  /* BOS = 1 */
    int pos = 0;
    int match = 0;
    int n_verify = MAX_T - 1;  /* 7 tokens to verify */

    for (int t = 0; t < n_verify; t++) {
        if (RANK == 3) {
            /* Embed token */
            pt_forward_embed(&ctx.w, ctx.state.x, dim, token);

            /* Send x downstream to R0 (with pos) */
            uint8_t sendbuf[4 + 512 * 4];
            int32_t pos_val = pos;
            memcpy(sendbuf, &pos_val, 4);
            memcpy(sendbuf + 4, ctx.state.x, dim * sizeof(float));
            dist.downstream.base.send_raw(&dist.downstream.base,
                                           sendbuf, 4 + dim * sizeof(float));

            /* Recv x from R2 (upstream) after all layers */
            uint8_t recvbuf[4 + 512 * 4];
            if (dist.upstream.base.recv_raw(&dist.upstream.base,
                                             recvbuf, 4 + dim * sizeof(float)) < 0) {
                printk("FAIL: decode recv step %d\n", t);
                clean_reboot();
            }
            memcpy(ctx.state.x, recvbuf + 4, dim * sizeof(float));

            /* Head: rmsnorm + classifier + argmax */
            pt_forward_head(&ctx.cfg, &ctx.w, &ctx.state, ctx.matvec);
            int next = argmax(ctx.state.logits, ctx.cfg.vocab_size);

            if (next == target[t + 1]) match++;
            printk("  pos %d: got %d expected %d %s\n",
                   t, next, target[t + 1],
                   next == target[t + 1] ? "OK" : "MISS");

            token = next;
            pos++;
        } else {
            /* Layer rank: recv [pos|x] from upstream, run layers, send downstream */
            uint8_t buf[4 + 512 * 4];
            if (dist.upstream.base.recv_raw(&dist.upstream.base,
                                             buf, 4 + dim * sizeof(float)) < 0) {
                printk("FAIL: decode recv step %d\n", t);
                clean_reboot();
            }
            int32_t p;
            memcpy(&p, buf, 4);
            memcpy(ctx.state.x, buf + 4, dim * sizeof(float));

            pt_forward_layers_range(&ctx.cfg, &ctx.w, &ctx.state, p,
                                    dist.l_start, dist.l_end, ctx.matvec);

            memcpy(buf + 4, ctx.state.x, dim * sizeof(float));
            dist.downstream.base.send_raw(&dist.downstream.base,
                                           buf, 4 + dim * sizeof(float));
        }
    }

    if (RANK == 3)
        printk("matched %d/%d target tokens\n", match, n_verify);

    printk("=== train_4pi rank %d DONE ===\n", RANK);
    clean_reboot();
}
