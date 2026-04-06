/*
 * Distributed pipeline implementation.
 *
 * All the ring-topology forward, backward, and SGD logic for pipeline-parallel
 * inference and training across a GPIO ring of Pi Zeros.
 */

#include "pt_dist_pipeline.h"
#include "pt_ops.h"
#include "pt_proto.h"
#include "rpi.h"

/* ── timestamp formatting ──────────────────────────────────── */

/* Print [MM:SS.mmm] timestamp from boot */
static void print_ts(void) {
    unsigned us = timer_get_usec();
    unsigned ms = us / 1000;
    unsigned s  = ms / 1000;
    unsigned m  = s / 60;
    printk("[%02d:%02d.%03d]", m, s % 60, ms % 1000);
}

/* ── Setup ─────────────────────────────────────────────────── */

void pt_dist_setup(pt_dist_t *dist, const pt_shard_info_t *shard,
                   int rank, int world_size) {
    *dist = pt_dist_init_gpio(rank, world_size);
    dist->l_start   = 0;
    dist->l_end     = shard->n_local;
    dist->has_embed = shard->has_embed;
    dist->has_head  = shard->has_head;
}

/* File-level verbose flag */
static int g_verbose = 0;
static int g_step    = 0;
static int g_rank    = -1;

void pt_dist_set_verbose(pt_dist_t *dist, int verbose) {
    g_verbose = verbose;
    g_rank = dist->rank;
}

/* Log a timed operation */
static void vlog(int rank, int step, const char *phase, const char *op, unsigned ms) {
    if (!g_verbose) return;
    print_ts();
    printk(" R%d step %2d | %s: %-24s %5dms\n", rank, step, phase, op, ms);
}

/* Log step summary (head rank only — has loss) */
static void vlog_step_head(int rank, int step, float loss, unsigned total_ms) {
    if (!g_verbose) return;
    unsigned s  = total_ms / 1000;
    unsigned ds = (total_ms % 1000) / 100;
    printk("========================================================\n");
    print_ts();
    printk(" STEP %d | loss = ", step);
    pt_pf(loss, 4);
    printk(" | %d.%ds/step\n", s, ds);
    printk("========================================================\n");
}

/* Log step summary (layer rank — no loss) */
static void vlog_step_layer(int rank, int step, unsigned total_ms) {
    if (!g_verbose) return;
    printk("--------------------------------------------------------\n");
    print_ts();
    printk(" R%d STEP %d DONE | %dms\n", rank, step, total_ms);
    printk("--------------------------------------------------------\n");
}

/* ── Ring sync ─────────────────────────────────────────────── */

void pt_dist_ring_sync(pt_dist_t *dist) {
    if (dist->has_embed && dist->has_head) {
        /* Coordinator: send PING around the ring */
        delay_ms(5000);
        if (g_verbose) {
            print_ts();
            printk(" R%d ring sync: sending PING...\n", dist->rank);
        }
        uint32_t dummy = 0x42;
        pt_proto_send(&dist->downstream.base, PT_OP_PING, &dummy, sizeof(dummy));
        uint32_t op, plen;
        if (pt_proto_recv(&dist->upstream.base, &op, &dummy, sizeof(dummy), &plen) < 0) {
            printk("FAIL: ring handshake timeout\n");
            clean_reboot();
        }
        if (g_verbose) {
            print_ts();
            printk(" R%d ring sync: OK\n", dist->rank);
        }
    } else {
        /* Forward the PING */
        uint32_t op, plen, dummy;
        if (pt_proto_recv(&dist->upstream.base, &op, &dummy, sizeof(dummy), &plen) < 0) {
            printk("FAIL: ring handshake timeout\n");
            clean_reboot();
        }
        pt_proto_send(&dist->downstream.base, PT_OP_PING, &dummy, sizeof(dummy));
        if (g_verbose) {
            print_ts();
            printk(" R%d ring sync: forwarded\n", dist->rank);
        }
    }
}

void pt_dist_reset_links(pt_dist_t *dist) {
    /* Reset both links to idle input mode */
    dist->downstream.mode = 0;
    dist->upstream.mode = 0;
    for (int i = 0; i < 8; i++) {
        gpio_set_input(dist->downstream.d_base + i);
        gpio_set_input(dist->upstream.d_base + i);
    }
    gpio_set_input(dist->downstream.clk_pin);
    gpio_set_input(dist->downstream.ack_pin);
    gpio_set_input(dist->upstream.clk_pin);
    gpio_set_input(dist->upstream.ack_pin);
    dev_barrier();
    delay_ms(100);
}

/* ── Distributed training step ─────────────────────────────── */

pt_dist_result_t pt_dist_train_step(pt_context_t *ctx, pt_dist_t *dist,
                                    const int *tokens, int T, float lr) {
    pt_dist_result_t result = { 0.0f, 0 };
    int rank = dist->rank;
    int n_local = dist->l_end - dist->l_start;
    int dim = ctx->cfg.dim;
    unsigned act_bytes = T * dim * sizeof(float);
    uint32_t t0 = timer_get_usec();

    pt_zero_grads(&ctx->grads);

    if (dist->has_embed && dist->has_head) {
        /* ═══════════════════════════════════
         * Embed + head rank (coordinator)
         * ═══════════════════════════════════ */
        uint32_t t1;

        /* Forward: embed */
        pt_forward_train_embed(&ctx->w, &ctx->acts, tokens, T, dim);
        t1 = timer_get_usec();
        vlog(rank, g_step, "fwd", "embedding", (t1 - t0) / 1000);

        /* Forward: send activations downstream */
        uint32_t ts = t1;
        float *fwd_out = ctx->acts.residuals;
        dist->downstream.base.send_raw(&dist->downstream.base, fwd_out, act_bytes);
        t1 = timer_get_usec();
        vlog(rank, g_step, "fwd", "send downstream", (t1 - ts) / 1000);

        /* Forward: recv from upstream (after all layer ranks) */
        ts = t1;
        float *fwd_in = ctx->acts.residuals;  /* n_local=0, slot is free */
        if (dist->upstream.base.recv_raw(&dist->upstream.base, fwd_in, act_bytes) < 0) {
            printk("FAIL: recv fwd act step %d\n", g_step);
            clean_reboot();
        }
        t1 = timer_get_usec();
        vlog(rank, g_step, "fwd", "recv upstream", (t1 - ts) / 1000);

        /* Forward: head (rmsnorm + classifier + loss) */
        ts = t1;
        float loss = pt_forward_train_head(&ctx->cfg, &ctx->w, &ctx->acts,
                                            tokens, T, ctx->matvec, ctx->trace);
        t1 = timer_get_usec();
        vlog(rank, g_step, "fwd", "head", (t1 - ts) / 1000);

        /* Backward: head */
        ts = t1;
        pt_backward_head(&ctx->cfg, &ctx->w, &ctx->grads, &ctx->acts,
                         &ctx->bb, tokens, T, ctx->matvec, ctx->trace);
        t1 = timer_get_usec();
        vlog(rank, g_step, "bwd", "head", (t1 - ts) / 1000);

        /* Backward: send d_res upstream */
        ts = t1;
        dist->upstream.base.send_raw(&dist->upstream.base, ctx->bb.d_res, act_bytes);
        t1 = timer_get_usec();
        vlog(rank, g_step, "bwd", "send upstream", (t1 - ts) / 1000);

        /* Backward: recv d_res from downstream (ring closure) */
        ts = t1;
        if (dist->downstream.base.recv_raw(&dist->downstream.base,
                                            ctx->bb.d_res, act_bytes) < 0) {
            printk("FAIL: recv bwd d_res step %d\n", g_step);
            clean_reboot();
        }
        t1 = timer_get_usec();
        vlog(rank, g_step, "bwd", "recv downstream", (t1 - ts) / 1000);

        /* Backward: embedding */
        ts = t1;
        pt_backward_embed(&ctx->grads, &ctx->bb, tokens, T, dim);
        t1 = timer_get_usec();
        vlog(rank, g_step, "bwd", "embedding", (t1 - ts) / 1000);

        /* SGD: update embed + head weights */
        ts = t1;
        pt_sgd_update_head(&ctx->w, &ctx->grads, lr, &ctx->cfg);
        t1 = timer_get_usec();
        vlog(rank, g_step, "sgd", "update", (t1 - ts) / 1000);

        result.loss = loss;
        result.total_ms = (t1 - t0) / 1000;
        vlog_step_head(rank, g_step, loss, result.total_ms);

    } else {
        /* ═══════════════════════════════════
         * Layer rank
         * ═══════════════════════════════════ */
        uint32_t t1;

        /* Forward: recv activations from upstream */
        float *act_in = ctx->acts.residuals;
        if (dist->upstream.base.recv_raw(&dist->upstream.base, act_in, act_bytes) < 0) {
            printk("FAIL: recv fwd act step %d\n", g_step);
            clean_reboot();
        }
        t1 = timer_get_usec();
        vlog(rank, g_step, "fwd", "recv upstream", (t1 - t0) / 1000);

        /* Forward: compute local layers */
        uint32_t ts = t1;
        pt_forward_train_layers_range(&ctx->cfg, &ctx->w, &ctx->acts, T,
                                       0, n_local, ctx->matvec, ctx->trace);
        t1 = timer_get_usec();
        vlog(rank, g_step, "fwd", "layers", (t1 - ts) / 1000);

        /* Forward: send activations downstream */
        ts = t1;
        float *act_out = ctx->acts.residuals + n_local * T * dim;
        dist->downstream.base.send_raw(&dist->downstream.base, act_out, act_bytes);
        t1 = timer_get_usec();
        vlog(rank, g_step, "fwd", "send downstream", (t1 - ts) / 1000);

        /* Backward: recv gradients from downstream (reversed) */
        ts = t1;
        if (dist->downstream.base.recv_raw(&dist->downstream.base,
                                            ctx->bb.d_res, act_bytes) < 0) {
            printk("FAIL: recv bwd d_res step %d\n", g_step);
            clean_reboot();
        }
        t1 = timer_get_usec();
        vlog(rank, g_step, "bwd", "recv downstream", (t1 - ts) / 1000);

        /* Backward: compute local layers */
        ts = t1;
        pt_backward_layers_range(&ctx->cfg, &ctx->w, &ctx->grads, &ctx->acts,
                                  &ctx->bb, T, 0, n_local,
                                  ctx->matvec, ctx->trace);
        t1 = timer_get_usec();
        vlog(rank, g_step, "bwd", "layers", (t1 - ts) / 1000);

        /* Backward: send gradients upstream (reversed) */
        ts = t1;
        dist->upstream.base.send_raw(&dist->upstream.base, ctx->bb.d_res, act_bytes);
        t1 = timer_get_usec();
        vlog(rank, g_step, "bwd", "send upstream", (t1 - ts) / 1000);

        /* SGD: update local layer weights */
        ts = t1;
        pt_sgd_update_layers(&ctx->w, &ctx->grads, lr, &ctx->cfg, 0, n_local);
        t1 = timer_get_usec();
        vlog(rank, g_step, "sgd", "update", (t1 - ts) / 1000);

        result.total_ms = (t1 - t0) / 1000;
        vlog_step_layer(rank, g_step, result.total_ms);
    }

    g_step++;
    return result;
}

/* ── Distributed inference step ────────────────────────────── */

static int g_infer_pos = 0;

int pt_dist_forward_step(pt_context_t *ctx, pt_dist_t *dist, int token) {
    int rank = dist->rank;
    int dim = ctx->cfg.dim;
    unsigned xfer_bytes = 4 + dim * sizeof(float);  /* [pos | x[dim]] */
    uint32_t t0 = timer_get_usec(), t1, ts;

    if (dist->has_embed && dist->has_head) {
        /* Embed token */
        pt_forward_embed(&ctx->w, ctx->state.x, dim, token);
        t1 = timer_get_usec();
        vlog(rank, g_infer_pos, "gen", "embed", (t1 - t0) / 1000);

        /* Send [pos | x] downstream */
        ts = t1;
        uint8_t buf[4 + 4096];  /* max dim=1024 → 4100 bytes */
        int32_t pos = ctx->pos;
        memcpy(buf, &pos, 4);
        memcpy(buf + 4, ctx->state.x, dim * sizeof(float));
        dist->downstream.base.send_raw(&dist->downstream.base, buf, xfer_bytes);
        t1 = timer_get_usec();
        vlog(rank, g_infer_pos, "gen", "send downstream", (t1 - ts) / 1000);

        /* Recv [pos | x] from upstream (after all layer ranks) */
        ts = t1;
        if (dist->upstream.base.recv_raw(&dist->upstream.base, buf, xfer_bytes) < 0) {
            printk("FAIL: dist forward recv pos %d\n", ctx->pos);
            clean_reboot();
        }
        memcpy(ctx->state.x, buf + 4, dim * sizeof(float));
        t1 = timer_get_usec();
        vlog(rank, g_infer_pos, "gen", "recv upstream", (t1 - ts) / 1000);

        /* Head: rmsnorm + classifier + argmax */
        ts = t1;
        pt_forward_head(&ctx->cfg, &ctx->w, &ctx->state, ctx->matvec);
        int next = argmax(ctx->state.logits, ctx->cfg.vocab_size);
        t1 = timer_get_usec();
        vlog(rank, g_infer_pos, "gen", "head", (t1 - ts) / 1000);

        if (g_verbose) {
            unsigned total = (t1 - t0) / 1000;
            printk("--------------------------------------------------------\n");
            print_ts();
            printk(" R%d pos %2d → token %d | %dms\n", rank, g_infer_pos, next, total);
            printk("--------------------------------------------------------\n");
        }

        ctx->pos++;
        g_infer_pos++;
        return next;

    } else {
        /* Layer rank: recv [pos | x], run layers, send downstream */
        uint8_t buf[4 + 4096];
        if (dist->upstream.base.recv_raw(&dist->upstream.base, buf, xfer_bytes) < 0) {
            printk("FAIL: dist forward recv (layer rank)\n");
            clean_reboot();
        }
        t1 = timer_get_usec();
        vlog(rank, g_infer_pos, "gen", "recv upstream", (t1 - t0) / 1000);

        int32_t pos;
        memcpy(&pos, buf, 4);
        memcpy(ctx->state.x, buf + 4, dim * sizeof(float));

        ts = t1;
        pt_forward_layers_range(&ctx->cfg, &ctx->w, &ctx->state, pos,
                                dist->l_start, dist->l_end, ctx->matvec);
        t1 = timer_get_usec();
        vlog(rank, g_infer_pos, "gen", "layers", (t1 - ts) / 1000);

        ts = t1;
        memcpy(buf + 4, ctx->state.x, dim * sizeof(float));
        dist->downstream.base.send_raw(&dist->downstream.base, buf, xfer_bytes);
        t1 = timer_get_usec();
        vlog(rank, g_infer_pos, "gen", "send downstream", (t1 - ts) / 1000);

        if (g_verbose) {
            unsigned total = (t1 - t0) / 1000;
            printk("--------------------------------------------------------\n");
            print_ts();
            printk(" R%d pos %2d DONE | %dms\n", rank, g_infer_pos, total);
            printk("--------------------------------------------------------\n");
        }

        g_infer_pos++;
        return 0;
    }
}
