/*
 * 4-Pi pipeline inference over GPIO ring.
 *
 * Ring: rank 0 -> rank 1 -> rank 2 -> rank 3 -> rank 0
 *       (embed)  (middle)  (middle)   (head)
 *
 * stories15M (6 layers): ranks 0,1,2 get 2 layers each, rank 3 = head only.
 * Head on last rank: only a 4-byte token returns via ring edge.
 *
 * Expected tokens: 1 -> 9038 -> 2501 -> 263 -> 931 -> 29892
 * Compile with -DRANK=0/1/2/3.
 */
#include "rpi.h"
#include "pt.h"
#include "pt_ops.h"
#include "pt_dist.h"
#include "pt_proto.h"
#include "profiler.h"

#ifndef RANK
#error "RANK must be defined (0-3)"
#endif

#define NUM_QPUS    12
#define ARENA_SIZE  (1 * 1024 * 1024)
#define N_STEPS     5
#define WEIGHT_BASE 0x02000000u
#define WORLD_SIZE  4

void notmain(void) {
    /* Init model + GPU */
    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, 0, ARENA_SIZE);

    int n_layers = ctx.cfg.n_layers;
    int dim = ctx.cfg.dim;
    unsigned x_bytes = dim * sizeof(float);

    /* Init distributed context */
    pt_dist_t dist = pt_dist_init_gpio(RANK, WORLD_SIZE);

    /* Layer assignment: 3 compute ranks, head-only last rank */
    int compute_ranks = WORLD_SIZE - 1;
    int layers_per = n_layers / compute_ranks;
    if (dist.has_head) {
        dist.l_start = n_layers;
        dist.l_end   = n_layers;
    } else {
        dist.l_start = RANK * layers_per;
        dist.l_end   = (RANK == compute_ranks - 1) ? n_layers
                                                    : (RANK + 1) * layers_per;
    }

    pt_print_config(&ctx);
    pt_dist_print(&dist);
    printk("activation: %d bytes\n", x_bytes);

    /* Rank 0 delays to let others boot and block on recv */
    if (dist.has_embed) delay_ms(3000);

    pt_reset_kv(&ctx);
    int token = 1;  /* BOS */
    int pos = 0;

    uint8_t sendbuf[4 + 512 * 4];  /* [pos | x], max dim=512 */
    uint8_t recvbuf[4 + 512 * 4];

    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0 = timer_get_usec();

        /* --- 1. Get input activations --- */
        if (dist.has_embed) {
            pt_forward_embed(&ctx.w, ctx.state.x, dim, token);
        } else {
            uint32_t op, len;
            if (pt_proto_recv(&dist.upstream.base, &op, recvbuf,
                              4 + x_bytes, &len) < 0) {
                printk("FAIL: recv timeout step %d\n", step);
                clean_reboot();
            }
            memcpy(&pos, recvbuf, 4);
            memcpy(ctx.state.x, recvbuf + 4, x_bytes);
        }

        /* --- 2. Run local layers --- */
        if (dist.l_start < dist.l_end)
            pt_forward_layers_range(&ctx.cfg, &ctx.w, &ctx.state, pos,
                                    dist.l_start, dist.l_end, ctx.matvec);

        uint32_t t1 = timer_get_usec();

        /* --- 3. Produce output --- */
        if (dist.has_head) {
            /* Head: rmsnorm + classifier + argmax */
            pt_forward_head(&ctx.cfg, &ctx.w, &ctx.state, ctx.matvec);
            int next = argmax(ctx.state.logits, ctx.cfg.vocab_size);

            uint32_t t2 = timer_get_usec();
            printk("step %d: %d -> %d | compute=%d head=%d us\n",
                   step, token, next, t1 - t0, t2 - t1);

            /* Send 4-byte token downstream (ring edge -> rank 0) */
            int32_t tok_val = next;
            pt_proto_send(&dist.downstream.base, PT_OP_DATA,
                          &tok_val, sizeof(tok_val));
        } else {
            /* Send [pos | x] downstream */
            int32_t pos_val = pos;
            memcpy(sendbuf, &pos_val, 4);
            memcpy(sendbuf + 4, ctx.state.x, x_bytes);
            pt_proto_send(&dist.downstream.base, PT_OP_DATA,
                          sendbuf, 4 + x_bytes);

            if (dist.has_embed) {
                uint32_t t2 = timer_get_usec();

                /* Rank 0: recv token from ring (rank 3 -> rank 0) */
                uint32_t op, len;
                int32_t next;
                if (pt_proto_recv(&dist.upstream.base, &op, &next,
                                  sizeof(next), &len) < 0) {
                    printk("FAIL: token recv timeout step %d\n", step);
                    clean_reboot();
                }

                uint32_t t3 = timer_get_usec();
                printk("step %d: %d -> %d | compute=%d send=%d ring=%d total=%d us\n",
                       step, token, next, t1 - t0, t2 - t1, t3 - t2, t3 - t0);

                token = next;
                pos++;
            } else {
                printk("step %d: pos=%d layers [%d,%d) compute=%d us\n",
                       step, pos, dist.l_start, dist.l_end, t1 - t0);
            }
        }
    }

    printk("=== pipeline_4pi rank %d DONE ===\n", RANK);
    clean_reboot();
}
