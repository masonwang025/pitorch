/*
 * 4-Pi pipeline inference for stories110M over GPIO ring.
 * Each Pi loads a weight SHARD (not the full model).
 *
 * Topology (110M = 12 layers, shared weights):
 *   R3: embed + head (0 layers)   — coordinator
 *   R0: layers [0,4)              — 4 layers
 *   R1: layers [4,8)              — 4 layers
 *   R2: layers [8,12)             — 4 layers
 *
 * Ring: R3(embed) → R0 → R1 → R2 → R3(head) → token back to R3
 *
 * Each rank loads its shard via initramfs and only allocates KV cache
 * for its own local layers. Activations (dim=768, 3 KB) flow through
 * the ring as [pos(4B) | x(dim*4B)] messages.
 *
 * Memory per rank (inference only, gpu_mem=32, 480 MB ARM):
 *   R0/R1/R2: shard=108MB + KV=25MB ≈ 133MB (28%)
 *   R3:       shard=94MB  + logits=128KB ≈ 94MB (20%)
 *
 * SD card: initramfs weights/shards/110M/rank<N>.bin 0x2000000
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
#define ARENA_SIZE   (2 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u
#define N_STEPS      16
#define WORLD_SIZE   4

void notmain(void) {
    mmu_init_and_enable();

    /* Init model from weight shard — inference only (max_T=0) */
    pt_context_t ctx;
    pt_shard_info_t shard;
    pt_pi_init_shard(&ctx, &shard, (void *)WEIGHT_BASE,
                     NUM_QPUS, 0, ARENA_SIZE);

    int n_local = shard.n_local;
    int dim = ctx.cfg.dim;
    unsigned x_bytes = dim * sizeof(float);

    /* Init distributed context */
    pt_dist_t dist = pt_dist_init_gpio(RANK, WORLD_SIZE);

    /* Layer assignment uses LOCAL indices [0, n_local) */
    dist.l_start   = 0;
    dist.l_end     = n_local;
    dist.has_embed = shard.has_embed;
    dist.has_head  = shard.has_head;

    pt_dist_print(&dist);
    printk("110M inference: rank %d, n_local=%d, dim=%d, x_bytes=%d\n",
           RANK, n_local, dim, x_bytes);

    /* ────────────────────────────────────────────────────────
     * Ring handshake
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
     * Inference loop (autoregressive decode)
     * ════════════════════════════════════════════════════════ */

    /* Message buffer: [pos(4B) | x(dim*4B)] */
    unsigned msg_bytes = 4 + x_bytes;
    uint8_t msgbuf[4 + 768 * 4];  /* max dim=768 for 110M */

    pt_reset_kv(&ctx);
    int token = 1;  /* BOS */
    int pos = 0;

    uint32_t total_start = timer_get_usec();

    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0 = timer_get_usec();

        if (RANK == 3) {
            /* ══════════════════════════════════════════
             * R3: embed + head (no layers)
             * ══════════════════════════════════════════ */

            /* Embed token → x */
            pt_forward_embed(&ctx.w, ctx.state.x, dim, token);
            uint32_t t_emb = timer_get_usec();

            /* Send [pos | x] downstream to R0 */
            int32_t pos_val = pos;
            memcpy(msgbuf, &pos_val, 4);
            memcpy(msgbuf + 4, ctx.state.x, x_bytes);
            pt_proto_send(&dist.downstream.base, PT_OP_DATA,
                          msgbuf, msg_bytes);
            uint32_t t_send = timer_get_usec();

            /* Wait for pipeline: R0 → R1 → R2 → R3 */
            uint32_t op, len;
            if (pt_proto_recv(&dist.upstream.base, &op,
                              msgbuf, msg_bytes, &len) < 0) {
                printk("FAIL: recv timeout step %d\n", step);
                clean_reboot();
            }
            memcpy(ctx.state.x, msgbuf + 4, x_bytes);
            uint32_t t_recv = timer_get_usec();

            /* Head: rmsnorm + classifier + argmax */
            pt_forward_head(&ctx.cfg, &ctx.w, &ctx.state, ctx.matvec);
            int next = argmax(ctx.state.logits, ctx.cfg.vocab_size);
            uint32_t t_head = timer_get_usec();

            printk("step %d: %d -> %d | emb=%d send=%d pipe=%d head=%d total=%d ms\n",
                   step, token, next,
                   (t_emb - t0) / 1000, (t_send - t_emb) / 1000,
                   (t_recv - t_send) / 1000, (t_head - t_recv) / 1000,
                   (t_head - t0) / 1000);

            token = next;
            pos++;

        } else {
            /* ══════════════════════════════════════════
             * R0/R1/R2: layer ranks
             * ══════════════════════════════════════════ */

            /* Recv [pos | x] from upstream */
            uint32_t op, len;
            if (pt_proto_recv(&dist.upstream.base, &op,
                              msgbuf, msg_bytes, &len) < 0) {
                printk("FAIL: recv timeout step %d\n", step);
                clean_reboot();
            }
            memcpy(&pos, msgbuf, 4);
            memcpy(ctx.state.x, msgbuf + 4, x_bytes);
            uint32_t t_recv = timer_get_usec();

            /* Run local layers */
            pt_forward_layers_range(&ctx.cfg, &ctx.w, &ctx.state, pos,
                                    0, n_local, ctx.matvec);
            uint32_t t_fwd = timer_get_usec();

            /* Send [pos | x] downstream */
            memcpy(msgbuf + 4, ctx.state.x, x_bytes);
            pt_proto_send(&dist.downstream.base, PT_OP_DATA,
                          msgbuf, msg_bytes);
            uint32_t t_send = timer_get_usec();

            printk("step %d: pos=%d recv=%d fwd=%d send=%d total=%d ms\n",
                   step, pos,
                   (t_recv - t0) / 1000, (t_fwd - t_recv) / 1000,
                   (t_send - t_fwd) / 1000, (t_send - t0) / 1000);
        }
    }

    uint32_t total_end = timer_get_usec();
    uint32_t total_ms = (total_end - total_start) / 1000;
    printk("\n=== rank %d: %d tokens in %d ms (%d ms/tok) ===\n",
           RANK, N_STEPS, total_ms, total_ms / N_STEPS);

    printk("=== infer_4pi_110M rank %d DONE ===\n", RANK);
    clean_reboot();
}
