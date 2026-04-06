/*
 * Pipeline inference over GPIO parallel bus: rank 0 (coordinator).
 *
 * Runs: embed → layers 0..SPLIT → send x → recv x → head → argmax.
 * Communicates with rank 1 via 8-bit GPIO parallel bus.
 *
 * SD card: initramfs weights/<model>.bin 0x2000000
 */
#include "rpi.h"
#include "pt.h"
#include "pt_ops.h"
#include "pt_link_gpio.h"
#include "pt_proto.h"
#include "profiler.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (1 * 1024 * 1024)
#define N_STEPS      5
#define WEIGHT_BASE  0x02000000u

/* Rank 0 uses high bank */
#define D_BASE  16
#define CLK_PIN 24
#define ACK_PIN 25

void notmain(void) {
    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, 0, ARENA_SIZE);
    pt_print_config(&ctx);

    int n_layers = ctx.cfg.n_layers;
    int split = n_layers / 2;
    int dim = ctx.cfg.dim;
    unsigned x_bytes = dim * sizeof(float);

    printk("pipeline: rank 0 does layers 0..%d, rank 1 does %d..%d\n",
           split, split, n_layers);
    printk("activation size: %d bytes\n", x_bytes);

    /* Init GPIO link */
    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);
    printk("gpio link ready: D=%d-%d CLK=%d ACK=%d\n",
           D_BASE, D_BASE + 7, CLK_PIN, ACK_PIN);

    /* Handshake */
    delay_ms(1000);
    uint32_t opcode, plen;
    uint32_t dummy = 0;
    printk("sending PING to rank 1...\n");
    pt_proto_send(&gpio.base, PT_OP_PING, &dummy, sizeof(dummy));

    if (pt_proto_recv(&gpio.base, &opcode, &dummy, sizeof(dummy), &plen) < 0 ||
        opcode != PT_OP_PONG) {
        printk("FAIL: no PONG from rank 1\n");
        clean_reboot();
    }
    printk("rank 1 connected!\n");
    delay_ms(100);

    /* Pipeline decode loop */
    pt_reset_kv(&ctx);
    int token = 1;  /* BOS */

    uint8_t sendbuf[4 + 512 * 4];  /* pos + x — max dim=512 */

    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0, t1, t2, t3, t4, t5;

        t0 = timer_get_usec();

        /* embed */
        pt_forward_embed(&ctx.w, ctx.state.x, dim, token);

        /* local layers [0, split) */
        pt_forward_layers_range(&ctx.cfg, &ctx.w, &ctx.state, ctx.pos,
                                0, split, ctx.matvec);

        t1 = timer_get_usec();

        /* send [pos | x] to rank 1 */
        int32_t pos_val = ctx.pos;
        memcpy(sendbuf, &pos_val, 4);
        memcpy(sendbuf + 4, ctx.state.x, x_bytes);
        pt_proto_send(&gpio.base, PT_OP_DATA, sendbuf, 4 + x_bytes);

        t2 = timer_get_usec();

        /* recv x back from rank 1 */
        uint32_t recv_op, recv_len;
        if (pt_proto_recv(&gpio.base, &recv_op, ctx.state.x, x_bytes, &recv_len) < 0) {
            printk("FAIL: timeout recv from rank 1 at step %d\n", step);
            clean_reboot();
        }

        t3 = timer_get_usec();

        /* head: final rmsnorm + classifier */
        pt_forward_head(&ctx.cfg, &ctx.w, &ctx.state, ctx.matvec);

        t4 = timer_get_usec();

        /* argmax */
        int next = argmax(ctx.state.logits, ctx.cfg.vocab_size);

        t5 = timer_get_usec();

        printk("step %d: %d -> %d | r0_compute=%d send=%d wait+recv=%d head=%d total=%d us\n",
               step, token, next,
               t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t0);

        token = next;
        ctx.pos++;
    }

    printk("=== pipeline rank 0 DONE ===\n");
    clean_reboot();
}
