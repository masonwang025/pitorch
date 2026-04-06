/*
 * Pipeline inference over GPIO parallel bus: rank 1 (worker).
 *
 * Runs: recv x → layers SPLIT..N → send x back.
 * Communicates with rank 0 via 8-bit GPIO parallel bus.
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

/* Rank 1 uses low bank */
#define D_BASE  4
#define CLK_PIN 12
#define ACK_PIN 13

void notmain(void) {
    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, 0, ARENA_SIZE);
    pt_print_config(&ctx);

    int n_layers = ctx.cfg.n_layers;
    int split = n_layers / 2;
    int dim = ctx.cfg.dim;
    unsigned x_bytes = dim * sizeof(float);

    printk("pipeline: rank 1 does layers %d..%d\n", split, n_layers);
    printk("activation size: %d bytes\n", x_bytes);

    /* Init GPIO link */
    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);
    printk("gpio link ready: D=%d-%d CLK=%d ACK=%d\n",
           D_BASE, D_BASE + 7, CLK_PIN, ACK_PIN);

    /* Wait for PING from rank 0 */
    uint32_t dummy = 0;
    uint32_t opcode, plen;
    printk("waiting for PING from rank 0...\n");
    if (pt_proto_recv(&gpio.base, &opcode, &dummy, sizeof(dummy), &plen) < 0 ||
        opcode != PT_OP_PING) {
        printk("FAIL: no PING from rank 0\n");
        clean_reboot();
    }
    printk("got PING, sending PONG...\n");
    pt_proto_send(&gpio.base, PT_OP_PONG, &dummy, sizeof(dummy));
    printk("handshake done!\n");

    /* Reset KV cache */
    pt_reset_kv(&ctx);

    uint8_t recvbuf[4 + 512 * 4];  /* pos + x — max dim=512 */

    for (int step = 0; step < N_STEPS; step++) {
        /* recv [pos | x] from rank 0 */
        uint32_t recv_op, recv_len;
        if (pt_proto_recv(&gpio.base, &recv_op, recvbuf, 4 + x_bytes, &recv_len) < 0) {
            printk("FAIL: timeout recv from rank 0 at step %d\n", step);
            clean_reboot();
        }

        /* extract pos and activation */
        int32_t pos;
        memcpy(&pos, recvbuf, 4);
        memcpy(ctx.state.x, recvbuf + 4, x_bytes);

        perf_t p;
        perf_start();

        /* run layers [split, n_layers) */
        pt_forward_layers_range(&ctx.cfg, &ctx.w, &ctx.state, pos,
                                split, n_layers, ctx.matvec);

        p = perf_stop();
        printk("step %d: pos=%d layers %d..%d  (%d us)\n",
               step, pos, split, n_layers, p.wall_us);

        /* send x back to rank 0 */
        pt_proto_send(&gpio.base, PT_OP_DATA, ctx.state.x, x_bytes);
    }

    printk("=== pipeline rank 1 DONE ===\n");
    clean_reboot();
}
