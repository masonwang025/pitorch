/*
 * Pipeline inference: rank 1 (worker).
 *
 * Runs: recv x → layers SPLIT..N → send x back.
 * Communicates with rank 0 via GPIO bit-bang UART.
 *
 * SD card: initramfs weights/<model>.bin 0x2000000
 */
#include "rpi.h"
#include "pt.h"
#include "pt_ops.h"
#include "pt_link.h"
#include "pt_proto.h"
#include "profiler.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (1 * 1024 * 1024)
#define N_STEPS      5
#define WEIGHT_BASE  0x02000000u

#define TX_PIN 17
#define RX_PIN 27
#define BAUD   57600

void notmain(void) {
    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, 0, ARENA_SIZE);
    pt_print_config(&ctx);

    int n_layers = ctx.cfg.n_layers;
    int split = n_layers / 2;   /* rank 1 does layers [split, n_layers) */
    int dim = ctx.cfg.dim;
    unsigned x_bytes = dim * sizeof(float);

    printk("pipeline: rank 1 does layers %d..%d\n", split, n_layers);
    printk("activation size: %d bytes\n", x_bytes);

    /* init link */
    pt_link_t link = pt_link_init(TX_PIN, RX_PIN, BAUD);
    printk("link ready (baud=%d)\n", BAUD);

    /* wait for PING from rank 0 (rank 0 boots later) */
    uint32_t dummy = 0;
    uint32_t opcode, plen;
    printk("waiting for PING from rank 0...\n");
    if (pt_proto_recv(&link.base, &opcode, &dummy, sizeof(dummy), &plen) < 0 ||
        opcode != PT_OP_PING) {
        printk("FAIL: no PING from rank 0\n");
        clean_reboot();
    }
    printk("got PING, sending PONG...\n");
    pt_proto_send(&link.base, PT_OP_PONG, &dummy, sizeof(dummy));
    printk("handshake done!\n");

    /* reset KV cache */
    pt_reset_kv(&ctx);

    /* buffer for receiving [pos | x] */
    uint8_t recvbuf[4 + 288 * 4];  /* pos + x — max dim=288 */

    for (int step = 0; step < N_STEPS; step++) {
        /* recv [pos | x] from rank 0 */
        uint32_t recv_op, recv_len;
        if (pt_proto_recv(&link.base, &recv_op, recvbuf, 4 + x_bytes, &recv_len) < 0) {
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
        pt_proto_send(&link.base, PT_OP_DATA, ctx.state.x, x_bytes);
    }

    printk("=== pipeline rank 1 DONE ===\n");
    clean_reboot();
}
