/*
 * Pipeline-parallel training: rank 1 (worker).
 *
 * Rank 1 owns: layers SPLIT..N.
 *
 * Forward:  recv act → layers SPLIT..N → send act
 * Backward: recv d_res → layers SPLIT..N bwd → send d_res
 * SGD:      update layers SPLIT..N weights
 *
 * SD card: initramfs weights/<model>.bin 0x2000000
 */
#include "rpi.h"
#include "pt.h"
#include "pt_ops.h"
#include "pt_link_gpio.h"
#include "pt_proto.h"
#include "profiler.h"
#include "mmu.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u
#define MAX_T        8
#define N_STEPS      20
#define LR           0.001f
#define RANK         1

/* Rank 1 uses low bank */
#define D_BASE  4
#define CLK_PIN 12
#define ACK_PIN 13

void notmain(void) {
    mmu_init_and_enable();

    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, MAX_T, ARENA_SIZE);
    pt_print_config(&ctx);

    int n_layers = ctx.cfg.n_layers;
    int split = n_layers / 2;
    int dim = ctx.cfg.dim;
    unsigned act_bytes = MAX_T * dim * sizeof(float);

    printk("pipeline training: rank 1, layers %d..%d\n", split, n_layers);
    printk("  %d steps, act_bytes=%d\n", N_STEPS, act_bytes);

    /* Init GPIO link */
    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);

    /* Handshake */
    uint32_t dummy = 0, opcode, plen;
    printk("waiting for PING...\n");
    if (pt_proto_recv(&gpio.base, &opcode, &dummy, sizeof(dummy), &plen) < 0 ||
        opcode != PT_OP_PING) {
        printk("FAIL: no PING\n");
        clean_reboot();
    }
    printk("got PING, sending PONG...\n");
    pt_proto_send(&gpio.base, PT_OP_PONG, &dummy, sizeof(dummy));
    printk("handshake done!\n");

    /* Training loop */
    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0 = timer_get_usec();

        pt_zero_grads(&ctx.grads);

        /* ── Forward ── */

        /* Recv activations[split] from rank 0 */
        float *act_in = ctx.acts.residuals + split * MAX_T * dim;
        if (gpio.base.recv_raw(&gpio.base, act_in, act_bytes) < 0) {
            printk("FAIL: recv fwd act at step %d\n", step);
            clean_reboot();
        }

        uint32_t t_recv_fwd = timer_get_usec();

        /* Layers split..n_layers */
        pt_forward_train_layers_range(&ctx.cfg, &ctx.w, &ctx.acts, MAX_T,
                                       split, n_layers, ctx.matvec, ctx.trace);

        uint32_t t_fwd1 = timer_get_usec();

        /* Send activations[n_layers] back to rank 0 */
        float *act_out = ctx.acts.residuals + n_layers * MAX_T * dim;
        gpio.base.send_raw(&gpio.base, act_out, act_bytes);

        uint32_t t_send_fwd = timer_get_usec();

        /* ── Backward ── */

        /* Recv d_res from rank 0 */
        if (gpio.base.recv_raw(&gpio.base, ctx.bb.d_res, act_bytes) < 0) {
            printk("FAIL: recv bwd d_res at step %d\n", step);
            clean_reboot();
        }

        uint32_t t_recv_bwd = timer_get_usec();

        /* Layers split..n_layers backward */
        pt_backward_layers_range(&ctx.cfg, &ctx.w, &ctx.grads, &ctx.acts,
                                  &ctx.bb, MAX_T, split, n_layers,
                                  ctx.matvec, ctx.trace);

        uint32_t t_bwd1 = timer_get_usec();

        /* Send d_res back to rank 0 */
        gpio.base.send_raw(&gpio.base, ctx.bb.d_res, act_bytes);

        uint32_t t_send_bwd = timer_get_usec();

        /* ── SGD ── */
        pt_sgd_update_layers(&ctx.w, &ctx.grads, LR, &ctx.cfg, split, n_layers);

        uint32_t t_sgd = timer_get_usec();

        printk("step %d: recv_fwd=%d fwd=%d send_fwd=%d "
               "recv_bwd=%d bwd=%d send_bwd=%d sgd=%d total=%d us\n",
               step,
               t_recv_fwd - t0, t_fwd1 - t_recv_fwd, t_send_fwd - t_fwd1,
               t_recv_bwd - t_send_fwd, t_bwd1 - t_recv_bwd,
               t_send_bwd - t_bwd1, t_sgd - t_send_bwd, t_sgd - t0);
    }

    printk("=== rank 1 DONE ===\n");
    clean_reboot();
}
