/*
 * GPipe-style microbatched pipeline training: rank 1 (worker).
 *
 * Rank 1 owns: layers SPLIT..N.
 *
 * Forward:  recv act(0) → F(0) → [send res(mb-1), recv act(mb), F(mb)] × (M-1) → send res(M-1)
 * Backward: recv d_res(M-1) → B(M-1) → [send d(mb+1), recv d_res(mb), B(mb)] × (M-2..0) → send d(0)
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
#include "mailbox.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u
#define MAX_T        8
#define N_STEPS      20
#define N_MB         4      /* microbatches per step */
#define LR           (0.001f / N_MB)   /* scale lr by 1/M for same effective step */
#define RANK         1

/* Rank 1 uses low bank */
#define D_BASE  4
#define CLK_PIN 12
#define ACK_PIN 13

/* Extra bump allocator for additional activation sets */
static unsigned extra_base, extra_off;
static void *extra_alloc(unsigned bytes) {
    void *p = (void *)(extra_base + extra_off);
    extra_off += (bytes + 15u) & ~15u;
    return p;
}

void notmain(void) {
    mmu_init_and_enable();

    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, MAX_T, ARENA_SIZE);
    pt_print_config(&ctx);

    int n_layers = ctx.cfg.n_layers;
    int split = n_layers / 2;
    int dim = ctx.cfg.dim;
    unsigned act_bytes = MAX_T * dim * sizeof(float);

    printk("microbatch pipeline: rank 1, layers %d..%d, M=%d\n", split, n_layers, N_MB);
    printk("  %d steps, lr=%d.%06d, act_bytes=%d\n",
           N_STEPS, (int)LR, (int)(LR * 1000000) % 1000000, act_bytes);

    /* Allocate M-1 extra activation sets */
    unsigned wt_size = ctx.cfg.vocab_size * ctx.cfg.dim;
    unsigned ht_size = ctx.cfg.hidden_dim * ctx.cfg.dim;
    unsigned tb_elems = wt_size > ht_size ? wt_size : ht_size;
    extra_base = ((unsigned)ctx.bb.w_transpose + tb_elems * 4 + 0xFFFFF) & ~0xFFFFF;
    extra_off = 0;

    pt_activations_t acts[N_MB];
    acts[0] = ctx.acts;  /* reuse the one from pt_pi_init */
    for (int m = 1; m < N_MB; m++)
        pt_scratch_alloc_activations(&acts[m], &ctx.cfg, MAX_T, extra_alloc);

    unsigned extra_end = extra_base + extra_off;
    printk("  extra memory: 0x%x-0x%x (%d KB)\n",
           extra_base, extra_end, extra_off / 1024);
    if (extra_end > arm_ram_end())
        panic("extra allocations exceed RAM\n");

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

        /* ═══════════════════════════════════════════════════
         * FORWARD PHASE: overlapped pipeline fill
         *
         * Schedule (M=4):
         *   recv act(0) → F(0)
         *   send res(0), recv act(1) → F(1)
         *   send res(1), recv act(2) → F(2)
         *   send res(2), recv act(3) → F(3)
         *   send res(3)
         * ═══════════════════════════════════════════════════ */

        /* Recv act(0), compute F(0) */
        float *act_in_0 = acts[0].residuals + split * MAX_T * dim;
        if (gpio.base.recv_raw(&gpio.base, act_in_0, act_bytes) < 0) {
            printk("FAIL: recv fwd act(0) at step %d\n", step);
            clean_reboot();
        }

        pt_forward_train_layers_range(&ctx.cfg, &ctx.w, &acts[0], MAX_T,
                                       split, n_layers, ctx.matvec, ctx.trace);

        /* F(1..M-1): send previous result, recv next, compute */
        for (int mb = 1; mb < N_MB; mb++) {
            /* Send result(mb-1) back to R0 */
            float *act_out_prev = acts[mb-1].residuals + n_layers * MAX_T * dim;
            gpio.base.send_raw(&gpio.base, act_out_prev, act_bytes);

            /* Recv act(mb) from R0 */
            float *act_in = acts[mb].residuals + split * MAX_T * dim;
            if (gpio.base.recv_raw(&gpio.base, act_in, act_bytes) < 0) {
                printk("FAIL: recv fwd act(%d) at step %d\n", mb, step);
                clean_reboot();
            }

            /* Compute F(mb) */
            pt_forward_train_layers_range(&ctx.cfg, &ctx.w, &acts[mb], MAX_T,
                                           split, n_layers, ctx.matvec, ctx.trace);
        }

        /* Send last result */
        float *act_out_last = acts[N_MB-1].residuals + n_layers * MAX_T * dim;
        gpio.base.send_raw(&gpio.base, act_out_last, act_bytes);

        uint32_t t_fwd = timer_get_usec();

        /* ═══════════════════════════════════════════════════
         * BACKWARD PHASE: overlapped pipeline drain
         *
         * Schedule (M=4):
         *   recv d_res(3) → B(3)
         *   send d(3), recv d_res(2) → B(2)
         *   send d(2), recv d_res(1) → B(1)
         *   send d(1), recv d_res(0) → B(0)
         *   send d(0)
         * ═══════════════════════════════════════════════════ */

        /* Recv d_res(M-1), compute B(M-1) */
        if (gpio.base.recv_raw(&gpio.base, ctx.bb.d_res, act_bytes) < 0) {
            printk("FAIL: recv bwd d_res(%d) at step %d\n", N_MB-1, step);
            clean_reboot();
        }

        pt_backward_layers_range(&ctx.cfg, &ctx.w, &ctx.grads, &acts[N_MB-1],
                                  &ctx.bb, MAX_T, split, n_layers,
                                  ctx.matvec, ctx.trace);

        /* B(M-2..0): send previous d_res, recv next, compute */
        for (int mb = N_MB - 2; mb >= 0; mb--) {
            /* Send d_res result from B(mb+1) back to R0 */
            gpio.base.send_raw(&gpio.base, ctx.bb.d_res, act_bytes);

            /* Recv d_res(mb) from R0 */
            if (gpio.base.recv_raw(&gpio.base, ctx.bb.d_res, act_bytes) < 0) {
                printk("FAIL: recv bwd d_res(%d) at step %d\n", mb, step);
                clean_reboot();
            }

            /* Compute B(mb) */
            pt_backward_layers_range(&ctx.cfg, &ctx.w, &ctx.grads, &acts[mb],
                                      &ctx.bb, MAX_T, split, n_layers,
                                      ctx.matvec, ctx.trace);
        }

        /* Send d_res(0) back to R0 */
        gpio.base.send_raw(&gpio.base, ctx.bb.d_res, act_bytes);

        uint32_t t_bwd = timer_get_usec();

        /* ═══ SGD ═══ */
        pt_sgd_update_layers(&ctx.w, &ctx.grads, LR, &ctx.cfg, split, n_layers);

        uint32_t t_sgd = timer_get_usec();

        printk("step %d: fwd=%d bwd=%d sgd=%d total=%d us\n",
               step,
               t_fwd - t0, t_bwd - t_fwd, t_sgd - t_bwd, t_sgd - t0);
    }

    printk("=== rank 1 DONE ===\n");
    clean_reboot();
}
