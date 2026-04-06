/*
 * Pipeline-parallel training: rank 0 (coordinator).
 *
 * Rank 0 owns: embedding, layers 0..SPLIT, final rmsnorm, classifier.
 * Rank 1 owns: layers SPLIT..N.
 *
 * Forward:  embed → layers 0..SPLIT → send act → recv act → head (loss)
 * Backward: head bwd → send d_res → recv d_res → layers 0..SPLIT bwd → embed bwd
 * SGD:      update embedding + layers 0..SPLIT + head weights
 *
 * Communication per step: 4 × T × dim × 4 bytes = 36 KB for stories15M T=8.
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
#define RANK         0

/* Rank 0 uses high bank */
#define D_BASE  16
#define CLK_PIN 24
#define ACK_PIN 25

/* Rank 0 trains on this target sequence */
static int target[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

void notmain(void) {
    mmu_init_and_enable();

    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, MAX_T, ARENA_SIZE);
    pt_print_config(&ctx);

    int n_layers = ctx.cfg.n_layers;
    int split = n_layers / 2;  /* 3 for stories15M */
    int dim = ctx.cfg.dim;
    unsigned act_bytes = MAX_T * dim * sizeof(float);

    printk("pipeline training: rank 0, layers 0..%d + head\n", split);
    printk("  %d steps, lr=%d.%03d, act_bytes=%d\n",
           N_STEPS, (int)LR, (int)(LR * 1000) % 1000, act_bytes);

    /* Init GPIO link */
    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);

    /* Handshake */
    delay_ms(1000);
    uint32_t opcode, plen, dummy = 0;
    printk("sending PING...\n");
    pt_proto_send(&gpio.base, PT_OP_PING, &dummy, sizeof(dummy));

    if (pt_proto_recv(&gpio.base, &opcode, &dummy, sizeof(dummy), &plen) < 0 ||
        opcode != PT_OP_PONG) {
        printk("FAIL: no PONG\n");
        clean_reboot();
    }
    printk("rank 1 connected!\n");
    delay_ms(100);

    /* Training loop */
    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0 = timer_get_usec();

        pt_zero_grads(&ctx.grads);

        /* ── Forward ── */

        /* Embedding */
        pt_forward_train_embed(&ctx.w, &ctx.acts, target, MAX_T, dim);

        /* Layers 0..split */
        pt_forward_train_layers_range(&ctx.cfg, &ctx.w, &ctx.acts, MAX_T,
                                       0, split, ctx.matvec, ctx.trace);

        uint32_t t_fwd0 = timer_get_usec();

        /* Send activations[split] to rank 1: T*dim floats */
        float *act_out = ctx.acts.residuals + split * MAX_T * dim;
        gpio.base.send_raw(&gpio.base, act_out, act_bytes);

        /* Recv activations[n_layers] from rank 1 */
        float *act_in = ctx.acts.residuals + n_layers * MAX_T * dim;
        if (gpio.base.recv_raw(&gpio.base, act_in, act_bytes) < 0) {
            printk("FAIL: recv fwd act at step %d\n", step);
            clean_reboot();
        }

        uint32_t t_comm_fwd = timer_get_usec();

        /* Head: final rmsnorm + classifier + loss */
        float loss = pt_forward_train_head(&ctx.cfg, &ctx.w, &ctx.acts,
                                            target, MAX_T, ctx.matvec, ctx.trace);

        uint32_t t_fwd1 = timer_get_usec();

        /* ── Backward ── */

        /* Head backward: classifier + final rmsnorm → d_res */
        pt_backward_head(&ctx.cfg, &ctx.w, &ctx.grads, &ctx.acts,
                         &ctx.bb, target, MAX_T, ctx.matvec, ctx.trace);

        uint32_t t_bwd_head = timer_get_usec();

        /* Send d_res to rank 1 (gradient flowing into layer n_layers-1) */
        gpio.base.send_raw(&gpio.base, ctx.bb.d_res, act_bytes);

        /* Recv d_res back from rank 1 (gradient flowing out of layer split) */
        if (gpio.base.recv_raw(&gpio.base, ctx.bb.d_res, act_bytes) < 0) {
            printk("FAIL: recv bwd d_res at step %d\n", step);
            clean_reboot();
        }

        uint32_t t_comm_bwd = timer_get_usec();

        /* Layers 0..split backward */
        pt_backward_layers_range(&ctx.cfg, &ctx.w, &ctx.grads, &ctx.acts,
                                  &ctx.bb, MAX_T, 0, split,
                                  ctx.matvec, ctx.trace);

        /* Embedding backward */
        pt_backward_embed(&ctx.grads, &ctx.bb, target, MAX_T, dim);

        uint32_t t_bwd0 = timer_get_usec();

        /* ── SGD ── */
        pt_sgd_update_head(&ctx.w, &ctx.grads, LR, &ctx.cfg);
        pt_sgd_update_layers(&ctx.w, &ctx.grads, LR, &ctx.cfg, 0, split);

        uint32_t t_sgd = timer_get_usec();

        printk("step %d: loss=%d.%04d | fwd_r0=%d comm_fwd=%d fwd_head=%d "
               "bwd_head=%d comm_bwd=%d bwd_r0=%d sgd=%d total=%d us\n",
               step,
               (int)loss, ((int)(loss * 10000)) % 10000,
               t_fwd0 - t0, t_comm_fwd - t_fwd0, t_fwd1 - t_comm_fwd,
               t_bwd_head - t_fwd1, t_comm_bwd - t_bwd_head,
               t_bwd0 - t_comm_bwd, t_sgd - t_bwd0, t_sgd - t0);
    }

    /* Verify: greedy decode from BOS */
    printk("verifying greedy decode...\n");
    pt_reset_kv(&ctx);
    int token = target[0];
    int match = 0;
    for (int t = 0; t < MAX_T - 1; t++) {
        token = pt_forward_step(&ctx, token);
        if (token == target[t + 1]) match++;
        printk("  pos %d: got %d expected %d %s\n",
               t, token, target[t + 1],
               token == target[t + 1] ? "OK" : "MISS");
    }

    printk("matched %d/%d target tokens\n", match, MAX_T - 1);
    printk("=== rank 0 DONE ===\n");
    clean_reboot();
}
