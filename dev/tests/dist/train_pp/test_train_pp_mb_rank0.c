/*
 * GPipe-style microbatched pipeline training: rank 0 (coordinator).
 *
 * Processes M microbatches per step with gradient accumulation.
 * Overlapped schedule: R0 computes F(mb+1) while R1 computes F(mb),
 * and R0 computes head_bwd(mb-1) while R1 computes B(mb).
 *
 * Rank 0 owns: embedding, layers 0..SPLIT, final rmsnorm, classifier.
 * Rank 1 owns: layers SPLIT..N.
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
#define RANK         0

/* Rank 0 uses high bank */
#define D_BASE  16
#define CLK_PIN 24
#define ACK_PIN 25

/* All microbatches use the same target (validates gradient accumulation) */
static int target[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

/* Extra bump allocator for additional activation sets + d_res saves */
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

    printk("microbatch pipeline: rank 0, layers 0..%d + head, M=%d\n", split, N_MB);
    printk("  %d steps, lr=%d.%06d, act_bytes=%d\n",
           N_STEPS, (int)LR, (int)(LR * 1000000) % 1000000, act_bytes);

    /* Allocate M-1 extra activation sets + d_res save buffers */
    unsigned wt_size = ctx.cfg.vocab_size * ctx.cfg.dim;
    unsigned ht_size = ctx.cfg.hidden_dim * ctx.cfg.dim;
    unsigned tb_elems = wt_size > ht_size ? wt_size : ht_size;
    extra_base = ((unsigned)ctx.bb.w_transpose + tb_elems * 4 + 0xFFFFF) & ~0xFFFFF;
    extra_off = 0;

    pt_activations_t acts[N_MB];
    acts[0] = ctx.acts;  /* reuse the one from pt_pi_init */
    for (int m = 1; m < N_MB; m++)
        pt_scratch_alloc_activations(&acts[m], &ctx.cfg, MAX_T, extra_alloc);

    /* Save buffers for d_res during overlapped backward (T*dim each) */
    float *d_res_save[N_MB];
    for (int m = 0; m < N_MB; m++)
        d_res_save[m] = (float *)extra_alloc(act_bytes);

    unsigned extra_end = extra_base + extra_off;
    printk("  extra memory: 0x%x-0x%x (%d KB)\n",
           extra_base, extra_end, extra_off / 1024);
    if (extra_end > arm_ram_end())
        panic("extra allocations exceed RAM\n");

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

        /* ═══════════════════════════════════════════════════
         * FORWARD PHASE: overlapped pipeline fill
         *
         * Schedule (M=4):
         *   R0: F(0)  →  send act(0)
         *   R0: F(1)  ‖  R1: F(0)  →  recv/send
         *   R0: F(2)  ‖  R1: F(1)  →  recv/send
         *   R0: F(3)  ‖  R1: F(2)  →  recv/send
         *              ‖  R1: F(3)  →  recv
         *   R0: head(0..3)
         * ═══════════════════════════════════════════════════ */

        /* F(0): embed + layers 0..split, send to R1 */
        pt_forward_train_embed(&ctx.w, &acts[0], target, MAX_T, dim);
        pt_forward_train_layers_range(&ctx.cfg, &ctx.w, &acts[0], MAX_T,
                                       0, split, ctx.matvec, ctx.trace);
        gpio.base.send_raw(&gpio.base,
                           acts[0].residuals + split * MAX_T * dim, act_bytes);
        /* Release bus so R1 can send when ready (prevents CLK collision) */
        gpio.base.recv_raw(&gpio.base, NULL, 0);

        /* F(1..M-1): R0 computes F(mb) while R1 computes F(mb-1) */
        for (int mb = 1; mb < N_MB; mb++) {
            /* Compute next microbatch (R1 working on previous in parallel) */
            pt_forward_train_embed(&ctx.w, &acts[mb], target, MAX_T, dim);
            pt_forward_train_layers_range(&ctx.cfg, &ctx.w, &acts[mb], MAX_T,
                                           0, split, ctx.matvec, ctx.trace);

            /* R1 finished F(mb-1), receive result */
            if (gpio.base.recv_raw(&gpio.base,
                    acts[mb-1].residuals + n_layers * MAX_T * dim,
                    act_bytes) < 0) {
                printk("FAIL: recv fwd act(%d)\n", mb-1);
                clean_reboot();
            }

            /* Send F(mb) to R1 */
            gpio.base.send_raw(&gpio.base,
                               acts[mb].residuals + split * MAX_T * dim, act_bytes);
            /* Release bus so R1 can send when ready */
            gpio.base.recv_raw(&gpio.base, NULL, 0);
        }

        /* Receive last result from R1 */
        if (gpio.base.recv_raw(&gpio.base,
                acts[N_MB-1].residuals + n_layers * MAX_T * dim,
                act_bytes) < 0) {
            printk("FAIL: recv fwd act(%d)\n", N_MB-1);
            clean_reboot();
        }

        uint32_t t_fwd = timer_get_usec();

        /* Head forward for all microbatches (sequential, R1 idle) */
        float total_loss = 0.0f;
        for (int mb = 0; mb < N_MB; mb++) {
            float loss = pt_forward_train_head(&ctx.cfg, &ctx.w, &acts[mb],
                                                target, MAX_T, ctx.matvec, ctx.trace);
            total_loss += loss;
        }
        float avg_loss = total_loss / N_MB;

        uint32_t t_head_fwd = timer_get_usec();

        /* ═══════════════════════════════════════════════════
         * BACKWARD PHASE: overlapped pipeline drain
         *
         * Schedule (M=4):
         *   R0: head_bwd(3)  →  send d_res(3)
         *   R0: head_bwd(2)  ‖  R1: B(3)  →  recv/layer_bwd(3)/send
         *   R0: head_bwd(1)  ‖  R1: B(2)  →  recv/layer_bwd(2)/send
         *   R0: head_bwd(0)  ‖  R1: B(1)  →  recv/layer_bwd(1)/send
         *                    ‖  R1: B(0)  →  recv/layer_bwd(0)
         *   R0: embed_bwd(0..3), SGD
         * ═══════════════════════════════════════════════════ */

        /* head_bwd(M-1), send d_res to R1 */
        pt_backward_head(&ctx.cfg, &ctx.w, &ctx.grads, &acts[N_MB-1],
                         &ctx.bb, target, MAX_T, ctx.matvec, ctx.trace);
        memcpy(d_res_save[N_MB-1], ctx.bb.d_res, act_bytes);
        gpio.base.send_raw(&gpio.base, ctx.bb.d_res, act_bytes);
        /* Release bus so R1 can send when ready */
        gpio.base.recv_raw(&gpio.base, NULL, 0);

        /* head_bwd(M-2..0): overlapped with R1's layer backward */
        for (int mb = N_MB - 2; mb >= 0; mb--) {
            /* Compute head backward for mb (R1 working on B(mb+1) in parallel) */
            pt_backward_head(&ctx.cfg, &ctx.w, &ctx.grads, &acts[mb],
                             &ctx.bb, target, MAX_T, ctx.matvec, ctx.trace);
            memcpy(d_res_save[mb], ctx.bb.d_res, act_bytes);

            /* R1 finished B(mb+1), receive d_res result */
            if (gpio.base.recv_raw(&gpio.base, ctx.bb.d_res, act_bytes) < 0) {
                printk("FAIL: recv bwd d_res(%d)\n", mb+1);
                clean_reboot();
            }

            /* Layer backward for mb+1 with received d_res */
            pt_backward_layers_range(&ctx.cfg, &ctx.w, &ctx.grads, &acts[mb+1],
                                      &ctx.bb, MAX_T, 0, split,
                                      ctx.matvec, ctx.trace);
            pt_backward_embed(&ctx.grads, &ctx.bb, target, MAX_T, dim);

            /* Send saved d_res for mb to R1 */
            gpio.base.send_raw(&gpio.base, d_res_save[mb], act_bytes);
            /* Release bus so R1 can send when ready */
            gpio.base.recv_raw(&gpio.base, NULL, 0);
        }

        /* Receive d_res(0) from R1 */
        if (gpio.base.recv_raw(&gpio.base, ctx.bb.d_res, act_bytes) < 0) {
            printk("FAIL: recv bwd d_res(0)\n");
            clean_reboot();
        }

        /* Layer backward for mb=0 */
        pt_backward_layers_range(&ctx.cfg, &ctx.w, &ctx.grads, &acts[0],
                                  &ctx.bb, MAX_T, 0, split,
                                  ctx.matvec, ctx.trace);
        pt_backward_embed(&ctx.grads, &ctx.bb, target, MAX_T, dim);

        uint32_t t_bwd = timer_get_usec();

        /* ═══ SGD ═══ */
        pt_sgd_update_head(&ctx.w, &ctx.grads, LR, &ctx.cfg);
        pt_sgd_update_layers(&ctx.w, &ctx.grads, LR, &ctx.cfg, 0, split);

        uint32_t t_sgd = timer_get_usec();

        printk("step %d: loss=%d.%04d | fwd=%d head=%d bwd=%d sgd=%d total=%d us\n",
               step,
               (int)avg_loss, ((int)(avg_loss * 10000)) % 10000,
               t_fwd - t0, t_head_fwd - t_fwd, t_bwd - t_head_fwd,
               t_sgd - t_bwd, t_sgd - t0);
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
