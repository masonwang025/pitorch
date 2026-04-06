/*
 * Data-parallel training: rank 1.
 *
 * Same as rank 0 but trains on a different target sequence.
 * After allreduce, weights are identical on both ranks.
 *
 * SD card: initramfs weights/<model>.bin 0x2000000
 */
#include "rpi.h"
#include "pt.h"
#include "pt_ops.h"
#include "pt_link_gpio.h"
#include "pt_proto.h"
#include "pt_allreduce.h"
#include "profiler.h"
#include "mmu.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u
#define MAX_T        8
#define N_STEPS      3
#define LR           0.001f
#define RANK         1

/* Rank 1 uses low bank */
#define D_BASE  4
#define CLK_PIN 12
#define ACK_PIN 13

/* Rank 1 trains on this target sequence (different from rank 0) */
static int target_b[] = { 1, 9038, 2501, 263, 931, 29892, 263, 365 };

void notmain(void) {
    mmu_init_and_enable();

    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, MAX_T, ARENA_SIZE);
    pt_print_config(&ctx);

    printk("data-parallel training: rank %d, %d steps, lr=%d.%03d\n",
           RANK, N_STEPS, (int)LR, (int)(LR * 1000) % 1000);
    printk("target: ");
    for (int i = 0; i < MAX_T; i++) printk("%d ", target_b[i]);
    printk("\n");
    printk("grad buffer: %d params (%d KB)\n",
           ctx.grads._n_params, ctx.grads._n_params * 4 / 1024);

    /* Init GPIO link */
    pt_link_gpio_t gpio = pt_link_gpio_init(D_BASE, CLK_PIN, ACK_PIN);
    printk("gpio link ready\n");

    /* Wait for PING from rank 0 */
    uint32_t opcode, plen, dummy = 0;
    printk("waiting for PING...\n");
    if (pt_proto_recv(&gpio.base, &opcode, &dummy, sizeof(dummy), &plen) < 0 ||
        opcode != PT_OP_PING) {
        printk("FAIL: no PING\n");
        clean_reboot();
    }
    printk("got PING, sending PONG...\n");
    pt_proto_send(&gpio.base, PT_OP_PONG, &dummy, sizeof(dummy));
    printk("handshake done!\n");

    /* Warm-up: small allreduce to verify bus works */
    {
        float test_buf[1024];
        for (int i = 0; i < 1024; i++) test_buf[i] = (float)(i + RANK * 1000);
        printk("warm-up allreduce (1024 floats)...\n");
        int rc = pt_allreduce_avg(RANK, test_buf, 1024, &gpio.base);
        if (rc < 0) {
            printk("FAIL: warm-up allreduce timeout\n");
            clean_reboot();
        }
        printk("warm-up: buf[0]=%d buf[1]=%d (expect 500, 500)\n",
               (int)test_buf[0], (int)test_buf[1]);
    }
    delay_ms(100);

    /* Training loop */
    for (int step = 0; step < N_STEPS; step++) {
        uint32_t t0 = timer_get_usec();

        /* Local forward + backward */
        pt_zero_grads(&ctx.grads);
        float loss = pt_forward_train(&ctx.cfg, &ctx.w, &ctx.acts,
                                      target_b, MAX_T, ctx.matvec, ctx.trace);
        pt_backward(&ctx.cfg, &ctx.w, &ctx.grads, &ctx.acts, &ctx.bb,
                    target_b, MAX_T, ctx.matvec, ctx.trace);

        uint32_t t1 = timer_get_usec();

        /* Allreduce gradients (average with rank 0) */
        int rc = pt_allreduce_avg(RANK, ctx.grads._mem, ctx.grads._n_params,
                                  &gpio.base);
        if (rc < 0) {
            printk("FAIL: allreduce timeout at step %d\n", step);
            clean_reboot();
        }

        uint32_t t2 = timer_get_usec();

        /* SGD update with averaged gradients */
        pt_sgd_update(&ctx.w, &ctx.grads, LR, &ctx.cfg);

        uint32_t t3 = timer_get_usec();

        printk("step %d: loss=%d.%04d | compute=%d allreduce=%d sgd=%d total=%d us\n",
               step,
               (int)loss, ((int)(loss * 10000)) % 10000,
               t1 - t0, t2 - t1, t3 - t2, t3 - t0);
    }

    /* Verify: greedy decode from BOS (same check as rank 0 but for target_b) */
    printk("verifying greedy decode...\n");
    pt_reset_kv(&ctx);
    int token = target_b[0];
    int match = 0;
    for (int t = 0; t < MAX_T - 1; t++) {
        token = pt_forward_step(&ctx, token);
        if (token == target_b[t + 1]) match++;
        printk("  pos %d: got %d expected %d %s\n",
               t, token, target_b[t + 1],
               token == target_b[t + 1] ? "OK" : "MISS");
    }

    printk("matched %d/%d target tokens\n", match, MAX_T - 1);
    printk("=== rank 1 DONE ===\n");
    clean_reboot();
}
