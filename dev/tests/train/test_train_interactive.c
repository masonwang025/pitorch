/*
 * Interactive training demo on Pi Zero.
 * Type a sentence over UART, watch the Pi fine-tune on it,
 * then generate from a prompt to verify memorization.
 *
 * SD card: initramfs weights/stories15M_full.bin 0x2000000
 *   (combined: model weights + tokenizer.bin)
 *
 * Commands:
 *   train: <text>       fine-tune on this sentence
 *   generate: <prompt>  generate continuation
 */
#include "rpi.h"
#include <string.h>
#include "mmu.h"
#include "pt.h"
#include "pt_text.h"
#include "profiler.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u

#define MAX_SEQ      32
#define MAX_STEPS    100
#define LR           0.001f
#define LOSS_TARGET  0.1f
#define GEN_TOKENS   64

static int read_line(char *buf, int max_len) {
    int i = 0;
    while (i < max_len - 1) {
        int c = uart_get8();
        if (c == '\n' || c == '\r') break;
        if (c == 0x7f || c == '\b') {
            if (i > 0) { i--; uart_put8('\b'); uart_put8(' '); uart_put8('\b'); }
            continue;
        }
        buf[i++] = (char)c;
        uart_put8((uint8_t)c);
    }
    buf[i] = '\0';
    return i;
}

void notmain(void) {
    mmu_init_and_enable();

    /* Load tokenizer before pt_pi_init — state buffers overlap tokenizer
     * region in the combined file. */
    pt_config_t tmp_cfg;
    pt_load_config(&tmp_cfg, (void *)WEIGHT_BASE);

    pt_tokenizer_t tok;
    pt_load_tokenizer(&tok, (void *)WEIGHT_BASE, tmp_cfg.vocab_size);

    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, MAX_SEQ, ARENA_SIZE);

    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, 0.0f, 0.9f, timer_get_usec());

    printk("commands:  train: <text>    generate: <prompt>\n\n");

    char buf[256];
    while (1) {
        printk("> ");
        int len = read_line(buf, sizeof(buf));
        printk("\n");
        if (len == 0) continue;

        if (len > 7 && memcmp(buf, "train: ", 7) == 0) {
            int tokens[MAX_SEQ];
            int n_tokens;
            pt_encode(&tok, buf + 7, 1, 0, tokens, &n_tokens);
            if (n_tokens > MAX_SEQ) n_tokens = MAX_SEQ;
            if (n_tokens < 2) { printk("too short\n"); continue; }
            printk("training on %d tokens\n", n_tokens);

            for (int step = 0; step < MAX_STEPS; step++) {
                unsigned t0 = timer_get_usec();
                float loss = pt_train_step(&ctx, tokens, n_tokens, LR);
                unsigned us = timer_get_usec() - t0;

                if (step < 10 || step % 10 == 0 || loss < LOSS_TARGET) {
                    printk("step %d: loss=", step); pt_pf(loss, 4);
                    printk(" (%ds)\n", us / 1000000);
                }
                if (loss < LOSS_TARGET && step >= 3) {
                    printk("done (%d steps)\n", step + 1);
                    break;
                }
            }

        } else if (len > 10 && memcmp(buf, "generate: ", 10) == 0) {
            pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                        buf + 10, GEN_TOKENS, ctx.matvec);
            printk("\n");

        } else {
            printk("use 'train: <text>' or 'generate: <prompt>'\n");
        }
    }
}
