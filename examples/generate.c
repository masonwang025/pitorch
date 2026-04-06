/*
 * generate.c — Interactive text generation on a single Pi Zero.
 *
 * Type a prompt over UART, get a story back. Tokens stream in real-time
 * at ~12 tok/s (stories15M with D-cache enabled).
 *
 * ── How to run ──────────────────────────────────────────────────────
 *
 *   cd examples && ./run.sh generate           # deploy to Pi 0 (default)
 *   cd examples && PI_DEVICE=2 ./run.sh generate   # deploy to Pi 2
 *
 * ── SD card ─────────────────────────────────────────────────────────
 *
 *   initramfs weights/stories15M_full.bin 0x2000000
 *   (combined file: model weights + tokenizer)
 *
 * ════════════════════════════════════════════════════════════════════
 */

#include "rpi.h"
#include "mmu.h"
#include "pt.h"
#include "pt_text.h"

#define WEIGHT_ADDR  ((void *)0x02000000)
#define NUM_QPUS     12
#define ARENA_SIZE   (1 * 1024 * 1024)
#define MAX_TOKENS   256
#define TEMPERATURE  0.0f
#define TOPP         0.9f

/* Read a line from UART with backspace support. */
static int read_prompt(char *buf, int max_len) {
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

    /* Load tokenizer before pt_pi_init (state buffers may overlap tokenizer region) */
    pt_config_t tmp_cfg;
    pt_load_config(&tmp_cfg, WEIGHT_ADDR);

    pt_tokenizer_t tok;
    pt_load_tokenizer(&tok, WEIGHT_ADDR, tmp_cfg.vocab_size);

    /* Initialize model + GPU */
    pt_context_t ctx;
    pt_pi_init(&ctx, WEIGHT_ADDR, NUM_QPUS, 0, ARENA_SIZE);

    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, TEMPERATURE, TOPP,
                    timer_get_usec());

    /* Interactive generation loop */
    char prompt[256];
    while (1) {
        printk("> ");
        int len = read_prompt(prompt, sizeof(prompt));
        if (len == 0) { printk("\n"); continue; }

        printk("\n...");
        pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                    prompt, MAX_TOKENS, ctx.matvec);
        printk("\n\n");
    }
}
