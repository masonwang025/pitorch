/*
 * Interactive text generation on Pi Zero.
 * Reads prompt over UART, tokenizes, prefills, decodes, streams text back.
 *
 * SD card: initramfs weights/stories15M_full.bin 0x2000000
 *   (combined file: model weights followed by tokenizer.bin)
 */
#include "rpi.h"
#include "mmu.h"
#include "pt.h"
#include "pt_text.h"
#include "profiler.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (1 * 1024 * 1024)
#define MAX_TOKENS   256
#define WEIGHT_BASE  0x02000000u

#define TEMPERATURE  0.0f
#define TOPP         0.9f

static int read_prompt(char *buf, int max_len) {
    int i = 0;
    while (i < max_len - 1) {
        int c = uart_get8();
        if (c == '\n' || c == '\r') break;
        if (c == 0x7f || c == '\b') {
            if (i > 0) {
                i--;
                uart_put8('\b'); uart_put8(' '); uart_put8('\b');
            }
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

    /* Load tokenizer BEFORE pt_pi_init — state buffers overlap the tokenizer
     * region in the combined file.  Tokenizer copies into static BSS pools,
     * so it's safe to read first; pt_pi_init can then freely overwrite. */
    pt_config_t tmp_cfg;
    pt_load_config(&tmp_cfg, (void *)WEIGHT_BASE);

    pt_tokenizer_t tok;
    pt_load_tokenizer(&tok, (void *)WEIGHT_BASE, tmp_cfg.vocab_size);

    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, 0, ARENA_SIZE);

    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, TEMPERATURE, TOPP,
                    timer_get_usec());

    char prompt[256];
    while (1) {
        printk("> ");
        int len = read_prompt(prompt, sizeof(prompt));
        printk("\n");
        if (len == 0) continue;

        pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                    prompt, MAX_TOKENS, ctx.matvec);
    }
}
