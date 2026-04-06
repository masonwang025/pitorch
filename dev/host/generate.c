/*
 * generate.c — Interactive text generation on Mac (host CPU).
 * Mirrors examples/generate.c but runs without a Pi.
 *
 * usage: ./generate <model.bin> <tokenizer.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "pt.h"
#include "pt_text.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.bin> <tokenizer.bin>\n", argv[0]);
        return 1;
    }

    void *model_data = pt_read_file(argv[1], NULL);
    pt_context_t ctx;
    pt_host_init(&ctx, model_data, 0);
    pt_print_config(&ctx);

    void *tok_data = pt_read_file(argv[2], NULL);
    pt_tokenizer_t tok;
    pt_tokenizer_init(&tok, tok_data, ctx.cfg.vocab_size);

    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, 0.0f, 0.9f, (uint64_t)time(NULL));

    char prompt[256];
    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(prompt, sizeof(prompt), stdin)) break;

        /* strip newline */
        int len = strlen(prompt);
        while (len > 0 && (prompt[len-1] == '\n' || prompt[len-1] == '\r'))
            prompt[--len] = '\0';
        if (len == 0) continue;

        pt_reset_kv(&ctx);
        pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                    prompt, ctx.cfg.seq_len, ctx.matvec);
        printf("\n\n");
    }

    pt_free(&ctx);
    free(model_data);
    free(tok_data);
    return 0;
}
