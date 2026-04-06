/*
 * Interactive train + generate on Mac.
 * Same flow as the Pi demo but runs natively.
 *
 * usage: ./test_train_interactive <model.bin> <tokenizer.bin>
 *
 * Commands:
 *   train: <text>       fine-tune on this sentence
 *   generate: <prompt>  generate continuation
 *   quit                exit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "pt.h"
#include "pt_text.h"
#include "pt_ops.h"

#define MAX_SEQ      32
#define MAX_STEPS    100
#define LR           0.001f
#define LOSS_TARGET  0.1f
#define GEN_TOKENS   64

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.bin> <tokenizer.bin>\n", argv[0]);
        return 1;
    }

    void *model_data = pt_read_file(argv[1], NULL);

    pt_context_t ctx;
    pt_host_init(&ctx, model_data, MAX_SEQ);
    pt_print_config(&ctx);

    void *tok_data = pt_read_file(argv[2], NULL);
    pt_tokenizer_t tok;
    pt_tokenizer_init(&tok, tok_data, ctx.cfg.vocab_size);

    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, 0.0f, 0.9f, 42);

    printf("commands:  train: <text>    generate: <prompt>    quit\n\n");

    char buf[512];
    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(buf, sizeof(buf), stdin)) break;

        int len = (int)strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'))
            buf[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(buf, "quit") == 0) break;

        if (len > 7 && memcmp(buf, "train: ", 7) == 0) {
            const char *text = buf + 7;
            printf("training on: \"%s\"\n", text);

            int tokens[MAX_SEQ];
            int n_tokens;
            pt_encode(&tok, text, 1, 0, tokens, &n_tokens);
            if (n_tokens > MAX_SEQ) n_tokens = MAX_SEQ;
            if (n_tokens < 2) { printf("too short\n"); continue; }

            for (int step = 0; step < MAX_STEPS; step++) {
                clock_t t0 = clock();
                float loss = pt_train_step(&ctx, tokens, n_tokens, LR);
                double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

                if (step < 10 || step % 10 == 0 || loss < LOSS_TARGET)
                    printf("step %-3d: loss=%.6f  (%.2fs)\n",
                           step, loss, elapsed);

                if (loss < LOSS_TARGET && step >= 3) {
                    printf("done (%d steps)\n\n", step + 1);
                    break;
                }
            }

        } else if (len > 10 && memcmp(buf, "generate: ", 10) == 0) {
            pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                        buf + 10, GEN_TOKENS, ctx.matvec);
            printf("\n\n");

        } else {
            printf("unknown command. use 'train: <text>' or 'generate: <prompt>'\n");
        }
    }

    pt_free(&ctx);
    free(model_data);
    free(tok_data);
    return 0;
}
