/*
 * Host-side generation verification.
 * Runs greedy generation on Mac CPU to verify the full pipeline
 * (encode -> forward -> sample -> decode) produces correct output.
 *
 * usage: ./test_generate_host <model.bin> <tokenizer.bin> [prompt]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pt.h"
#include "pt_text.h"
#include "pt_ops.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.bin> <tokenizer.bin> [prompt]\n", argv[0]);
        return 1;
    }

    void *model_data = pt_read_file(argv[1], NULL);

    pt_context_t ctx;
    pt_host_init(&ctx, model_data, 0);  /* inference only */
    pt_print_config(&ctx);

    void *tok_data = pt_read_file(argv[2], NULL);
    pt_tokenizer_t tok;
    pt_tokenizer_init(&tok, tok_data, ctx.cfg.vocab_size);
    printf("tokenizer: %d tokens\n\n", tok.vocab_size);

    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, 0.0f, 0.9f, 42);

    const char *prompt = argc > 3 ? argv[3] : "Once upon a time";
    printf("prompt: \"%s\"\n", prompt);

    /* encode */
    int prompt_tokens[512];
    int n_prompt;
    pt_encode(&tok, prompt, 1, 0, prompt_tokens, &n_prompt);
    printf("encoded: %d tokens [", n_prompt);
    for (int i = 0; i < n_prompt; i++)
        printf("%d%s", prompt_tokens[i], i < n_prompt-1 ? ", " : "");
    printf("]\n\n");

    /* generate (greedy, 20 tokens) */
    int max_tokens = 20;
    if (max_tokens > ctx.cfg.seq_len) max_tokens = ctx.cfg.seq_len;

    pt_reset_kv(&ctx);

    int token = prompt_tokens[0];
    printf("generating %d tokens (greedy, CPU)...\n%s", max_tokens, prompt);

    int all_tokens[256];
    int n_total = 0;

    for (int pos = 0; pos < max_tokens; pos++) {
        pt_forward(&ctx.cfg, &ctx.w, &ctx.state, token, pos, ctx.matvec);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = argmax(ctx.state.logits, ctx.cfg.vocab_size);
            if (next == 1 || next == 2) break;
            printf("%s", pt_decode(&tok, token, next));
        }

        all_tokens[n_total++] = next;
        token = next;
    }
    printf("\n\n");

    /* verify first 5 greedy tokens from BOS match Phase 3 reference */
    int ref[] = {9038, 2501, 263, 931, 29892};
    if (n_prompt == 5 && prompt_tokens[1] == 9038) {
        printf("verifying Phase 3 greedy match:\n");
        int ok = 1;
        for (int i = 0; i < 5 && i < n_total; i++) {
            int match = (all_tokens[i] == ref[i]);
            printf("  step %d: got %d, expected %d %s\n",
                   i, all_tokens[i], ref[i], match ? "MATCH" : "MISMATCH");
            if (!match) ok = 0;
        }
        printf(ok ? "\nALL MATCH\n" : "\nFAILED\n");
    }

    printf("\nall generated tokens: [");
    for (int i = 0; i < n_total; i++)
        printf("%d%s", all_tokens[i], i < n_total-1 ? ", " : "");
    printf("]\n");

    pt_free(&ctx);
    free(model_data);
    free(tok_data);
    return 0;
}
