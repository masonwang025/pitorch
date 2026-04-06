#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama2.h"
#include "pt_ops.h"

#define N_STEPS 5
#define SPLIT   3   /* layers 0..SPLIT-1 on rank 0, SPLIT..n_layers-1 on rank 1 */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    /* load model */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *data = malloc(file_size);
    if ((long)fread(data, 1, file_size, f) != file_size) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);

    pt_config_t cfg;
    pt_load_config(&cfg, data);
    pt_weights_t w;
    pt_load_weights(&w, &cfg, data);

    printf("model: dim=%d layers=%d split=%d\n", cfg.dim, cfg.n_layers, SPLIT);

    /* two independent states: monolithic and staged */
    pt_state_t s_ref, s_stg;
    pt_alloc_state(&s_ref, &cfg);
    pt_alloc_state(&s_stg, &cfg);

    int token_ref = 1, token_stg = 1;  /* BOS */
    int all_match = 1;

    for (int step = 0; step < N_STEPS; step++) {
        /* --- monolithic reference --- */
        pt_forward(&cfg, &w, &s_ref, token_ref, step, smatvec_cpu);
        int ref_tok = argmax(s_ref.logits, cfg.vocab_size);

        /* --- staged: embed + layers[0,SPLIT) + layers[SPLIT,n_layers) + head --- */
        pt_forward_embed(&w, s_stg.x, cfg.dim, token_stg);
        pt_forward_layers_range(&cfg, &w, &s_stg, step, 0, SPLIT, smatvec_cpu);

        /* simulate network boundary: copy activation */
        float *boundary = (float *)malloc(cfg.dim * sizeof(float));
        memcpy(boundary, s_stg.x, cfg.dim * sizeof(float));
        memcpy(s_stg.x, boundary, cfg.dim * sizeof(float));
        free(boundary);

        pt_forward_layers_range(&cfg, &w, &s_stg, step, SPLIT, cfg.n_layers, smatvec_cpu);
        pt_forward_head(&cfg, &w, &s_stg, smatvec_cpu);
        int stg_tok = argmax(s_stg.logits, cfg.vocab_size);

        /* compare logits bitwise */
        int logits_match = (memcmp(s_ref.logits, s_stg.logits,
                                   cfg.vocab_size * sizeof(float)) == 0);

        printf("step %d: ref=%d stg=%d logits=%s\n",
               step, ref_tok, stg_tok,
               logits_match ? "MATCH" : "MISMATCH");

        if (!logits_match) {
            all_match = 0;
            /* show first divergence */
            for (int i = 0; i < cfg.vocab_size; i++) {
                if (s_ref.logits[i] != s_stg.logits[i]) {
                    printf("  first diff at logit[%d]: ref=%f stg=%f\n",
                           i, s_ref.logits[i], s_stg.logits[i]);
                    break;
                }
            }
        }

        token_ref = ref_tok;
        token_stg = stg_tok;
    }

    printf("\n%s\n", all_match ? "=== PASS ===" : "=== FAIL ===");

    pt_free_state(&s_ref);
    pt_free_state(&s_stg);
    free(data);
    return all_match ? 0 : 1;
}
