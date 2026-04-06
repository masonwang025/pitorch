#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama2.h"
#include "pt_ops.h"

#define N_STEPS 5

static void top5(const float *logits, int n, int *idx, float *val) {
    for (int k = 0; k < 5; k++) { idx[k] = -1; val[k] = -1e30f; }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 5; k++) {
            if (logits[i] > val[k]) {
                for (int j = 4; j > k; j--) { idx[j] = idx[j-1]; val[j] = val[j-1]; }
                idx[k] = i;
                val[k] = logits[i];
                break;
            }
        }
    }
}

static int load_expected(const char *path, int out[][5], int max_steps) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int n = 0;
    while (n < max_steps &&
           fscanf(f, "%d %d %d %d %d",
                  &out[n][0], &out[n][1], &out[n][2],
                  &out[n][3], &out[n][4]) == 5)
        n++;
    fclose(f);
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin> [expected.txt]\n", argv[0]);
        return 1;
    }
    const char *model_path    = argv[1];
    const char *expected_path = argc > 2 ? argv[2] : NULL;

    /* load model file into memory */
    FILE *f = fopen(model_path, "rb");
    if (!f) { perror(model_path); return 1; }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *data = malloc(file_size);
    if (!data) { fprintf(stderr, "malloc failed\n"); return 1; }
    if ((long)fread(data, 1, file_size, f) != file_size) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);

    /* config */
    pt_config_t cfg;
    pt_load_config(&cfg, data);
    printf("config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads,
           cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len);

    /* weights */
    pt_weights_t w;
    pt_load_weights(&w, &cfg, data);
    printf("weights loaded: %.1f MB\n", file_size / (1024.0 * 1024.0));
    printf("spot check: emb[0]=%f  wq[0]=%f  w1[0]=%f\n\n",
           w.token_embedding[0], w.wq[0], w.w1[0]);

    /* state */
    pt_state_t s;
    pt_alloc_state(&s, &cfg);

    /* load expected top-5 indices (from reference.py) */
    int expected[N_STEPS][5];
    int n_expected = 0;
    if (expected_path)
        n_expected = load_expected(expected_path, expected, N_STEPS);

    /* autoregressive inference */
    int token = 1;  /* BOS */
    int n_match = 0;

    for (int step = 0; step < N_STEPS; step++) {
        pt_forward(&cfg, &w, &s, token, step, smatvec_cpu);

        int t5[5]; float v5[5];
        top5(s.logits, cfg.vocab_size, t5, v5);

        printf("step %d: token=%-5d -> top5: [%d, %d, %d, %d, %d]",
               step, token, t5[0], t5[1], t5[2], t5[3], t5[4]);

        if (step < n_expected) {
            int match = 1;
            for (int k = 0; k < 5; k++)
                if (t5[k] != expected[step][k]) { match = 0; break; }
            printf(match ? "  MATCH" : "  MISMATCH (expected [%d, %d, %d, %d, %d])",
                   expected[step][0], expected[step][1], expected[step][2],
                   expected[step][3], expected[step][4]);
            n_match += match;
        }
        printf("\n");
        token = t5[0];  /* greedy argmax */
    }

    if (n_expected > 0)
        printf("\n%d/%d steps match PyTorch reference.\n", n_match, N_STEPS);

    pt_free_state(&s);
    free(data);
    return (n_expected > 0 && n_match == N_STEPS) ? 0 : 1;
}
