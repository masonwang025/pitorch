/*
 * Validate pt_forward_train() against T calls to pt_forward().
 * Compiles and runs natively on Mac.
 *
 * usage: ./test_forward_train <model.bin> [expected.txt]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llama2.h"
#include "pt_ops.h"
#include "pt_train.h"

#define T_SEQ  16

static float fabsf_(float x) { return x < 0 ? -x : x; }

static void top5(const float *logits, int n, int *idx) {
    float val[5];
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin> [expected.txt]\n", argv[0]);
        return 1;
    }

    /* ── load model ── */
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
    printf("config: dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads,
           cfg.n_kv_heads, cfg.vocab_size, cfg.seq_len);

    pt_weights_t w;
    pt_load_weights(&w, &cfg, data);

    int V = cfg.vocab_size;

    /* ── step 1: run inference forward T times to build token sequence ── */
    pt_state_t s;
    pt_alloc_state(&s, &cfg);

    int tokens[T_SEQ];
    tokens[0] = 1;  /* BOS */

    /* save per-position logits from inference path */
    float *ref_logits = (float *)malloc((size_t)T_SEQ * V * sizeof(float));

    printf("\n--- inference forward (reference) ---\n");
    for (int t = 0; t < T_SEQ; t++) {
        pt_forward(&cfg, &w, &s, tokens[t], t, smatvec_cpu);
        memcpy(ref_logits + t * V, s.logits, V * sizeof(float));

        int t5[5];
        top5(s.logits, V, t5);
        printf("pos %2d: token=%-5d -> top5 [%d, %d, %d, %d, %d]\n",
               t, tokens[t], t5[0], t5[1], t5[2], t5[3], t5[4]);

        if (t < T_SEQ - 1)
            tokens[t + 1] = t5[0];  /* greedy next token */
    }

    printf("\ntokens:");
    for (int t = 0; t < T_SEQ; t++) printf(" %d", tokens[t]);
    printf("\n");

    pt_free_state(&s);

    /* ── step 2: run training forward on the same tokens ── */
    pt_activations_t acts;
    pt_alloc_activations(&acts, &cfg, T_SEQ);

    printf("\n--- training forward ---\n");
    float loss = pt_forward_train(&cfg, &w, &acts, tokens, T_SEQ, smatvec_cpu, NULL);
    printf("loss: %.6f\n", loss);

    /* ── step 3: compare logits ── */
    printf("\n--- comparing logits ---\n");
    int all_match = 1;
    for (int t = 0; t < T_SEQ; t++) {
        float *ref = ref_logits + t * V;
        float *got = acts.logits + t * V;

        float max_err = 0.0f;
        int max_err_idx = 0;
        for (int v = 0; v < V; v++) {
            float err = fabsf_(got[v] - ref[v]);
            if (err > max_err) { max_err = err; max_err_idx = v; }
        }

        int t5_ref[5], t5_got[5];
        top5(ref, V, t5_ref);
        top5(got, V, t5_got);

        int top5_match = 1;
        for (int k = 0; k < 5; k++)
            if (t5_ref[k] != t5_got[k]) top5_match = 0;

        printf("pos %2d: max_err=%.2e (idx %d)  top5 %s\n",
               t, max_err, max_err_idx,
               top5_match ? "MATCH" : "MISMATCH");

        if (max_err > 1e-4f || !top5_match) all_match = 0;
    }

    /* ── step 4: verify loss independently ── */
    printf("\n--- verifying loss ---\n");
    float check_loss = 0.0f;
    for (int t = 0; t < T_SEQ - 1; t++) {
        float *lg = ref_logits + t * V;
        int target = tokens[t + 1];

        float max_lg = lg[0];
        for (int v = 1; v < V; v++)
            if (lg[v] > max_lg) max_lg = lg[v];

        double sum = 0.0;
        for (int v = 0; v < V; v++)
            sum += exp((double)(lg[v] - max_lg));

        check_loss += -(lg[target] - max_lg - (float)log(sum));
    }
    check_loss /= (float)(T_SEQ - 1);

    float loss_err = fabsf_(loss - check_loss);
    printf("train loss:     %.6f\n", loss);
    printf("reference loss: %.6f\n", check_loss);
    printf("loss error:     %.2e\n", loss_err);

    int loss_ok = loss_err < 0.01f;
    printf("loss: %s\n", loss_ok ? "MATCH" : "MISMATCH");

    /* ── step 5: load and compare against PyTorch expected.txt if available ── */
    int py_ok = 1;
    if (argc > 2) {
        printf("\n--- PyTorch reference ---\n");
        FILE *ef = fopen(argv[2], "r");
        if (ef) {
            int eT;
            if (fscanf(ef, "%d", &eT) == 1) {
                int etoks[T_SEQ];
                for (int t = 0; t < eT && t < T_SEQ; t++)
                    fscanf(ef, "%d", &etoks[t]);
                float eloss;
                fscanf(ef, "%f", &eloss);
                printf("PyTorch loss: %.6f  (ours: %.6f  err: %.2e)\n",
                       eloss, loss, fabsf_(loss - eloss));
                if (fabsf_(loss - eloss) > 0.05f) py_ok = 0;

                for (int t = 0; t < eT && t < T_SEQ; t++) {
                    int et5[5];
                    fscanf(ef, "%d %d %d %d %d",
                           &et5[0], &et5[1], &et5[2], &et5[3], &et5[4]);
                    int t5_got[5];
                    top5(acts.logits + t * V, V, t5_got);
                    int match = 1;
                    for (int k = 0; k < 5; k++)
                        if (et5[k] != t5_got[k]) match = 0;
                    if (!match) {
                        printf("  pos %d: MISMATCH vs PyTorch\n", t);
                        py_ok = 0;
                    }
                }
                if (py_ok) printf("all positions match PyTorch reference\n");
            }
            fclose(ef);
        } else {
            printf("could not open %s\n", argv[2]);
        }
    }

    /* ── summary ── */
    printf("\n=== %s ===\n",
           (all_match && loss_ok) ? "ALL PASS" : "FAIL");

    pt_free_activations(&acts);
    free(ref_logits);
    free(data);
    return (all_match && loss_ok) ? 0 : 1;
}
