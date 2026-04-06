/*
 * Host-side validation: 4-stage pipeline training simulation.
 *
 * Simulates the 4-Pi pipeline training topology:
 *   R3: embed + head (no layers)
 *   R0: layers [0,2)
 *   R1: layers [2,4)
 *   R2: layers [4,6)
 *
 * Forward:  R3 embed → R0 layers → R1 layers → R2 layers → R3 head
 * Backward: R3 head_bwd → R2 layers_bwd → R1 layers_bwd → R0 layers_bwd → R3 embed_bwd
 * SGD:      each rank updates only its own weights
 *
 * Transfers simulated via memcpy. Compares against monolithic training.
 *
 * Usage: ./test_stage_train_4pi <model.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama2.h"
#include "pt_ops.h"
#include "pt_train.h"

#define MAX_T   8
#define LR      0.001f
#define N_STEPS 20
#define WORLD_SIZE 4

/* Layer splits: 3 compute ranks (R0,R1,R2), 2 layers each for stories15M */
#define SPLIT0  0
#define SPLIT1  2
#define SPLIT2  4

static int tokens[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

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
    int shared = ((const int *)data)[5] > 0;
    int L = cfg.n_layers;
    int dim = cfg.dim;
    size_t act_bytes = (size_t)MAX_T * dim * sizeof(float);

    printf("model: dim=%d layers=%d T=%d shared_weights=%d\n",
           dim, L, MAX_T, shared);
    printf("4-pi split: R0=[0,%d) R1=[%d,%d) R2=[%d,%d) R3=embed+head\n",
           SPLIT1, SPLIT1, SPLIT2, SPLIT2, L);

    /* ================================================================
     * Test 1: 4-stage pipeline (single process, shared weights)
     * Validates that the 4-way split produces identical results.
     * ================================================================ */
    printf("\n--- Test 1: 4-stage pipeline vs monolithic (shared weights) ---\n");
    {
        void *data_ref = malloc(file_size); memcpy(data_ref, data, file_size);
        void *data_stg = malloc(file_size); memcpy(data_stg, data, file_size);

        pt_weights_t w_ref, w_stg;
        pt_load_weights(&w_ref, &cfg, data_ref);
        pt_load_weights(&w_stg, &cfg, data_stg);

        pt_activations_t a_ref, a_stg;
        pt_alloc_activations(&a_ref, &cfg, MAX_T);
        pt_alloc_activations(&a_stg, &cfg, MAX_T);

        pt_grads_t g_ref, g_stg;
        pt_alloc_grads(&g_ref, &cfg, shared);
        pt_alloc_grads(&g_stg, &cfg, shared);

        pt_backward_buf_t b_ref, b_stg;
        pt_alloc_backward_buf(&b_ref, &cfg, MAX_T);
        pt_alloc_backward_buf(&b_stg, &cfg, MAX_T);

        int pass = 1;
        for (int step = 0; step < N_STEPS; step++) {
            /* Reference: monolithic */
            pt_zero_grads(&g_ref);
            float loss_ref = pt_forward_train(&cfg, &w_ref, &a_ref, tokens, MAX_T,
                                              smatvec_cpu, NULL);
            pt_backward(&cfg, &w_ref, &g_ref, &a_ref, &b_ref, tokens, MAX_T, NULL, NULL);
            pt_sgd_update(&w_ref, &g_ref, LR, &cfg);

            /* Staged: 4-way split, same weight buffer */
            pt_zero_grads(&g_stg);

            /* R3: embed */
            pt_forward_train_embed(&w_stg, &a_stg, tokens, MAX_T, dim);
            /* R0: layers [0,2) */
            pt_forward_train_layers_range(&cfg, &w_stg, &a_stg, MAX_T,
                                          SPLIT0, SPLIT1, smatvec_cpu, NULL);
            /* R1: layers [2,4) */
            pt_forward_train_layers_range(&cfg, &w_stg, &a_stg, MAX_T,
                                          SPLIT1, SPLIT2, smatvec_cpu, NULL);
            /* R2: layers [4,6) */
            pt_forward_train_layers_range(&cfg, &w_stg, &a_stg, MAX_T,
                                          SPLIT2, L, smatvec_cpu, NULL);
            /* R3: head */
            float loss_stg = pt_forward_train_head(&cfg, &w_stg, &a_stg, tokens, MAX_T,
                                                    smatvec_cpu, NULL);

            /* Backward: R3 head → R2 → R1 → R0 → R3 embed */
            pt_backward_head(&cfg, &w_stg, &g_stg, &a_stg, &b_stg,
                             tokens, MAX_T, NULL, NULL);
            pt_backward_layers_range(&cfg, &w_stg, &g_stg, &a_stg, &b_stg, MAX_T,
                                     SPLIT2, L, NULL, NULL);
            pt_backward_layers_range(&cfg, &w_stg, &g_stg, &a_stg, &b_stg, MAX_T,
                                     SPLIT1, SPLIT2, NULL, NULL);
            pt_backward_layers_range(&cfg, &w_stg, &g_stg, &a_stg, &b_stg, MAX_T,
                                     SPLIT0, SPLIT1, NULL, NULL);
            pt_backward_embed(&g_stg, &b_stg, tokens, MAX_T, dim);

            /* SGD: split updates */
            pt_sgd_update_head(&w_stg, &g_stg, LR, &cfg);
            pt_sgd_update_layers(&w_stg, &g_stg, LR, &cfg, SPLIT0, SPLIT1);
            pt_sgd_update_layers(&w_stg, &g_stg, LR, &cfg, SPLIT1, SPLIT2);
            pt_sgd_update_layers(&w_stg, &g_stg, LR, &cfg, SPLIT2, L);

            int ok = (loss_ref == loss_stg) &&
                     (memcmp(data_ref, data_stg, file_size) == 0);
            printf("step %2d: loss ref=%.4f stg=%.4f %s\n",
                   step, loss_ref, loss_stg, ok ? "OK" : "FAIL");
            if (!ok) { pass = 0; break; }
        }
        printf("test 1: %s\n", pass ? "PASS" : "FAIL");

        pt_free_activations(&a_ref); pt_free_activations(&a_stg);
        pt_free_grads(&g_ref); pt_free_grads(&g_stg);
        pt_free_backward_buf(&b_ref); pt_free_backward_buf(&b_stg);
        free(data_ref); free(data_stg);
    }

    /* ================================================================
     * Test 2: 4-process simulation (independent weight copies per rank)
     *
     * Simulates what the 4 Pis actually do:
     *   R3: embed + head, updates head weights
     *   R0: layers [0,2), updates layer 0-1 weights
     *   R1: layers [2,4), updates layer 2-3 weights
     *   R2: layers [4,6), updates layer 4-5 weights
     *   Activations/gradients transferred via memcpy (simulating GPIO)
     * ================================================================ */
    printf("\n--- Test 2: 4-process simulation vs monolithic ---\n");
    {
        /* Reference: monolithic */
        void *data_ref = malloc(file_size); memcpy(data_ref, data, file_size);
        pt_weights_t w_ref;
        pt_load_weights(&w_ref, &cfg, data_ref);
        pt_activations_t a_ref;
        pt_alloc_activations(&a_ref, &cfg, MAX_T);
        pt_grads_t g_ref;
        pt_alloc_grads(&g_ref, &cfg, shared);
        pt_backward_buf_t b_ref;
        pt_alloc_backward_buf(&b_ref, &cfg, MAX_T);

        /* R3: embed + head */
        void *data_r3 = malloc(file_size); memcpy(data_r3, data, file_size);
        pt_weights_t w_r3;
        pt_load_weights(&w_r3, &cfg, data_r3);
        pt_activations_t a_r3;
        pt_alloc_activations(&a_r3, &cfg, MAX_T);
        pt_grads_t g_r3;
        pt_alloc_grads(&g_r3, &cfg, shared);
        pt_backward_buf_t b_r3;
        pt_alloc_backward_buf(&b_r3, &cfg, MAX_T);

        /* R0: layers [0,2) */
        void *data_r0 = malloc(file_size); memcpy(data_r0, data, file_size);
        pt_weights_t w_r0;
        pt_load_weights(&w_r0, &cfg, data_r0);
        pt_activations_t a_r0;
        pt_alloc_activations(&a_r0, &cfg, MAX_T);
        pt_grads_t g_r0;
        pt_alloc_grads(&g_r0, &cfg, shared);
        pt_backward_buf_t b_r0;
        pt_alloc_backward_buf(&b_r0, &cfg, MAX_T);

        /* R1: layers [2,4) */
        void *data_r1 = malloc(file_size); memcpy(data_r1, data, file_size);
        pt_weights_t w_r1;
        pt_load_weights(&w_r1, &cfg, data_r1);
        pt_activations_t a_r1;
        pt_alloc_activations(&a_r1, &cfg, MAX_T);
        pt_grads_t g_r1;
        pt_alloc_grads(&g_r1, &cfg, shared);
        pt_backward_buf_t b_r1;
        pt_alloc_backward_buf(&b_r1, &cfg, MAX_T);

        /* R2: layers [4,6) */
        void *data_r2 = malloc(file_size); memcpy(data_r2, data, file_size);
        pt_weights_t w_r2;
        pt_load_weights(&w_r2, &cfg, data_r2);
        pt_activations_t a_r2;
        pt_alloc_activations(&a_r2, &cfg, MAX_T);
        pt_grads_t g_r2;
        pt_alloc_grads(&g_r2, &cfg, shared);
        pt_backward_buf_t b_r2;
        pt_alloc_backward_buf(&b_r2, &cfg, MAX_T);

        int pass = 1;
        for (int step = 0; step < N_STEPS; step++) {
            /* ── Reference: monolithic ── */
            pt_zero_grads(&g_ref);
            float loss_ref = pt_forward_train(&cfg, &w_ref, &a_ref, tokens, MAX_T,
                                              smatvec_cpu, NULL);
            pt_backward(&cfg, &w_ref, &g_ref, &a_ref, &b_ref, tokens, MAX_T, NULL, NULL);
            pt_sgd_update(&w_ref, &g_ref, LR, &cfg);

            /* ── 4-Pi simulation ── */
            pt_zero_grads(&g_r3);
            pt_zero_grads(&g_r0);
            pt_zero_grads(&g_r1);
            pt_zero_grads(&g_r2);

            /* FORWARD: R3 embed → R0 → R1 → R2 → R3 head */

            /* R3: embed */
            pt_forward_train_embed(&w_r3, &a_r3, tokens, MAX_T, dim);

            /* GPIO: R3 sends residuals[0] to R0 */
            float *r3_act_out = a_r3.residuals;  /* residuals[0 * T * dim] */
            float *r0_act_in  = a_r0.residuals;
            memcpy(r0_act_in, r3_act_out, act_bytes);

            /* R0: layers [0,2) */
            pt_forward_train_layers_range(&cfg, &w_r0, &a_r0, MAX_T,
                                          SPLIT0, SPLIT1, smatvec_cpu, NULL);

            /* GPIO: R0 sends residuals[2] to R1 */
            float *r0_act_out = a_r0.residuals + SPLIT1 * MAX_T * dim;
            float *r1_act_in  = a_r1.residuals + SPLIT1 * MAX_T * dim;
            memcpy(r1_act_in, r0_act_out, act_bytes);

            /* R1: layers [2,4) */
            pt_forward_train_layers_range(&cfg, &w_r1, &a_r1, MAX_T,
                                          SPLIT1, SPLIT2, smatvec_cpu, NULL);

            /* GPIO: R1 sends residuals[4] to R2 */
            float *r1_act_out = a_r1.residuals + SPLIT2 * MAX_T * dim;
            float *r2_act_in  = a_r2.residuals + SPLIT2 * MAX_T * dim;
            memcpy(r2_act_in, r1_act_out, act_bytes);

            /* R2: layers [4,6) */
            pt_forward_train_layers_range(&cfg, &w_r2, &a_r2, MAX_T,
                                          SPLIT2, L, smatvec_cpu, NULL);

            /* GPIO: R2 sends residuals[6] to R3 */
            float *r2_act_out = a_r2.residuals + L * MAX_T * dim;
            float *r3_act_in  = a_r3.residuals + L * MAX_T * dim;
            memcpy(r3_act_in, r2_act_out, act_bytes);

            /* R3: head */
            float loss_pp = pt_forward_train_head(&cfg, &w_r3, &a_r3, tokens, MAX_T,
                                                   smatvec_cpu, NULL);

            /* BACKWARD: R3 head → R2 → R1 → R0 → R3 embed */

            /* R3: head backward */
            pt_backward_head(&cfg, &w_r3, &g_r3, &a_r3, &b_r3,
                             tokens, MAX_T, NULL, NULL);

            /* GPIO: R3 sends d_res to R2 */
            memcpy(b_r2.d_res, b_r3.d_res, act_bytes);

            /* R2: layers [4,6) backward */
            pt_backward_layers_range(&cfg, &w_r2, &g_r2, &a_r2, &b_r2, MAX_T,
                                     SPLIT2, L, NULL, NULL);

            /* GPIO: R2 sends d_res to R1 */
            memcpy(b_r1.d_res, b_r2.d_res, act_bytes);

            /* R1: layers [2,4) backward */
            pt_backward_layers_range(&cfg, &w_r1, &g_r1, &a_r1, &b_r1, MAX_T,
                                     SPLIT1, SPLIT2, NULL, NULL);

            /* GPIO: R1 sends d_res to R0 */
            memcpy(b_r0.d_res, b_r1.d_res, act_bytes);

            /* R0: layers [0,2) backward */
            pt_backward_layers_range(&cfg, &w_r0, &g_r0, &a_r0, &b_r0, MAX_T,
                                     SPLIT0, SPLIT1, NULL, NULL);

            /* GPIO: R0 sends d_res to R3 (ring closure) */
            memcpy(b_r3.d_res, b_r0.d_res, act_bytes);

            /* R3: embed backward */
            pt_backward_embed(&g_r3, &b_r3, tokens, MAX_T, dim);

            /* SGD: each rank updates only its own weights */
            pt_sgd_update_head(&w_r3, &g_r3, LR, &cfg);
            pt_sgd_update_layers(&w_r0, &g_r0, LR, &cfg, SPLIT0, SPLIT1);
            pt_sgd_update_layers(&w_r1, &g_r1, LR, &cfg, SPLIT1, SPLIT2);
            pt_sgd_update_layers(&w_r2, &g_r2, LR, &cfg, SPLIT2, L);

            int loss_ok = (loss_ref == loss_pp);
            printf("step %2d: loss ref=%.4f pp=%.4f %s\n",
                   step, loss_ref, loss_pp, loss_ok ? "OK" : "FAIL");

            if (!loss_ok) {
                pass = 0;
                printf("  DIVERGED: ref=%f pp=%f diff=%e\n",
                       loss_ref, loss_pp, loss_ref - loss_pp);
                break;
            }
        }
        printf("test 2: %s\n", pass ? "PASS" : "FAIL");

        pt_free_activations(&a_ref);
        pt_free_activations(&a_r3); pt_free_activations(&a_r0);
        pt_free_activations(&a_r1); pt_free_activations(&a_r2);
        pt_free_grads(&g_ref);
        pt_free_grads(&g_r3); pt_free_grads(&g_r0);
        pt_free_grads(&g_r1); pt_free_grads(&g_r2);
        pt_free_backward_buf(&b_ref);
        pt_free_backward_buf(&b_r3); pt_free_backward_buf(&b_r0);
        pt_free_backward_buf(&b_r1); pt_free_backward_buf(&b_r2);
        free(data_ref); free(data_r3); free(data_r0);
        free(data_r1); free(data_r2);
    }

    /* ================================================================
     * Test 3: GPU backward path (4-process with matvec)
     * ================================================================ */
    printf("\n--- Test 3: 4-process GPU backward path ---\n");
    {
        void *data_ref = malloc(file_size); memcpy(data_ref, data, file_size);
        pt_weights_t w_ref;
        pt_load_weights(&w_ref, &cfg, data_ref);
        pt_activations_t a_ref;
        pt_alloc_activations(&a_ref, &cfg, MAX_T);
        pt_grads_t g_ref;
        pt_alloc_grads(&g_ref, &cfg, shared);
        pt_backward_buf_t b_ref;
        pt_alloc_backward_buf(&b_ref, &cfg, MAX_T);

        void *data_r3 = malloc(file_size); memcpy(data_r3, data, file_size);
        pt_weights_t w_r3; pt_load_weights(&w_r3, &cfg, data_r3);
        pt_activations_t a_r3; pt_alloc_activations(&a_r3, &cfg, MAX_T);
        pt_grads_t g_r3; pt_alloc_grads(&g_r3, &cfg, shared);
        pt_backward_buf_t b_r3; pt_alloc_backward_buf(&b_r3, &cfg, MAX_T);

        void *data_r0 = malloc(file_size); memcpy(data_r0, data, file_size);
        pt_weights_t w_r0; pt_load_weights(&w_r0, &cfg, data_r0);
        pt_activations_t a_r0; pt_alloc_activations(&a_r0, &cfg, MAX_T);
        pt_grads_t g_r0; pt_alloc_grads(&g_r0, &cfg, shared);
        pt_backward_buf_t b_r0; pt_alloc_backward_buf(&b_r0, &cfg, MAX_T);

        void *data_r1 = malloc(file_size); memcpy(data_r1, data, file_size);
        pt_weights_t w_r1; pt_load_weights(&w_r1, &cfg, data_r1);
        pt_activations_t a_r1; pt_alloc_activations(&a_r1, &cfg, MAX_T);
        pt_grads_t g_r1; pt_alloc_grads(&g_r1, &cfg, shared);
        pt_backward_buf_t b_r1; pt_alloc_backward_buf(&b_r1, &cfg, MAX_T);

        void *data_r2 = malloc(file_size); memcpy(data_r2, data, file_size);
        pt_weights_t w_r2; pt_load_weights(&w_r2, &cfg, data_r2);
        pt_activations_t a_r2; pt_alloc_activations(&a_r2, &cfg, MAX_T);
        pt_grads_t g_r2; pt_alloc_grads(&g_r2, &cfg, shared);
        pt_backward_buf_t b_r2; pt_alloc_backward_buf(&b_r2, &cfg, MAX_T);

        int pass = 1;
        for (int step = 0; step < N_STEPS; step++) {
            pt_zero_grads(&g_ref);
            float loss_ref = pt_forward_train(&cfg, &w_ref, &a_ref, tokens, MAX_T,
                                              smatvec_cpu, NULL);
            pt_backward(&cfg, &w_ref, &g_ref, &a_ref, &b_ref, tokens, MAX_T,
                        smatvec_cpu, NULL);
            pt_sgd_update(&w_ref, &g_ref, LR, &cfg);

            pt_zero_grads(&g_r3); pt_zero_grads(&g_r0);
            pt_zero_grads(&g_r1); pt_zero_grads(&g_r2);

            /* Forward */
            pt_forward_train_embed(&w_r3, &a_r3, tokens, MAX_T, dim);
            memcpy(a_r0.residuals, a_r3.residuals, act_bytes);

            pt_forward_train_layers_range(&cfg, &w_r0, &a_r0, MAX_T,
                                          SPLIT0, SPLIT1, smatvec_cpu, NULL);
            memcpy(a_r1.residuals + SPLIT1 * MAX_T * dim,
                   a_r0.residuals + SPLIT1 * MAX_T * dim, act_bytes);

            pt_forward_train_layers_range(&cfg, &w_r1, &a_r1, MAX_T,
                                          SPLIT1, SPLIT2, smatvec_cpu, NULL);
            memcpy(a_r2.residuals + SPLIT2 * MAX_T * dim,
                   a_r1.residuals + SPLIT2 * MAX_T * dim, act_bytes);

            pt_forward_train_layers_range(&cfg, &w_r2, &a_r2, MAX_T,
                                          SPLIT2, L, smatvec_cpu, NULL);
            memcpy(a_r3.residuals + L * MAX_T * dim,
                   a_r2.residuals + L * MAX_T * dim, act_bytes);

            float loss_pp = pt_forward_train_head(&cfg, &w_r3, &a_r3, tokens, MAX_T,
                                                   smatvec_cpu, NULL);

            /* Backward with matvec (GPU path) */
            pt_backward_head(&cfg, &w_r3, &g_r3, &a_r3, &b_r3,
                             tokens, MAX_T, smatvec_cpu, NULL);
            memcpy(b_r2.d_res, b_r3.d_res, act_bytes);

            pt_backward_layers_range(&cfg, &w_r2, &g_r2, &a_r2, &b_r2, MAX_T,
                                     SPLIT2, L, smatvec_cpu, NULL);
            memcpy(b_r1.d_res, b_r2.d_res, act_bytes);

            pt_backward_layers_range(&cfg, &w_r1, &g_r1, &a_r1, &b_r1, MAX_T,
                                     SPLIT1, SPLIT2, smatvec_cpu, NULL);
            memcpy(b_r0.d_res, b_r1.d_res, act_bytes);

            pt_backward_layers_range(&cfg, &w_r0, &g_r0, &a_r0, &b_r0, MAX_T,
                                     SPLIT0, SPLIT1, smatvec_cpu, NULL);
            memcpy(b_r3.d_res, b_r0.d_res, act_bytes);

            pt_backward_embed(&g_r3, &b_r3, tokens, MAX_T, dim);

            pt_sgd_update_head(&w_r3, &g_r3, LR, &cfg);
            pt_sgd_update_layers(&w_r0, &g_r0, LR, &cfg, SPLIT0, SPLIT1);
            pt_sgd_update_layers(&w_r1, &g_r1, LR, &cfg, SPLIT1, SPLIT2);
            pt_sgd_update_layers(&w_r2, &g_r2, LR, &cfg, SPLIT2, L);

            int loss_ok = (loss_ref == loss_pp);
            printf("step %2d: loss ref=%.4f pp=%.4f %s\n",
                   step, loss_ref, loss_pp, loss_ok ? "OK" : "FAIL");

            if (!loss_ok) {
                pass = 0;
                printf("  DIVERGED: ref=%f pp=%f diff=%e\n",
                       loss_ref, loss_pp, loss_ref - loss_pp);
                break;
            }
        }
        printf("test 3: %s\n", pass ? "PASS" : "FAIL");

        pt_free_activations(&a_ref);
        pt_free_activations(&a_r3); pt_free_activations(&a_r0);
        pt_free_activations(&a_r1); pt_free_activations(&a_r2);
        pt_free_grads(&g_ref);
        pt_free_grads(&g_r3); pt_free_grads(&g_r0);
        pt_free_grads(&g_r1); pt_free_grads(&g_r2);
        pt_free_backward_buf(&b_ref);
        pt_free_backward_buf(&b_r3); pt_free_backward_buf(&b_r0);
        pt_free_backward_buf(&b_r1); pt_free_backward_buf(&b_r2);
        free(data_ref); free(data_r3); free(data_r0);
        free(data_r1); free(data_r2);
    }

    free(data);
    return 0;
}
