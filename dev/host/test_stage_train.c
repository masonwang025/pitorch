/*
 * Host-side validation: staged pipeline training with split SGD.
 *
 * Two modes:
 *   1. Single-process pipeline: both rank halves update same weights (sanity check)
 *   2. Two-process simulation: each "rank" has independent weights, only updates its own
 *      layers, communicates activations/gradients via memcpy (simulates GPIO transfers)
 *
 * Compares against monolithic pt_forward_train → pt_backward → pt_sgd_update.
 *
 * Usage: ./test_stage_train <model.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama2.h"
#include "pt_ops.h"
#include "pt_train.h"

#define MAX_T   8
#define SPLIT   3   /* layers 0..2 on rank 0, 3..5 on rank 1 */
#define LR      0.001f
#define N_STEPS 20

static int tokens[] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    /* Load model */
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

    printf("model: dim=%d layers=%d split=%d T=%d shared_weights=%d\n",
           dim, L, SPLIT, MAX_T, shared);

    /* ================================================================
     * Test 1: Single-process pipeline (split SGD, shared weight buffer)
     * ================================================================ */
    printf("\n--- Test 1: single-process pipeline vs monolithic ---\n");
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
            pt_zero_grads(&g_ref);
            float loss_ref = pt_forward_train(&cfg, &w_ref, &a_ref, tokens, MAX_T,
                                              smatvec_cpu, NULL);
            pt_backward(&cfg, &w_ref, &g_ref, &a_ref, &b_ref, tokens, MAX_T, NULL, NULL);
            pt_sgd_update(&w_ref, &g_ref, LR, &cfg);

            pt_zero_grads(&g_stg);
            pt_forward_train_embed(&w_stg, &a_stg, tokens, MAX_T, dim);
            pt_forward_train_layers_range(&cfg, &w_stg, &a_stg, MAX_T, 0, SPLIT, smatvec_cpu, NULL);
            pt_forward_train_layers_range(&cfg, &w_stg, &a_stg, MAX_T, SPLIT, L, smatvec_cpu, NULL);
            float loss_stg = pt_forward_train_head(&cfg, &w_stg, &a_stg, tokens, MAX_T, smatvec_cpu, NULL);

            pt_backward_head(&cfg, &w_stg, &g_stg, &a_stg, &b_stg, tokens, MAX_T, NULL, NULL);
            pt_backward_layers_range(&cfg, &w_stg, &g_stg, &a_stg, &b_stg, MAX_T, SPLIT, L, NULL, NULL);
            pt_backward_layers_range(&cfg, &w_stg, &g_stg, &a_stg, &b_stg, MAX_T, 0, SPLIT, NULL, NULL);
            pt_backward_embed(&g_stg, &b_stg, tokens, MAX_T, dim);

            pt_sgd_update_head(&w_stg, &g_stg, LR, &cfg);
            pt_sgd_update_layers(&w_stg, &g_stg, LR, &cfg, 0, SPLIT);
            pt_sgd_update_layers(&w_stg, &g_stg, LR, &cfg, SPLIT, L);

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
     * Test 2: Two-process simulation (independent weight copies per rank)
     *
     * This simulates what the Pis actually do:
     *   - Rank 0 has its own model copy, updates head + layers 0..SPLIT
     *   - Rank 1 has its own model copy, updates layers SPLIT..L
     *   - Activations/gradients transferred via memcpy (simulating GPIO)
     * ================================================================ */
    printf("\n--- Test 2: two-process simulation vs monolithic ---\n");
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

        /* Rank 0: owns embedding + layers 0..SPLIT + head */
        void *data_r0 = malloc(file_size); memcpy(data_r0, data, file_size);
        pt_weights_t w_r0;
        pt_load_weights(&w_r0, &cfg, data_r0);
        pt_activations_t a_r0;
        pt_alloc_activations(&a_r0, &cfg, MAX_T);
        pt_grads_t g_r0;
        pt_alloc_grads(&g_r0, &cfg, shared);
        pt_backward_buf_t b_r0;
        pt_alloc_backward_buf(&b_r0, &cfg, MAX_T);

        /* Rank 1: owns layers SPLIT..L */
        void *data_r1 = malloc(file_size); memcpy(data_r1, data, file_size);
        pt_weights_t w_r1;
        pt_load_weights(&w_r1, &cfg, data_r1);
        pt_activations_t a_r1;
        pt_alloc_activations(&a_r1, &cfg, MAX_T);
        pt_grads_t g_r1;
        pt_alloc_grads(&g_r1, &cfg, shared);
        pt_backward_buf_t b_r1;
        pt_alloc_backward_buf(&b_r1, &cfg, MAX_T);

        /* Transfer buffer (simulating GPIO wire) */
        size_t act_bytes = (size_t)MAX_T * dim * sizeof(float);

        int pass = 1;
        for (int step = 0; step < N_STEPS; step++) {
            /* ── Reference: monolithic ── */
            pt_zero_grads(&g_ref);
            float loss_ref = pt_forward_train(&cfg, &w_ref, &a_ref, tokens, MAX_T,
                                              smatvec_cpu, NULL);
            pt_backward(&cfg, &w_ref, &g_ref, &a_ref, &b_ref, tokens, MAX_T, NULL, NULL);
            pt_sgd_update(&w_ref, &g_ref, LR, &cfg);

            /* ── Rank 0: forward embed + layers 0..SPLIT ── */
            pt_zero_grads(&g_r0);
            pt_forward_train_embed(&w_r0, &a_r0, tokens, MAX_T, dim);
            pt_forward_train_layers_range(&cfg, &w_r0, &a_r0, MAX_T,
                                          0, SPLIT, smatvec_cpu, NULL);

            /* GPIO: rank 0 sends activations[SPLIT] to rank 1 */
            float *r0_act_out = a_r0.residuals + SPLIT * MAX_T * dim;
            float *r1_act_in  = a_r1.residuals + SPLIT * MAX_T * dim;
            memcpy(r1_act_in, r0_act_out, act_bytes);

            /* ── Rank 1: forward layers SPLIT..L ── */
            pt_zero_grads(&g_r1);
            pt_forward_train_layers_range(&cfg, &w_r1, &a_r1, MAX_T,
                                          SPLIT, L, smatvec_cpu, NULL);

            /* GPIO: rank 1 sends activations[L] back to rank 0 */
            float *r1_act_out = a_r1.residuals + L * MAX_T * dim;
            float *r0_act_in  = a_r0.residuals + L * MAX_T * dim;
            memcpy(r0_act_in, r1_act_out, act_bytes);

            /* ── Rank 0: head forward + loss ── */
            float loss_pp = pt_forward_train_head(&cfg, &w_r0, &a_r0, tokens, MAX_T,
                                                   smatvec_cpu, NULL);

            /* ── Rank 0: backward head ── */
            pt_backward_head(&cfg, &w_r0, &g_r0, &a_r0, &b_r0,
                             tokens, MAX_T, NULL, NULL);

            /* GPIO: rank 0 sends d_res to rank 1 */
            memcpy(b_r1.d_res, b_r0.d_res, act_bytes);

            /* ── Rank 1: backward layers SPLIT..L ── */
            pt_backward_layers_range(&cfg, &w_r1, &g_r1, &a_r1, &b_r1, MAX_T,
                                     SPLIT, L, NULL, NULL);

            /* GPIO: rank 1 sends d_res back to rank 0 */
            memcpy(b_r0.d_res, b_r1.d_res, act_bytes);

            /* ── Rank 0: backward layers 0..SPLIT + embed ── */
            pt_backward_layers_range(&cfg, &w_r0, &g_r0, &a_r0, &b_r0, MAX_T,
                                     0, SPLIT, NULL, NULL);
            pt_backward_embed(&g_r0, &b_r0, tokens, MAX_T, dim);

            /* ── SGD: each rank updates only its own weights ── */
            pt_sgd_update_head(&w_r0, &g_r0, LR, &cfg);
            pt_sgd_update_layers(&w_r0, &g_r0, LR, &cfg, 0, SPLIT);
            pt_sgd_update_layers(&w_r1, &g_r1, LR, &cfg, SPLIT, L);

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

        pt_free_activations(&a_ref); pt_free_activations(&a_r0); pt_free_activations(&a_r1);
        pt_free_grads(&g_ref); pt_free_grads(&g_r0); pt_free_grads(&g_r1);
        pt_free_backward_buf(&b_ref); pt_free_backward_buf(&b_r0); pt_free_backward_buf(&b_r1);
        free(data_ref); free(data_r0); free(data_r1);
    }

    /* ================================================================
     * Test 3: GPU backward path (backward_input_pretrans with smatvec_cpu)
     *
     * The Pi uses the GPU code path in backward (w_transpose != NULL).
     * This tests the same path on the host with smatvec_cpu.
     * ================================================================ */
    printf("\n--- Test 3: GPU backward path (two-process sim) ---\n");
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

        void *data_r0 = malloc(file_size); memcpy(data_r0, data, file_size);
        pt_weights_t w_r0;
        pt_load_weights(&w_r0, &cfg, data_r0);
        pt_activations_t a_r0;
        pt_alloc_activations(&a_r0, &cfg, MAX_T);
        pt_grads_t g_r0;
        pt_alloc_grads(&g_r0, &cfg, shared);
        pt_backward_buf_t b_r0;
        pt_alloc_backward_buf(&b_r0, &cfg, MAX_T);

        void *data_r1 = malloc(file_size); memcpy(data_r1, data, file_size);
        pt_weights_t w_r1;
        pt_load_weights(&w_r1, &cfg, data_r1);
        pt_activations_t a_r1;
        pt_alloc_activations(&a_r1, &cfg, MAX_T);
        pt_grads_t g_r1;
        pt_alloc_grads(&g_r1, &cfg, shared);
        pt_backward_buf_t b_r1;
        pt_alloc_backward_buf(&b_r1, &cfg, MAX_T);

        size_t act_bytes = (size_t)MAX_T * dim * sizeof(float);

        int pass = 1;
        for (int step = 0; step < N_STEPS; step++) {
            /* Reference: monolithic with GPU backward path */
            pt_zero_grads(&g_ref);
            float loss_ref = pt_forward_train(&cfg, &w_ref, &a_ref, tokens, MAX_T,
                                              smatvec_cpu, NULL);
            pt_backward(&cfg, &w_ref, &g_ref, &a_ref, &b_ref, tokens, MAX_T,
                        smatvec_cpu, NULL);
            pt_sgd_update(&w_ref, &g_ref, LR, &cfg);

            /* Pipeline with GPU backward path */
            pt_zero_grads(&g_r0);
            pt_forward_train_embed(&w_r0, &a_r0, tokens, MAX_T, dim);
            pt_forward_train_layers_range(&cfg, &w_r0, &a_r0, MAX_T,
                                          0, SPLIT, smatvec_cpu, NULL);

            float *r0_act_out = a_r0.residuals + SPLIT * MAX_T * dim;
            float *r1_act_in  = a_r1.residuals + SPLIT * MAX_T * dim;
            memcpy(r1_act_in, r0_act_out, act_bytes);

            pt_zero_grads(&g_r1);
            pt_forward_train_layers_range(&cfg, &w_r1, &a_r1, MAX_T,
                                          SPLIT, L, smatvec_cpu, NULL);

            float *r1_act_out = a_r1.residuals + L * MAX_T * dim;
            float *r0_act_in  = a_r0.residuals + L * MAX_T * dim;
            memcpy(r0_act_in, r1_act_out, act_bytes);

            float loss_pp = pt_forward_train_head(&cfg, &w_r0, &a_r0, tokens, MAX_T,
                                                   smatvec_cpu, NULL);

            /* Backward with GPU path: pass smatvec_cpu as matvec */
            pt_backward_head(&cfg, &w_r0, &g_r0, &a_r0, &b_r0,
                             tokens, MAX_T, smatvec_cpu, NULL);

            memcpy(b_r1.d_res, b_r0.d_res, act_bytes);

            pt_backward_layers_range(&cfg, &w_r1, &g_r1, &a_r1, &b_r1, MAX_T,
                                     SPLIT, L, smatvec_cpu, NULL);

            memcpy(b_r0.d_res, b_r1.d_res, act_bytes);

            pt_backward_layers_range(&cfg, &w_r0, &g_r0, &a_r0, &b_r0, MAX_T,
                                     0, SPLIT, smatvec_cpu, NULL);
            pt_backward_embed(&g_r0, &b_r0, tokens, MAX_T, dim);

            pt_sgd_update_head(&w_r0, &g_r0, LR, &cfg);
            pt_sgd_update_layers(&w_r0, &g_r0, LR, &cfg, 0, SPLIT);
            pt_sgd_update_layers(&w_r1, &g_r1, LR, &cfg, SPLIT, L);

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

        pt_free_activations(&a_ref); pt_free_activations(&a_r0); pt_free_activations(&a_r1);
        pt_free_grads(&g_ref); pt_free_grads(&g_r0); pt_free_grads(&g_r1);
        pt_free_backward_buf(&b_ref); pt_free_backward_buf(&b_r0); pt_free_backward_buf(&b_r1);
        free(data_ref); free(data_r0); free(data_r1);
    }

    free(data);
    return 0;
}
