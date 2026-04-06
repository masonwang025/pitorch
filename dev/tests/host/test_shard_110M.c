/*
 * test_shard_train.c
 *
 * Verifies that loading 4 shards and simulating the pipeline-parallel
 * training flow produces the EXACT same loss as the full-model training.
 *
 * Pipeline layout:
 *   R3: embed -> R0: layers[0,3) -> R1: layers[3,6) -> R2: layers[6,8) -> R3: head -> loss
 *   Backward: R3 head_bwd -> R2 bwd -> R1 bwd -> R0 bwd -> R3 embed_bwd
 *   SGD on each shard's weights
 */

#include "pt.h"
#include "pt_shard.h"
#include "pt_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_T  8
#define LR     0.001f
#define NUM_RANKS 4

static const int tokens[MAX_T] = { 1, 365, 471, 263, 9038, 2501, 7826, 931 };

int main(void) {
    printf("=== shard pipeline training test ===\n\n");

    /* ────────────────────────────────────────────────────────────
     * Part 1: Full model reference loss
     * ──────────────────────────────────────────────────────────── */
    long full_sz;
    void *full_data = pt_read_file("../../weights/stories110M.bin", &full_sz);
    printf("loaded full model: %ld bytes\n", full_sz);

    pt_context_t full;
    pt_host_init(&full, full_data, MAX_T);
    printf("full model: dim=%d layers=%d vocab=%d\n",
           full.cfg.dim, full.cfg.n_layers, full.cfg.vocab_size);

    float ref_loss = pt_train_step(&full, tokens, MAX_T, LR);
    printf("reference loss: %.10f\n\n", ref_loss);

    /* ────────────────────────────────────────────────────────────
     * Part 2: Load all 4 shards
     * ──────────────────────────────────────────────────────────── */
    const char *shard_paths[NUM_RANKS] = {
        "../../weights/shards/110M/rank0.bin",
        "../../weights/shards/110M/rank1.bin",
        "../../weights/shards/110M/rank2.bin",
        "../../weights/shards/110M/rank3.bin",
    };

    /* Per-rank data */
    void              *shard_data[NUM_RANKS];
    pt_shard_info_t    shard_info[NUM_RANKS];
    pt_config_t        global_cfg;           /* same for all shards */
    pt_config_t        local_cfg[NUM_RANKS]; /* n_layers = n_local */
    pt_weights_t       sw[NUM_RANKS];
    pt_activations_t   sa[NUM_RANKS];
    pt_grads_t         sg[NUM_RANKS];
    pt_backward_buf_t  sb[NUM_RANKS];
    int                shared_weights;

    for (int r = 0; r < NUM_RANKS; r++) {
        long sz;
        shard_data[r] = pt_read_file(shard_paths[r], &sz);
        printf("loaded shard rank%d: %ld bytes\n", r, sz);

        pt_load_shard_header(&global_cfg, &shard_info[r], shard_data[r]);
        shared_weights = ((const int *)shard_data[r])[5] > 0;
        pt_load_shard_weights(&sw[r], &global_cfg, &shard_info[r], shard_data[r]);

        /* Local config: n_layers = n_local for correct allocation sizing */
        local_cfg[r] = global_cfg;
        local_cfg[r].n_layers = shard_info[r].n_local;

        printf("  rank=%d layers=[%d,%d) n_local=%d embed=%d head=%d\n",
               shard_info[r].rank, shard_info[r].l_start, shard_info[r].l_end,
               shard_info[r].n_local, shard_info[r].has_embed, shard_info[r].has_head);

        /* Allocate training buffers for this shard */
        pt_alloc_activations(&sa[r], &local_cfg[r], MAX_T);
        pt_alloc_grads(&sg[r], &local_cfg[r], shared_weights);
        pt_alloc_backward_buf(&sb[r], &local_cfg[r], MAX_T);
    }
    printf("\n");

    int dim = global_cfg.dim;
    int T   = MAX_T;

    /* ────────────────────────────────────────────────────────────
     * Part 3: Pipeline forward pass
     *
     * R3 (rank 3): embed -> residuals[0]
     * R0 (rank 0): layers [0, n_local_0) -> output residuals
     * R1 (rank 1): layers [0, n_local_1) -> output residuals
     * R2 (rank 2): layers [0, n_local_2) -> output residuals
     * R3 (rank 3): head -> loss
     * ──────────────────────────────────────────────────────────── */

    /* Zero all grads */
    for (int r = 0; r < NUM_RANKS; r++)
        pt_zero_grads(&sg[r]);

    /* Step 1: R3 does embedding */
    int r3 = 3;
    sa[r3].T = T;
    pt_forward_train_embed(&sw[r3], &sa[r3], tokens, T, dim);
    printf("R3: embed done\n");

    /* Copy R3 residuals[0] -> R0 residuals[0] (T*dim floats) */
    int r0 = 0;
    memcpy(sa[r0].residuals, sa[r3].residuals, (size_t)T * dim * sizeof(float));

    /* Step 2: R0 runs its local layers [0, n_local_0) */
    int nl0 = shard_info[r0].n_local;
    pt_forward_train_layers_range(&local_cfg[r0], &sw[r0], &sa[r0], T,
                                  0, nl0, smatvec_cpu, NULL);
    printf("R0: layers [%d,%d) done\n", shard_info[r0].l_start, shard_info[r0].l_end);

    /* Copy R0 output residuals -> R1 input residuals */
    int r1 = 1;
    int nl1 = shard_info[r1].n_local;
    memcpy(sa[r1].residuals, sa[r0].residuals + nl0 * T * dim,
           (size_t)T * dim * sizeof(float));

    /* Step 3: R1 runs its local layers [0, n_local_1) */
    pt_forward_train_layers_range(&local_cfg[r1], &sw[r1], &sa[r1], T,
                                  0, nl1, smatvec_cpu, NULL);
    printf("R1: layers [%d,%d) done\n", shard_info[r1].l_start, shard_info[r1].l_end);

    /* Copy R1 output residuals -> R2 input residuals */
    int r2 = 2;
    int nl2 = shard_info[r2].n_local;
    memcpy(sa[r2].residuals, sa[r1].residuals + nl1 * T * dim,
           (size_t)T * dim * sizeof(float));

    /* Step 4: R2 runs its local layers [0, n_local_2) */
    pt_forward_train_layers_range(&local_cfg[r2], &sw[r2], &sa[r2], T,
                                  0, nl2, smatvec_cpu, NULL);
    printf("R2: layers [%d,%d) done\n", shard_info[r2].l_start, shard_info[r2].l_end);

    /* Copy R2 output residuals -> R3 residuals[0]
     * R3 has n_local=0, so head reads from residuals + 0*T*dim = residuals[0] */
    memcpy(sa[r3].residuals, sa[r2].residuals + nl2 * T * dim,
           (size_t)T * dim * sizeof(float));

    /* Step 5: R3 does head (final rmsnorm + classifier + loss) */
    float pipeline_loss = pt_forward_train_head(&local_cfg[r3], &sw[r3], &sa[r3],
                                                tokens, T, smatvec_cpu, NULL);
    printf("R3: head done, pipeline loss = %.10f\n\n", pipeline_loss);

    /* ────────────────────────────────────────────────────────────
     * Part 4: Pipeline backward pass
     *
     * R3: head backward -> d_res
     * R2: backward layers -> d_res
     * R1: backward layers -> d_res
     * R0: backward layers -> d_res
     * R3: embed backward
     * ──────────────────────────────────────────────────────────── */

    /* Step 1: R3 backward head */
    pt_backward_head(&local_cfg[r3], &sw[r3], &sg[r3], &sa[r3], &sb[r3],
                     tokens, T, smatvec_cpu, NULL);
    printf("R3: head backward done\n");

    /* Copy R3 d_res -> R2 d_res */
    memcpy(sb[r2].d_res, sb[r3].d_res, (size_t)T * dim * sizeof(float));

    /* Step 2: R2 backward layers */
    pt_backward_layers_range(&local_cfg[r2], &sw[r2], &sg[r2], &sa[r2], &sb[r2],
                             T, 0, nl2, smatvec_cpu, NULL);
    printf("R2: backward layers done\n");

    /* Copy R2 d_res -> R1 d_res */
    memcpy(sb[r1].d_res, sb[r2].d_res, (size_t)T * dim * sizeof(float));

    /* Step 3: R1 backward layers */
    pt_backward_layers_range(&local_cfg[r1], &sw[r1], &sg[r1], &sa[r1], &sb[r1],
                             T, 0, nl1, smatvec_cpu, NULL);
    printf("R1: backward layers done\n");

    /* Copy R1 d_res -> R0 d_res */
    memcpy(sb[r0].d_res, sb[r1].d_res, (size_t)T * dim * sizeof(float));

    /* Step 4: R0 backward layers */
    pt_backward_layers_range(&local_cfg[r0], &sw[r0], &sg[r0], &sa[r0], &sb[r0],
                             T, 0, nl0, smatvec_cpu, NULL);
    printf("R0: backward layers done\n");

    /* Copy R0 d_res -> R3 d_res for embed backward */
    memcpy(sb[r3].d_res, sb[r0].d_res, (size_t)T * dim * sizeof(float));

    /* Step 5: R3 embed backward */
    pt_backward_embed(&sg[r3], &sb[r3], tokens, T, dim);
    printf("R3: embed backward done\n\n");

    /* ────────────────────────────────────────────────────────────
     * Part 5: SGD update on each shard
     * ──────────────────────────────────────────────────────────── */
    /* R0: update layers [0, n_local_0) using local indices */
    pt_sgd_update_layers(&sw[r0], &sg[r0], LR, &local_cfg[r0], 0, nl0);
    printf("R0: SGD done\n");

    /* R1: update layers [0, n_local_1) */
    pt_sgd_update_layers(&sw[r1], &sg[r1], LR, &local_cfg[r1], 0, nl1);
    printf("R1: SGD done\n");

    /* R2: update layers [0, n_local_2) */
    pt_sgd_update_layers(&sw[r2], &sg[r2], LR, &local_cfg[r2], 0, nl2);
    printf("R2: SGD done\n");

    /* R3: update embed + head */
    pt_sgd_update_head(&sw[r3], &sg[r3], LR, &local_cfg[r3]);
    printf("R3: SGD done\n\n");

    /* ────────────────────────────────────────────────────────────
     * Part 6: Compare losses
     * ──────────────────────────────────────────────────────────── */
    printf("=== results ===\n");
    printf("reference loss:  %.10f\n", ref_loss);
    printf("pipeline  loss:  %.10f\n", pipeline_loss);
    printf("difference:      %.2e\n", fabsf(ref_loss - pipeline_loss));

    if (fabsf(ref_loss - pipeline_loss) < 1e-6f) {
        printf("\nPASS: losses match (within 1e-6)\n");
    } else {
        printf("\nFAIL: losses do not match!\n");
        /* Clean up before exit */
        for (int r = 0; r < NUM_RANKS; r++) {
            pt_free_activations(&sa[r]);
            pt_free_grads(&sg[r]);
            pt_free_backward_buf(&sb[r]);
            free(shard_data[r]);
        }
        pt_free(&full);
        free(full_data);
        return 1;
    }

    /* Clean up */
    for (int r = 0; r < NUM_RANKS; r++) {
        pt_free_activations(&sa[r]);
        pt_free_grads(&sg[r]);
        pt_free_backward_buf(&sb[r]);
        free(shard_data[r]);
    }
    pt_free(&full);
    free(full_data);

    return 0;
}
