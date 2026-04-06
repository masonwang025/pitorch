/*
 * Mac-side training: SGD overfit on a fixed token sequence.
 * Verifies loss drops below 0.1 and greedy inference reproduces the target.
 *
 * usage: ./test_train <model.bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pt.h"
#include "pt_ops.h"

#define T_SEQ       16
#define N_STEPS     200
#define LR          0.001f
#define LOSS_TARGET 0.1f

static int target[] = {
    1, 365, 471, 263, 9038, 2501, 7826, 931,
    4257, 2354, 29889, 727, 2217, 29892, 2296, 373
};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    void *data = pt_read_file(argv[1], NULL);

    pt_context_t ctx;
    pt_host_init(&ctx, data, T_SEQ);
    pt_print_config(&ctx);
    printf("target: %d tokens, lr=%.4f\n\n", T_SEQ, LR);

    /* ── training loop ── */
    float final_loss = 0.0f;
    for (int step = 0; step < N_STEPS; step++) {
        clock_t t0 = clock();
        float loss = pt_train_step(&ctx, target, T_SEQ, LR);
        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

        final_loss = loss;
        if (step < 10 || step % 10 == 0 || loss < LOSS_TARGET)
            printf("step %-3d: loss=%.6f  (%.2fs)\n", step, loss, elapsed);

        if (loss < LOSS_TARGET && step >= 5) {
            printf("\nconverged at step %d\n", step);
            break;
        }
    }

    /* ── verify ── */
    printf("\n--- verification ---\n");
    pt_reset_kv(&ctx);
    int token = target[0];
    int match = 1;

    for (int t = 0; t < T_SEQ - 1; t++) {
        int next = pt_forward_step(&ctx, token);
        if (next != target[t + 1]) match = 0;
        token = next;
    }

    printf("target:    ");
    for (int t = 0; t < T_SEQ; t++) printf("%d ", target[t]);
    printf("\ngenerated: %d ", target[0]);
    pt_reset_kv(&ctx);
    token = target[0];
    for (int t = 0; t < T_SEQ - 1; t++) {
        token = pt_forward_step(&ctx, token);
        printf("%d ", token);
    }
    printf("\n\n%s\n", match ? "MATCH" : "MISMATCH");

    pt_free(&ctx);
    free(data);
    return (final_loss < LOSS_TARGET && match) ? 0 : 1;
}
