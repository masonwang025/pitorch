/*
 * Phase 6: Profiled training using the clean pt_context_t API.
 * Loads stories15M, runs SGD with tracing enabled, writes trace.json + meta.json.
 *
 * usage: ./test_train_profiled <model.bin> [trace_dir]
 *
 * trace_dir defaults to ../../traces/runs/<timestamp>_train/
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "pt.h"
#include "pt_ops.h"

#define T_SEQ       8
#define N_STEPS     20
#define LR          0.001f
#define LOSS_TARGET 0.1f

static int target[] = {
    1, 365, 471, 263, 9038, 2501, 7826, 931
};

static void make_timestamp(char *buf, int len) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    snprintf(buf, len, "%04d%02d%02d_%02d%02d%02d",
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin> [trace_dir]\n", argv[0]);
        return 1;
    }

    void *data = pt_read_file(argv[1], NULL);

    pt_context_t ctx;
    pt_host_init(&ctx, data, T_SEQ);
    pt_enable_trace(&ctx, NULL);

    printf("pitorch profiled train | %dd %dL %dKv | CPU\n",
           ctx.cfg.dim, ctx.cfg.n_layers, ctx.cfg.vocab_size / 1000);
    printf("T=%d steps=%d lr=%.4f\n\n", T_SEQ, N_STEPS, LR);

    clock_t wall_start = clock();
    float final_loss = 0.0f;
    int actual_steps = 0;

    for (int step = 0; step < N_STEPS; step++) {
        float loss = pt_train_step(&ctx, target, T_SEQ, LR);
        final_loss = loss;
        actual_steps = step + 1;
        printf("step %-3d: loss=%.6f\n", step, loss);
        if (loss < LOSS_TARGET && step >= 3) {
            printf("converged at step %d\n", step);
            break;
        }
    }

    double wall_s = (double)(clock() - wall_start) / CLOCKS_PER_SEC;
    printf("\ntotal: %.1fs, %d events recorded\n", wall_s, ctx.trace->count);

    /* ── verify via greedy decode ── */
    pt_reset_kv(&ctx);
    int token = target[0];
    int match = 1;
    for (int t = 0; t < T_SEQ - 1; t++) {
        int next = pt_forward_step(&ctx, token);
        if (next != target[t + 1]) match = 0;
        token = next;
    }
    printf("verification: %s\n", match ? "MATCH" : "MISMATCH");

    /* ── write trace + meta ── */
    char trace_dir[512];
    if (argc > 2) {
        snprintf(trace_dir, sizeof(trace_dir), "%s", argv[2]);
    } else {
        char ts[32];
        make_timestamp(ts, sizeof(ts));
        snprintf(trace_dir, sizeof(trace_dir), "../../traces/runs/%s_train", ts);
    }
    mkdir(trace_dir, 0755);

    char path[600];
    snprintf(path, sizeof(path), "%s/trace.json", trace_dir);
    pt_trace_write_json(ctx.trace, path);
    printf("wrote %s (%d events)\n", path, ctx.trace->count);

    snprintf(path, sizeof(path), "%s/meta.json", trace_dir);
    pt_trace_write_meta(path,
                        "stories15M training profiled",
                        "stories15M",
                        ctx.cfg.dim, ctx.cfg.n_layers, ctx.cfg.vocab_size,
                        "mac_host", 0,
                        actual_steps, LR, final_loss, (float)wall_s,
                        "Phase 6 baseline");
    printf("wrote %s\n", path);

    pt_free(&ctx);
    free(data);
    return match ? 0 : 1;
}
