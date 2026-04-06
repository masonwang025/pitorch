#ifndef PITORCH_PT_H
#define PITORCH_PT_H

/*
 * Top-level pitorch context API.
 *
 * pt_context_t bundles config, weights, inference state, and (optionally)
 * training state into one struct.  One call to pt_host_init / pt_pi_init
 * replaces all the boilerplate (scratch allocator, GPU setup, memory map).
 *
 * Pass max_T > 0 for training (allocates activations, grads, backward buf).
 * Pass max_T == 0 for inference only (just the KV-cache state).
 *
 * The low-level API (pt_forward, pt_forward_train, pt_backward, etc.)
 * remains available for fine-grained use.
 */

#include "llama2.h"
#include "pt_train.h"
#include "pt_shard.h"
#include "trace.h"

typedef struct {
    pt_config_t       cfg;
    pt_weights_t      w;
    int               shared_weights;
    pt_matvec_fn      matvec;

    /* inference */
    pt_state_t        state;
    int               pos;        /* current decode position (for pt_forward_step) */

    /* training (zeroed when max_T == 0) */
    pt_activations_t  acts;
    pt_grads_t        grads;
    pt_backward_buf_t bb;
    int               max_T;

    /* profiling — NULL means disabled */
    pt_trace_t       *trace;

    /* Pi-specific (0 on host) */
    int               num_qpus;
} pt_context_t;

/*
 * Host (Mac) initialization: mallocs everything.
 * weight_data: raw .bin file contents (header + weights).
 * max_T > 0: allocate training buffers.  max_T == 0: inference only.
 */
#ifndef __RPI__
void pt_host_init(pt_context_t *ctx, void *weight_data, int max_T);
void pt_free(pt_context_t *ctx);
#else
/*
 * Pi initialization: sets up GPU, computes memory layout, scratch-allocates.
 * weight_data: pointer to weights in memory (e.g. 0x02000000 from SD).
 * num_qpus: QPU count for GPU matvec (typically 12).
 * max_T > 0: allocate training buffers.  max_T == 0: inference only.
 * arena_bytes: GPU arena size (1 MB for inference, 2 MB for training).
 */
void pt_pi_init(pt_context_t *ctx, void *weight_data,
                int num_qpus, int max_T, unsigned arena_bytes);

/*
 * Pi initialization from a weight shard file (pipeline-parallel).
 * Same as pt_pi_init but reads the shard header and only allocates
 * grads/state for owned layers + embed/head components.
 * Fills shard_out with rank/layer info.
 * cfg.n_layers is set to n_local (the local layer count).
 */
void pt_pi_init_shard(pt_context_t *ctx, pt_shard_info_t *shard_out,
                      void *shard_data, int num_qpus, int max_T,
                      unsigned arena_bytes);
#endif

/* ── inference helpers ─────────────────────────────────────── */

/*
 * Single inference step: forward pass + argmax.
 * Feeds token at ctx->pos, increments pos, returns the greedy next token.
 */
int pt_forward_step(pt_context_t *ctx, int token);

/* Zero the KV cache and reset pos to 0. */
void pt_reset_kv(pt_context_t *ctx);

/* Print model config banner (dim, layers, vocab, QPUs). */
void pt_print_config(const pt_context_t *ctx);

/* ── host utilities ────────────────────────────────────────── */

#ifndef __RPI__
/* Read entire file into malloc'd buffer. Exits on failure. */
void *pt_read_file(const char *path, long *out_size);
#endif

/* ── training ──────────────────────────────────────────────── */

/*
 * One full training step: zero_grads → forward_train → backward → sgd_update.
 * Returns the cross-entropy loss.  Requires max_T > 0 at init.
 */
float pt_train_step(pt_context_t *ctx, const int *tokens, int T, float lr);

/* ── profiling ─────────────────────────────────────────────── */

void pt_enable_trace(pt_context_t *ctx, pt_trace_t *preallocated);
void pt_disable_trace(pt_context_t *ctx);

#endif
