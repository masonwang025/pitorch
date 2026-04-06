#ifndef PT_DIST_PIPELINE_H
#define PT_DIST_PIPELINE_H

/*
 * High-level distributed pipeline API.
 *
 * Wraps the ring topology, handshake, and per-rank forward/backward
 * communication into single-call functions. Designed so that distributed
 * training and inference look as simple as their single-Pi equivalents:
 *
 *   pt_dist_train_step(ctx, dist, tokens, T, lr);   // one line
 *   pt_dist_forward_step(ctx, dist, token);          // one line
 *
 * All ranks call the same function; the implementation dispatches
 * based on dist->has_embed / dist->has_head.
 */

#include "pt.h"
#include "pt_dist.h"

/* ── Setup ─────────────────────────────────────────────────── */

/*
 * Configure distributed context from shard info.
 * Sets layer range, embed/head ownership, and GPIO links.
 */
void pt_dist_setup(pt_dist_t *dist, const pt_shard_info_t *shard,
                   int rank, int world_size);

/*
 * Ring synchronization handshake.
 * Embed+head rank sends PING downstream around the ring.
 * All other ranks forward it. Confirms full ring connectivity.
 * Call this before the first forward/training step, and again
 * after pt_dist_reset_links() between training and inference.
 */
void pt_dist_ring_sync(pt_dist_t *dist);

/*
 * Reset GPIO links to idle state.
 * Required between training and inference phases because the
 * backward pass leaves links in reversed direction mode.
 */
void pt_dist_reset_links(pt_dist_t *dist);

/* ── Logging ───────────────────────────────────────────────── */

/*
 * When verbose > 0, pipeline functions print timestamped per-op logs:
 *
 *   [00:05.123] R3 step  0 | fwd: embedding           23ms
 *   [00:05.146] R3 step  0 | fwd: send                  3ms
 *   ...
 *   ════════════════════════════════════════════════════════
 *   STEP 0 | loss = 10.3142 | 170.2s/step
 *   ════════════════════════════════════════════════════════
 */
void pt_dist_set_verbose(pt_dist_t *dist, int verbose);

/* ── Training ──────────────────────────────────────────────── */

/* Result from a distributed training step. */
typedef struct {
    float    loss;       /* loss on embed+head rank, 0 on layer ranks */
    unsigned total_ms;   /* wall-clock step time in milliseconds */
} pt_dist_result_t;

/*
 * One distributed training step across all ranks.
 *
 * Embed+head rank: embed → send → recv → head_fwd → head_bwd → send → recv → embed_bwd → sgd
 * Layer ranks:     recv → forward_layers → send → recv → backward_layers → send → sgd
 *
 * Returns loss (on embed+head rank) and total step time.
 */
pt_dist_result_t pt_dist_train_step(pt_context_t *ctx, pt_dist_t *dist,
                                    const int *tokens, int T, float lr);

/* ── Inference ─────────────────────────────────────────────── */

/*
 * One distributed inference step (single-token forward).
 *
 * Embed+head rank: embed token → send → recv → head → argmax → returns next token.
 * Layer ranks:     recv → forward_layers → send → returns 0.
 *
 * The embed+head rank provides `token`; layer ranks ignore it.
 */
int pt_dist_forward_step(pt_context_t *ctx, pt_dist_t *dist, int token);

#endif
