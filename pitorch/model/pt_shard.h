#ifndef PT_SHARD_H
#define PT_SHARD_H

/*
 * Weight shard file format for pipeline-parallel training.
 *
 * Each Pi loads only its shard (its layers + optionally embed/head),
 * not the full model. This cuts per-Pi memory and SD card boot time.
 *
 * Header: standard 28-byte llama2.c header (global config) followed
 * by a 28-byte shard extension with rank info and layer range.
 *
 * Weights follow in the same order as the standard format but only
 * for the owned components.
 */

#include "llama2.h"

#define PT_SHARD_MAGIC  0x53485244  /* "SHRD" */

typedef struct {
    int rank;
    int world_size;
    int l_start;        /* global layer start */
    int l_end;          /* global layer end */
    int n_local;        /* l_end - l_start */
    int has_embed;
    int has_head;
    int n_layers_global;
} pt_shard_info_t;

/*
 * Parse shard header (56 bytes total: 28 standard + 28 extension).
 * Fills global_cfg with the full model config (n_layers = global count).
 * Fills shard with rank/layer info.
 * Panics if shard_magic is missing.
 */
void pt_load_shard_header(pt_config_t *global_cfg, pt_shard_info_t *shard,
                          const void *data);

/*
 * Set weight pointers into the shard data blob.
 * data points to start of file (including the 56-byte header).
 * Only sets pointers for weights present in the shard:
 *   - token_embedding / wcls: only if has_embed (wcls aliases embed if shared)
 *   - per-layer weights: only for [0, n_local) local layers
 *   - rms_final_weight: only if has_head
 *   - freq_cis: always (needed for RoPE in every layer)
 * Pointers for absent components are set to NULL.
 */
void pt_load_shard_weights(pt_weights_t *w, const pt_config_t *global_cfg,
                           const pt_shard_info_t *shard, void *data);

/*
 * Compute shard file size from the header (for memory layout).
 */
unsigned pt_shard_file_size(const void *data);

#endif
