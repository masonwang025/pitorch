#ifndef PITORCH_TEXT_H
#define PITORCH_TEXT_H

/*
 * Text layer: tokenizer, sampler, and generation loop.
 * Everything between "user types a string" and "model produces a string."
 */

#include <stdint.h>
#include "llama2.h"

/* ── tokenizer ── */

typedef struct {
    char *str;
    int id;
} pt_token_index_t;

typedef struct {
    char **vocab;
    float *vocab_scores;
    int vocab_size;
    int max_token_length;
    /* sorted index for encode (built lazily on first encode call) */
    int sorted_ready;
    pt_token_index_t *sorted_vocab;
} pt_tokenizer_t;

/* Parse tokenizer.bin into t. data points to the raw binary. */
void pt_tokenizer_init(pt_tokenizer_t *t, const void *data, int vocab_size);

/*
 * Load tokenizer from a combined model+tokenizer file.
 * The tokenizer binary immediately follows the model weights.
 * combined_file points to the start of the file (same pointer passed to pt_load_weights).
 */
void pt_load_tokenizer(pt_tokenizer_t *t, const void *combined_file, int vocab_size);

/* Decode token id to string. Handles BOS space-stripping and <0xHH> bytes. */
const char *pt_decode(pt_tokenizer_t *t, int prev_token, int token);

/* BPE-encode text into token ids. Caller provides output buffer (>= 512). */
void pt_encode(pt_tokenizer_t *t, const char *text, int bos, int eos,
               int *tokens, int *n_tokens);

/* ── sampler ── */

typedef struct {
    int vocab_size;
    float temperature;
    float topp;
    uint64_t rng_state;
} pt_sampler_t;

void pt_sampler_init(pt_sampler_t *s, int vocab_size,
                     float temp, float topp, uint64_t seed);

int pt_sample(pt_sampler_t *s, float *logits);

/* ── generation ── */

void pt_generate(const pt_config_t *cfg, const pt_weights_t *w, pt_state_t *s,
                 pt_tokenizer_t *tok, pt_sampler_t *sampler,
                 const char *prompt, int max_tokens, pt_matvec_fn matvec);

#endif
