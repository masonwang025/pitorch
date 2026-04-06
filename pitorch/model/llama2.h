#ifndef PITORCH_LLAMA2_H
#define PITORCH_LLAMA2_H

/*
 * Llama2 model: config, weight layout, inference state, forward pass.
 * Portable C — no Pi-specific dependencies. Used on both Mac and Pi.
 * Weight format matches karpathy/llama2.c .bin checkpoints.
 */

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} pt_config_t;

typedef struct {
    float *token_embedding;   /* [vocab_size, dim] */
    float *rms_att_weight;    /* [n_layers, dim] */
    float *wq;                /* [n_layers, dim, dim] */
    float *wk;                /* [n_layers, kv_dim, dim] */
    float *wv;                /* [n_layers, kv_dim, dim] */
    float *wo;                /* [n_layers, dim, dim] */
    float *rms_ffn_weight;    /* [n_layers, dim] */
    float *w1;                /* [n_layers, hidden_dim, dim] */
    float *w2;                /* [n_layers, dim, hidden_dim] */
    float *w3;                /* [n_layers, hidden_dim, dim] */
    float *rms_final_weight;  /* [dim] */
    float *wcls;              /* [vocab_size, dim] (may alias token_embedding) */
} pt_weights_t;

typedef struct {
    float *x;            /* [dim] activation */
    float *xb;           /* [dim] after rmsnorm / attention output */
    float *xb2;          /* [dim] output projection scratch */
    float *q;            /* [dim] query */
    float *k;            /* [kv_dim] key */
    float *v;            /* [kv_dim] value */
    float *hb;           /* [hidden_dim] FFN hidden */
    float *hb2;          /* [hidden_dim] FFN gate */
    float *att;          /* [n_heads, seq_len] attention scores */
    float *logits;       /* [vocab_size] output logits */
    float *key_cache;    /* [n_layers, seq_len, kv_dim] */
    float *value_cache;  /* [n_layers, seq_len, kv_dim] */
} pt_state_t;

/* Parse 28-byte header. Handles negative vocab_size (non-shared weights). */
void pt_load_config(pt_config_t *cfg, const void *data);

/* Set weight pointers into a memory blob. data points to start of file (header included). */
void pt_load_weights(pt_weights_t *w, const pt_config_t *cfg, void *data);

typedef void *(*pt_bump_alloc_fn)(unsigned bytes);

#ifndef __RPI__
/* Allocate inference state buffers (uses malloc — host only). */
void pt_alloc_state(pt_state_t *s, const pt_config_t *cfg);

/* Free inference state buffers. */
void pt_free_state(pt_state_t *s);
#else
/* Allocate inference state from a bump allocator (Pi scratch region). */
void pt_scratch_alloc_state(pt_state_t *s, const pt_config_t *cfg,
                            pt_bump_alloc_fn alloc);
#endif

/* Compute expected .bin file size from the 28-byte header. */
unsigned pt_file_size(const void *data);

/* Matvec function signature: y = W @ x,  W is [out_dim, in_dim] row-major. */
typedef void (*pt_matvec_fn)(const float *W, const float *x, float *y,
                             int out_dim, int in_dim);

/* Forward pass: given token at position pos, fills s->logits.
 * Uses matvec for all linear layers (pass smatvec_cpu, or a GPU wrapper). */
void pt_forward(const pt_config_t *cfg, const pt_weights_t *w,
                pt_state_t *s, int token, int pos, pt_matvec_fn matvec);

/* ── Staged forward pass (for distributed inference) ── */

/* Look up token embedding into s->x. */
void pt_forward_embed(const pt_weights_t *w, float *x, int dim, int token);

/* Run transformer layers [l_start, l_end) on s->x. Updates KV cache. */
void pt_forward_layers_range(const pt_config_t *cfg, const pt_weights_t *w,
                              pt_state_t *s, int pos,
                              int l_start, int l_end,
                              pt_matvec_fn matvec);

/* Final rmsnorm + classifier projection. Fills s->logits. */
void pt_forward_head(const pt_config_t *cfg, const pt_weights_t *w,
                     pt_state_t *s, pt_matvec_fn matvec);

#endif
