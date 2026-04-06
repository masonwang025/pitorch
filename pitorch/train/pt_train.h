#ifndef PITORCH_TRAIN_H
#define PITORCH_TRAIN_H

/*
 * Training state and forward pass for Llama2.
 * Batched forward: processes T tokens at once, saves all intermediate
 * activations for the backward pass, computes cross-entropy loss.
 *
 * Portable C — no Pi-specific dependencies.
 */

#include "llama2.h"
#include "trace.h"

/*
 * Activations saved during the forward pass.
 * All arrays are flat; indexing helpers below.
 *
 * For a model with L layers, T tokens, dim D, kv_dim K, hidden H:
 *   residuals:     [(L+1) * T * D]     layer boundary residual stream
 *   residuals_att: [L * T * D]         post-attention residuals (input to FFN norm)
 *   xb_att:        [L * T * D]         attention rmsnorm output
 *   q:             [L * T * D]         post-RoPE queries
 *   k:             [L * T * K]         post-RoPE keys
 *   v:             [L * T * K]         values
 *   att:           [L * n_heads * T * T]   post-softmax attention weights
 *   att_out:       [L * T * D]         weighted value sums (input to Wo)
 *   xb_ffn:        [L * T * D]         FFN rmsnorm output
 *   hb:            [L * T * H]         w1 output (pre-SiLU, saved for backward)
 *   hb2:           [L * T * H]         w3 output (gate)
 *   x_final:       [T * D]             final rmsnorm output (input to Wcls)
 *   logits:        [T * V]             classifier output
 */
typedef struct {
    float *residuals;
    float *residuals_att;
    float *xb_att;
    float *q;
    float *k;
    float *v;
    float *att;
    float *att_out;
    float *xb_ffn;
    float *hb;
    float *hb2;
    float *x_final;
    float *logits;
    float *scratch;        /* [max(hidden_dim, dim)] reusable temp */
    int T;
} pt_activations_t;

#ifndef __RPI__
void pt_alloc_activations(pt_activations_t *a, const pt_config_t *cfg, int T);
void pt_free_activations(pt_activations_t *a);
#else
void pt_scratch_alloc_activations(pt_activations_t *a, const pt_config_t *cfg,
                                  int T, pt_bump_alloc_fn alloc);
#endif

/*
 * Batched forward pass for training.
 * Processes tokens[0..T-1], saves all activations into *a,
 * and returns the mean cross-entropy loss predicting tokens[1..T-1]
 * from logits at positions 0..T-2.
 */
float pt_forward_train(const pt_config_t *cfg, const pt_weights_t *w,
                       pt_activations_t *a, const int *tokens, int T,
                       pt_matvec_fn matvec, pt_trace_t *trace);

/*
 * Weight gradients. Same layout as pt_weights_t.
 * _mem is a single backing allocation; all pointers index into it.
 * If weights are shared (wcls == token_embedding), wcls aliases
 * token_embedding here too, so both backward paths accumulate correctly.
 */
typedef struct {
    float *token_embedding;
    float *rms_att_weight;
    float *wq, *wk, *wv, *wo;
    float *rms_ffn_weight;
    float *w1, *w2, *w3;
    float *rms_final_weight;
    float *wcls;
    float *_mem;
    int _n_params;
} pt_grads_t;

#ifndef __RPI__
void pt_alloc_grads(pt_grads_t *g, const pt_config_t *cfg, int shared_weights);
void pt_free_grads(pt_grads_t *g);
#else
void pt_scratch_alloc_grads(pt_grads_t *g, const pt_config_t *cfg,
                            int shared, unsigned base_addr);
#endif

void pt_zero_grads(pt_grads_t *g);

/*
 * Backward scratch buffers — temporary storage used during pt_backward.
 * Caller allocates (via pt_alloc_backward_buf or scratch allocator)
 * and passes to pt_backward.
 */
typedef struct {
    float *d_res;       /* [T * dim] */
    float *d_aout;      /* [T * dim] */
    float *d_q;         /* [T * dim] */
    float *d_k;         /* [T * kv_dim] */
    float *d_v;         /* [T * kv_dim] */
    float *d_logit;     /* [vocab_size] */
    float *d_xb;        /* [dim] */
    float *d_qpre;      /* [dim] */
    float *d_kpre;      /* [kv_dim] */
    float *d_hid;       /* [hidden_dim] */
    float *d_hid2;      /* [hidden_dim] */
    float *sh;          /* [hidden_dim] */
    float *d_att_s;     /* [T] */
    float *d_pre_sm;    /* [T] */
    float *w_transpose; /* [max(V*dim, hdim*dim)] for GPU transposed matvec */
    float *d_temp;      /* [max(dim, hdim)] accumulation scratch */
    float *d_logit_all; /* [T * vocab_size] all position loss gradients for batched weight grad */
    /* Batch buffers for deferred weight gradients (GEMM phase) */
    float *drt_wo_all;  /* [T * dim] saved d_res values for batched wo weight grad */
    float *d_qpre_all;  /* [T * dim] saved pre-RoPE query grads for batched wq weight grad */
    float *d_kpre_all;  /* [T * kv_dim] saved pre-RoPE key grads for batched wk weight grad */
    float *sh_all;      /* [T * hidden_dim] saved silu(hb)*hb2 for batched w2 weight grad */
    float *d_hid_all;   /* [T * hidden_dim] saved d_hid for batched w1 weight grad */
    float *d_hid2_all;  /* [T * hidden_dim] saved d_hid2 for batched w3 weight grad */
    float *drt_ffn_all; /* [T * dim] saved d_res for batched w2 weight grad */
} pt_backward_buf_t;

#ifndef __RPI__
void pt_alloc_backward_buf(pt_backward_buf_t *b,
                           const pt_config_t *cfg, int T);
void pt_free_backward_buf(pt_backward_buf_t *b);
#else
void pt_scratch_alloc_backward_buf(pt_backward_buf_t *b,
                                   const pt_config_t *cfg, int T,
                                   pt_bump_alloc_fn alloc);
#endif

/*
 * Full backward pass.
 * Given saved activations from pt_forward_train and the target tokens,
 * computes gradients for all weights and accumulates into *g.
 * Caller must zero *g before calling (via pt_zero_grads).
 */
/*
 * matvec: when non-NULL, GPU-accelerates transposed matvecs in the backward
 * pass (transpose W into b->w_transpose, call matvec). When NULL, CPU-only.
 */
void pt_backward(const pt_config_t *cfg, const pt_weights_t *w,
                 pt_grads_t *g, const pt_activations_t *a,
                 pt_backward_buf_t *b,
                 const int *tokens, int T,
                 pt_matvec_fn matvec, pt_trace_t *trace);

/* Vanilla SGD: w -= lr * grad for all parameters. */
void pt_sgd_update(pt_weights_t *w, const pt_grads_t *g,
                   float lr, const pt_config_t *cfg);

/* SGD for layer subsets (pipeline-parallel) */
void pt_sgd_update_layers(pt_weights_t *w, const pt_grads_t *g,
                          float lr, const pt_config_t *cfg,
                          int l_start, int l_end);
void pt_sgd_update_head(pt_weights_t *w, const pt_grads_t *g,
                        float lr, const pt_config_t *cfg);

/* ── Staged training (for pipeline-parallel) ────────────────── */

/*
 * Embedding lookup for batched training.
 * Writes tokens[0..T-1] into a->residuals[0..T-1].
 */
void pt_forward_train_embed(const pt_weights_t *w, pt_activations_t *a,
                            const int *tokens, int T, int dim);

/*
 * Run transformer layers [l_start, l_end) during training forward.
 * Reads from a->residuals[l_start * T * dim], writes to a->residuals[l_end * T * dim].
 * Saves all intermediate activations for backward.
 */
void pt_forward_train_layers_range(const pt_config_t *cfg, const pt_weights_t *w,
                                    pt_activations_t *a, int T,
                                    int l_start, int l_end,
                                    pt_matvec_fn matvec, pt_trace_t *trace);

/*
 * Final rmsnorm + classifier + cross-entropy loss.
 * Reads from a->residuals[L * T * dim], writes a->x_final and a->logits.
 * Returns mean loss.
 */
float pt_forward_train_head(const pt_config_t *cfg, const pt_weights_t *w,
                            pt_activations_t *a, const int *tokens, int T,
                            pt_matvec_fn matvec, pt_trace_t *trace);

/*
 * Backward through classifier + final rmsnorm.
 * Fills b->d_res[0..T*dim-1] with gradients flowing into the last layer.
 * Accumulates wcls and rms_final_weight gradients.
 */
void pt_backward_head(const pt_config_t *cfg, const pt_weights_t *w,
                      pt_grads_t *g, const pt_activations_t *a,
                      pt_backward_buf_t *b,
                      const int *tokens, int T,
                      pt_matvec_fn matvec, pt_trace_t *trace);

/*
 * Backward through transformer layers [l_end-1, ..., l_start].
 * On entry, b->d_res holds the gradient flowing in from the layer above.
 * On exit, b->d_res holds the gradient flowing out to the layer below.
 * Accumulates weight gradients for layers in [l_start, l_end).
 */
void pt_backward_layers_range(const pt_config_t *cfg, const pt_weights_t *w,
                               pt_grads_t *g, const pt_activations_t *a,
                               pt_backward_buf_t *b, int T,
                               int l_start, int l_end,
                               pt_matvec_fn matvec, pt_trace_t *trace);

/*
 * Backward through embedding.
 * Accumulates into g->token_embedding.
 */
void pt_backward_embed(pt_grads_t *g, const pt_backward_buf_t *b,
                       const int *tokens, int T, int dim);

#endif
