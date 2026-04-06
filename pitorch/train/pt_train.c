#include <string.h>
#ifndef __RPI__
#include <stdlib.h>
#endif
#include "pt_train.h"
#include "pt_backward_ops.h"
#include "pt_ops.h"
#include "pt_math.h"
#include "trace.h"

#ifdef __RPI__
#include "rpi.h"
#include "gemm_rect.h"
#include "arena.h"
#define TLOG(...) printk(__VA_ARGS__)
#else
#define TLOG(...)
#endif

/* ── allocation (host only) ────────────────────────────────── */

#ifndef __RPI__
void pt_alloc_activations(pt_activations_t *a, const pt_config_t *cfg, int T) {
    int dim  = cfg->dim;
    int hdim = cfg->hidden_dim;
    int hd   = dim / cfg->n_heads;
    int kvd  = cfg->n_kv_heads * hd;
    int L    = cfg->n_layers;
    int nh   = cfg->n_heads;
    int V    = cfg->vocab_size;

    a->T            = T;
    a->residuals     = (float *)calloc((size_t)(L + 1) * T * dim, sizeof(float));
    a->residuals_att = (float *)calloc((size_t)L * T * dim, sizeof(float));
    a->xb_att        = (float *)calloc((size_t)L * T * dim, sizeof(float));
    a->q             = (float *)calloc((size_t)L * T * dim, sizeof(float));
    a->k             = (float *)calloc((size_t)L * T * kvd, sizeof(float));
    a->v             = (float *)calloc((size_t)L * T * kvd, sizeof(float));
    a->att           = (float *)calloc((size_t)L * nh * T * T, sizeof(float));
    a->att_out       = (float *)calloc((size_t)L * T * dim, sizeof(float));
    a->xb_ffn        = (float *)calloc((size_t)L * T * dim, sizeof(float));
    a->hb            = (float *)calloc((size_t)L * T * hdim, sizeof(float));
    a->hb2           = (float *)calloc((size_t)L * T * hdim, sizeof(float));
    a->x_final       = (float *)calloc((size_t)T * dim, sizeof(float));
    a->logits        = (float *)calloc((size_t)T * V, sizeof(float));
    int scratch_sz   = hdim > dim ? hdim : dim;
    a->scratch       = (float *)calloc((size_t)scratch_sz, sizeof(float));
}

void pt_free_activations(pt_activations_t *a) {
    free(a->residuals);     free(a->residuals_att);
    free(a->xb_att);        free(a->q);
    free(a->k);             free(a->v);
    free(a->att);           free(a->att_out);
    free(a->xb_ffn);        free(a->hb);
    free(a->hb2);           free(a->x_final);
    free(a->logits);        free(a->scratch);
}
#else /* __RPI__ */

void pt_scratch_alloc_activations(pt_activations_t *a, const pt_config_t *cfg,
                                  int T, pt_bump_alloc_fn alloc) {
    int dim  = cfg->dim;
    int hdim = cfg->hidden_dim;
    int hd   = dim / cfg->n_heads;
    int kvd  = cfg->n_kv_heads * hd;
    int L    = cfg->n_layers;
    int nh   = cfg->n_heads;
    int V    = cfg->vocab_size;
    a->T            = T;
    a->residuals     = alloc((L + 1) * T * dim * 4);
    a->residuals_att = alloc(L * T * dim * 4);
    a->xb_att        = alloc(L * T * dim * 4);
    a->q             = alloc(L * T * dim * 4);
    a->k             = alloc(L * T * kvd * 4);
    a->v             = alloc(L * T * kvd * 4);
    a->att           = alloc(L * nh * T * T * 4);
    a->att_out       = alloc(L * T * dim * 4);
    a->xb_ffn        = alloc(L * T * dim * 4);
    a->hb            = alloc(L * T * hdim * 4);
    a->hb2           = alloc(L * T * hdim * 4);
    a->x_final       = alloc(T * dim * 4);
    a->logits        = alloc(T * V * 4);
    int sz = hdim > dim ? hdim : dim;
    a->scratch       = alloc(sz * 4);
}

void pt_scratch_alloc_backward_buf(pt_backward_buf_t *b,
                                   const pt_config_t *cfg, int T,
                                   pt_bump_alloc_fn alloc) {
    int dim  = cfg->dim;
    int hdim = cfg->hidden_dim;
    int hd   = dim / cfg->n_heads;
    int kvd  = cfg->n_kv_heads * hd;
    int V    = cfg->vocab_size;
    b->d_res    = alloc(T * dim * 4);
    b->d_aout   = alloc(T * dim * 4);
    b->d_q      = alloc(T * dim * 4);
    b->d_k      = alloc(T * kvd * 4);
    b->d_v      = alloc(T * kvd * 4);
    b->d_logit  = alloc(V * 4);
    b->d_xb     = alloc(dim * 4);
    b->d_qpre   = alloc(dim * 4);
    b->d_kpre   = alloc(kvd * 4);
    b->d_hid    = alloc(hdim * 4);
    b->d_hid2   = alloc(hdim * 4);
    b->sh       = alloc(hdim * 4);
    b->d_att_s  = alloc(T * 4);
    b->d_pre_sm = alloc(T * 4);
    b->d_logit_all = alloc(T * V * 4);
    /* Batch buffers for deferred weight gradients */
    b->drt_wo_all  = alloc(T * dim * 4);
    b->d_qpre_all  = alloc(T * dim * 4);
    b->d_kpre_all  = alloc(T * kvd * 4);
    b->sh_all      = alloc(T * hdim * 4);
    b->d_hid_all   = alloc(T * hdim * 4);
    b->d_hid2_all  = alloc(T * hdim * 4);
    b->drt_ffn_all = alloc(T * dim * 4);
    /* w_transpose and d_temp allocated separately (large, may need fixed addr) */
    b->w_transpose = 0;
    b->d_temp      = 0;
}

void pt_scratch_alloc_grads(pt_grads_t *g, const pt_config_t *cfg,
                            int shared, unsigned base_addr) {
    int dim  = cfg->dim;
    int hdim = cfg->hidden_dim;
    int hd   = dim / cfg->n_heads;
    int kvd  = cfg->n_kv_heads * hd;
    int L    = cfg->n_layers;
    int V    = cfg->vocab_size;

    unsigned total = (unsigned)V * dim
                   + (unsigned)L * dim
                   + (unsigned)L * dim * dim
                   + (unsigned)L * kvd * dim
                   + (unsigned)L * kvd * dim
                   + (unsigned)L * dim * dim
                   + (unsigned)L * dim
                   + (unsigned)L * hdim * dim
                   + (unsigned)L * dim * hdim
                   + (unsigned)L * hdim * dim
                   + (unsigned)dim;
    if (!shared) total += (unsigned)V * dim;

    TLOG("(n=%d addr=0x%X ", total, base_addr);
    g->_mem = (float *)base_addr;
    g->_n_params = (int)total;
    TLOG("memset..");
    memset(g->_mem, 0, (size_t)total * sizeof(float));
    TLOG("done) ");

    float *p = g->_mem;
    g->token_embedding = p; p += V * dim;
    g->rms_att_weight  = p; p += L * dim;
    g->wq = p; p += L * dim * dim;
    g->wk = p; p += L * kvd * dim;
    g->wv = p; p += L * kvd * dim;
    g->wo = p; p += L * dim * dim;
    g->rms_ffn_weight = p; p += L * dim;
    g->w1 = p; p += L * hdim * dim;
    g->w2 = p; p += L * dim * hdim;
    g->w3 = p; p += L * hdim * dim;
    g->rms_final_weight = p; p += dim;
    g->wcls = shared ? g->token_embedding : p;
}
#endif

/* ── batched forward pass ──────────────────────────────────── */

float pt_forward_train(const pt_config_t *cfg, const pt_weights_t *w,
                       pt_activations_t *a, const int *tokens, int T,
                       pt_matvec_fn matvec, pt_trace_t *trace) {
    int dim      = cfg->dim;
    int hdim     = cfg->hidden_dim;
    int nh       = cfg->n_heads;
    int hd       = dim / nh;
    int kvd      = cfg->n_kv_heads * hd;
    int kv_mul   = nh / cfg->n_kv_heads;
    int L        = cfg->n_layers;
    int V        = cfg->vocab_size;
    float scale  = 1.0f / pt_sqrtf((float)hd);

    a->T = T;

    /* ── embedding ── */
    TLOG("[fwd:emb]");
    pt_trace_begin(trace, "embedding", "fwd", -1);
    for (int t = 0; t < T; t++)
        embedding_lookup(a->residuals + t * dim,
                         w->token_embedding, dim, tokens[t]);
    pt_trace_end(trace);

    /* ── transformer layers ── */
    for (int l = 0; l < L; l++) {
        TLOG("[fwd:L%d", l);
        pt_trace_begin(trace, "layer", "fwd", l);
        float *res_in  = a->residuals     + l * T * dim;
        float *res_mid = a->residuals_att + l * T * dim;
        float *res_out = a->residuals     + (l + 1) * T * dim;

        /* ── QKV projections + RoPE for all positions ── */
        pt_trace_begin(trace, "qkv", "fwd", l);
        for (int t = 0; t < T; t++) {
            float *xb = a->xb_att + (l * T + t) * dim;
            rmsnorm(xb, res_in + t * dim,
                    w->rms_att_weight + l * dim, dim);

            float *qt = a->q + (l * T + t) * dim;
            float *kt = a->k + (l * T + t) * kvd;
            float *vt = a->v + (l * T + t) * kvd;

            matvec(w->wq + l * dim * dim,   xb, qt, dim, dim);
            matvec(w->wk + l * kvd * dim,   xb, kt, kvd, dim);
            matvec(w->wv + l * kvd * dim,   xb, vt, kvd, dim);

            rope(qt, kt, dim, hd, t);
        }
        pt_trace_end(trace);

        TLOG(":qkv");
        /* ── multi-head causal attention + output projection ── */
        pt_trace_begin(trace, "attention", "fwd", l);
        for (int t = 0; t < T; t++) {
            float *qt   = a->q       + (l * T + t) * dim;
            float *ao_t = a->att_out + (l * T + t) * dim;

            for (int h = 0; h < nh; h++) {
                float *q_h   = qt + h * hd;
                float *att_h = a->att + ((l * nh + h) * T + t) * T;

                for (int t2 = 0; t2 <= t; t2++) {
                    float *k_t2 = a->k + (l * T + t2) * kvd
                                  + (h / kv_mul) * hd;
                    float score = 0.0f;
                    for (int i = 0; i < hd; i++)
                        score += q_h[i] * k_t2[i];
                    att_h[t2] = score * scale;
                }
                for (int t2 = t + 1; t2 < T; t2++)
                    att_h[t2] = 0.0f;

                softmax(att_h, t + 1);

                float *out_h = ao_t + h * hd;
                memset(out_h, 0, hd * sizeof(float));
                for (int t2 = 0; t2 <= t; t2++) {
                    float *v_t2 = a->v + (l * T + t2) * kvd
                                  + (h / kv_mul) * hd;
                    float a_val = att_h[t2];
                    for (int i = 0; i < hd; i++)
                        out_h[i] += a_val * v_t2[i];
                }
            }

            /* Wo projection + attention residual */
            matvec(w->wo + l * dim * dim, ao_t, a->scratch, dim, dim);
            vec_add(res_mid + t * dim, res_in + t * dim, a->scratch, dim);
        }
        pt_trace_end(trace);

        TLOG(":att");
        /* ── feed-forward (SwiGLU) ── */
        pt_trace_begin(trace, "ffn", "fwd", l);
        for (int t = 0; t < T; t++) {
            float *xb  = a->xb_ffn + (l * T + t) * dim;
            rmsnorm(xb, res_mid + t * dim,
                    w->rms_ffn_weight + l * dim, dim);

            float *hbt  = a->hb  + (l * T + t) * hdim;
            float *hb2t = a->hb2 + (l * T + t) * hdim;

            matvec(w->w1 + l * hdim * dim, xb, hbt,  hdim, dim);
            matvec(w->w3 + l * hdim * dim, xb, hb2t, hdim, dim);

            /*
             * Compute silu(hb) * hb2 into scratch.
             * hb and hb2 are preserved (pre-silu) for backward.
             */
            float *sh = a->scratch;
            for (int i = 0; i < hdim; i++) {
                float sig = 1.0f / (1.0f + pt_expf(-hbt[i]));
                sh[i] = (hbt[i] * sig) * hb2t[i];
            }

            matvec(w->w2 + l * dim * hdim, sh, res_out + t * dim, dim, hdim);
            vec_add(res_out + t * dim, res_out + t * dim,
                    res_mid + t * dim, dim);
        }
        pt_trace_end(trace);
        pt_trace_end(trace); /* end layer */
        TLOG(":ffn]");
    }

    TLOG("[fwd:cls]");
    /* ── final rmsnorm + classifier ── */
    pt_trace_begin(trace, "classifier", "fwd", -1);
    float *res_final = a->residuals + L * T * dim;
    for (int t = 0; t < T; t++) {
        rmsnorm(a->x_final + t * dim, res_final + t * dim,
                w->rms_final_weight, dim);
        matvec(w->wcls, a->x_final + t * dim,
               a->logits + t * V, V, dim);
    }
    pt_trace_end(trace);

    TLOG("[fwd:loss]");
    /* ── cross-entropy loss: positions 0..T-2 predict tokens 1..T-1 ── */
    pt_trace_begin(trace, "loss", "fwd", -1);
    float loss = 0.0f;
    for (int t = 0; t < T - 1; t++) {
        float *lg = a->logits + t * V;
        int target = tokens[t + 1];

        float max_lg = lg[0];
        for (int v = 1; v < V; v++)
            if (lg[v] > max_lg) max_lg = lg[v];

        float sum = 0.0f;
        for (int v = 0; v < V; v++)
            sum += pt_expf(lg[v] - max_lg);

        loss += -(lg[target] - max_lg - pt_logf(sum));
    }
    loss /= (float)(T - 1);
    pt_trace_end(trace);

    return loss;
}

/* ── gradient allocation (host only) ──────────────────────── */

#ifndef __RPI__
void pt_alloc_grads(pt_grads_t *g, const pt_config_t *cfg, int shared) {
    int dim  = cfg->dim;
    int hdim = cfg->hidden_dim;
    int hd   = dim / cfg->n_heads;
    int kvd  = cfg->n_kv_heads * hd;
    int L    = cfg->n_layers;
    int V    = cfg->vocab_size;

    size_t n = (size_t)V * dim
             + (size_t)L * dim
             + (size_t)L * dim * dim
             + (size_t)L * kvd * dim
             + (size_t)L * kvd * dim
             + (size_t)L * dim * dim
             + (size_t)L * dim
             + (size_t)L * hdim * dim
             + (size_t)L * dim * hdim
             + (size_t)L * hdim * dim
             + (size_t)dim;
    if (!shared) n += (size_t)V * dim;

    g->_mem = (float *)calloc(n, sizeof(float));
    g->_n_params = (int)n;

    float *p = g->_mem;
    g->token_embedding = p; p += V * dim;
    g->rms_att_weight  = p; p += L * dim;
    g->wq = p; p += L * dim * dim;
    g->wk = p; p += L * kvd * dim;
    g->wv = p; p += L * kvd * dim;
    g->wo = p; p += L * dim * dim;
    g->rms_ffn_weight = p; p += L * dim;
    g->w1 = p; p += L * hdim * dim;
    g->w2 = p; p += L * dim * hdim;
    g->w3 = p; p += L * hdim * dim;
    g->rms_final_weight = p; p += dim;
    g->wcls = shared ? g->token_embedding : p;
}

void pt_free_grads(pt_grads_t *g) { free(g->_mem); }

void pt_alloc_backward_buf(pt_backward_buf_t *b,
                           const pt_config_t *cfg, int T) {
    int dim  = cfg->dim;
    int hdim = cfg->hidden_dim;
    int hd   = dim / cfg->n_heads;
    int kvd  = cfg->n_kv_heads * hd;
    int V    = cfg->vocab_size;
    b->d_res    = (float *)calloc((size_t)T * dim,  sizeof(float));
    b->d_aout   = (float *)calloc((size_t)T * dim,  sizeof(float));
    b->d_q      = (float *)calloc((size_t)T * dim,  sizeof(float));
    b->d_k      = (float *)calloc((size_t)T * kvd,  sizeof(float));
    b->d_v      = (float *)calloc((size_t)T * kvd,  sizeof(float));
    b->d_logit  = (float *)calloc((size_t)V,        sizeof(float));
    b->d_xb     = (float *)calloc((size_t)dim,      sizeof(float));
    b->d_qpre   = (float *)calloc((size_t)dim,      sizeof(float));
    b->d_kpre   = (float *)calloc((size_t)kvd,      sizeof(float));
    b->d_hid    = (float *)calloc((size_t)hdim,     sizeof(float));
    b->d_hid2   = (float *)calloc((size_t)hdim,     sizeof(float));
    b->sh       = (float *)calloc((size_t)hdim,     sizeof(float));
    b->d_att_s  = (float *)calloc((size_t)T,        sizeof(float));
    b->d_pre_sm = (float *)calloc((size_t)T,        sizeof(float));
    b->d_logit_all = (float *)calloc((size_t)T * V,  sizeof(float));
    /* Batch buffers for deferred weight gradients */
    b->drt_wo_all  = (float *)calloc((size_t)T * dim,  sizeof(float));
    b->d_qpre_all  = (float *)calloc((size_t)T * dim,  sizeof(float));
    b->d_kpre_all  = (float *)calloc((size_t)T * kvd,  sizeof(float));
    b->sh_all      = (float *)calloc((size_t)T * hdim, sizeof(float));
    b->d_hid_all   = (float *)calloc((size_t)T * hdim, sizeof(float));
    b->d_hid2_all  = (float *)calloc((size_t)T * hdim, sizeof(float));
    b->drt_ffn_all = (float *)calloc((size_t)T * dim,  sizeof(float));
    size_t wt_sz = (size_t)V * dim;
    size_t hd_sz = (size_t)hdim * dim;
    b->w_transpose = (float *)calloc(wt_sz > hd_sz ? wt_sz : hd_sz, sizeof(float));
    b->d_temp      = (float *)calloc(dim > hdim ? dim : hdim, sizeof(float));
}

void pt_free_backward_buf(pt_backward_buf_t *b) {
    free(b->d_res);    free(b->d_aout);  free(b->d_q);
    free(b->d_k);      free(b->d_v);     free(b->d_logit);
    free(b->d_xb);     free(b->d_qpre);  free(b->d_kpre);
    free(b->d_hid);    free(b->d_hid2);  free(b->sh);
    free(b->d_att_s);  free(b->d_pre_sm);
    free(b->d_logit_all);
    free(b->drt_wo_all); free(b->d_qpre_all); free(b->d_kpre_all);
    free(b->sh_all); free(b->d_hid_all); free(b->d_hid2_all);
    free(b->drt_ffn_all);
    free(b->w_transpose); free(b->d_temp);
}
#endif

void pt_zero_grads(pt_grads_t *g) {
    memset(g->_mem, 0, (size_t)g->_n_params * sizeof(float));
}

/* ── transpose helper ─────────────────────────────────────── */

static void transpose(float *out, const float *in, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j * rows + i] = in[i * cols + j];
}

/*
 * GPU-accelerated dx += W^T @ dy, using a PRE-TRANSPOSED W.
 * w_t must already hold W^T (call transpose() before the position loop).
 * If accumulate: adds result to dx. If !accumulate: overwrites dx.
 */
static void backward_input_pretrans(
    float *d_x, const float *w_t, const float *d_y,
    int out_dim, int in_dim,
    float *d_temp, pt_matvec_fn matvec, int accumulate)
{
    if (accumulate) {
        matvec(w_t, d_y, d_temp, in_dim, out_dim);
        for (int i = 0; i < in_dim; i++) d_x[i] += d_temp[i];
    } else {
        matvec(w_t, d_y, d_x, in_dim, out_dim);
    }
}

/*
 * Batched weight gradient: d_W += sum_t d_y[t] ⊗ x[t]^T
 * Tiles the outer dimension so each tile of d_W stays cache-hot while
 * accumulating contributions from all T positions.
 *
 * Tile sizing is critical: each tile of d_W must fit in L1 cache.
 *   Pi Zero (ARM1176, 16KB L1, no L2): BW_TILE=6 → 6×512×4 = 12KB
 *     Leaves 4KB for x vector (2KB) and loop temporaries.
 *     Old BW_TILE=128 created 256KB tiles → every access missed L1 → 21.7s.
 *   Mac/x86 (32KB+ L1, large L2): BW_TILE=128 → tile fits in L2.
 */
#ifdef __RPI__
#  ifndef BW_TILE
#  define BW_TILE 6
#  endif
#else
#  ifndef BW_TILE
#  define BW_TILE 128
#  endif
#endif
void matmul_backward_weight_batched(
    float *d_W, const float *d_y_all, const float *x_all,
    int out_dim, int in_dim, int T)
{
    for (int i0 = 0; i0 < out_dim; i0 += BW_TILE) {
        int i1 = (i0 + BW_TILE < out_dim) ? i0 + BW_TILE : out_dim;
        for (int t = 0; t < T; t++) {
            const float *dy = d_y_all + t * out_dim;
            const float *x  = x_all  + t * in_dim;
            for (int i = i0; i < i1; i++) {
                float dyi = dy[i];
                float *dw = d_W + i * in_dim;
                for (int j = 0; j < in_dim; j++)
                    dw[j] += dyi * x[j];
            }
        }
    }
}
#undef BW_TILE

/*
 * GPU-accelerated batched weight gradient using rectangular GEMM.
 *
 * dW[out_dim × in_dim] += D^T[out_dim × B] × X[B × in_dim]
 *
 * D = d_y_all [(B) × out_dim], X = x_all [(B) × in_dim].
 * Transposes D into scratch, then launches GEMM on 12 QPUs.
 *
 * Requires: in_dim % 16 == 0 (for QPU 16-wide SIMD writes).
 * scratch must hold at least out_dim * B floats.
 */
#ifdef __RPI__
/*
 * GPU-accelerated classifier weight gradient via rectangular GEMM.
 *
 *   dW[out_dim × in_dim] += dY^T[out_dim × B] × X[B × in_dim]
 *
 * Transposes dY from [B × out_dim] to [out_dim × B] into scratch,
 * then dispatches a single sgemm_rect_tmu call. Requires an arena
 * large enough for the full C buffer (out_dim × in_dim × 4 bytes)
 * plus uniforms (~out_dim × B × 4 bytes) and code/ptrs (~1 KB).
 *
 * For stories15M (V=32000, dim=288, B=7): C=36.8 MB, uniforms=0.9 MB → ~38 MB.
 * For stories42M (V=32000, dim=512, B=7): C=65.5 MB, uniforms=0.9 MB → ~67 MB.
 * For stories110M (V=32000, dim=768, B=7): C=98.3 MB, uniforms=0.9 MB → ~100 MB.
 *
 * With gpu_mem=128 and a 100 MB arena, all models fit in a single dispatch.
 *
 * NOTE: If the arena is too small (e.g. gpu_mem=64 with a 2 MB arena),
 * M-tiling can be used: tile out_dim into chunks of M_TILE rows, calling
 * gpu_arena_reset() between chunks. Each chunk needs M_TILE × in_dim × 4
 * bytes for C plus M_TILE × B × 4 for uniforms. For example, M_TILE=1024
 * with N=288 uses ~1.2 MB per chunk, requiring ~32 chunks for V=32000.
 * This adds overhead from repeated arena reset + C buffer copy-in/copy-out.
 */
static void matmul_backward_weight_gpu(
    float *d_W, const float *d_y_all, const float *x_all,
    int out_dim, int in_dim, int B, float *scratch)
{
    /* Transpose d_y_all from [B × out_dim] to [out_dim × B] into scratch.
     * scratch must hold out_dim * B floats. */
    for (int t = 0; t < B; t++)
        for (int i = 0; i < out_dim; i++)
            scratch[i * B + t] = d_y_all[t * out_dim + i];

    gpu_arena_reset();
    sgemm_rect_tmu(scratch, x_all, d_W,
                    out_dim, B, in_dim,
                    12, /*accumulate=*/1);
}
#endif

/* ── full backward pass ───────────────────────────────────── */

void pt_backward(const pt_config_t *cfg, const pt_weights_t *w,
                 pt_grads_t *g, const pt_activations_t *a,
                 pt_backward_buf_t *b,
                 const int *tokens, int T,
                 pt_matvec_fn matvec, pt_trace_t *trace) {
    int dim    = cfg->dim;
    int hdim   = cfg->hidden_dim;
    int nh     = cfg->n_heads;
    int hd     = dim / nh;
    int kvd    = cfg->n_kv_heads * hd;
    int kv_mul = nh / cfg->n_kv_heads;
    int L      = cfg->n_layers;
    int V      = cfg->vocab_size;
    float scale = 1.0f / pt_sqrtf((float)hd);

    float *d_res    = b->d_res;
    float *d_aout   = b->d_aout;
    float *d_q      = b->d_q;
    float *d_k      = b->d_k;
    float *d_v      = b->d_v;
    /* d_logit single-position buffer no longer needed — using d_logit_all */
    float *d_xb     = b->d_xb;
    float *d_qpre   = b->d_qpre;
    float *d_kpre   = b->d_kpre;
    float *d_hid    = b->d_hid;
    float *d_hid2   = b->d_hid2;
    float *d_att_s  = b->d_att_s;
    float *d_pre_sm = b->d_pre_sm;
    float *w_t      = b->w_transpose;
    float *d_temp   = b->d_temp;
    int gpu = (matvec && w_t && d_temp);

    memset(d_res, 0, (size_t)T * dim * sizeof(float));

    TLOG("[bwd:cls:");
    /* ── loss + classifier + final rmsnorm backward ─────── */
    pt_trace_begin(trace, "classifier", "bwd", -1);
    float inv_T1 = 1.0f / (float)(T - 1);

    /* Hoist classifier transpose: wcls is [V, dim] → w_t = wcls^T [dim, V].
     * Previously transposed T times (once per position); now once. */
    if (gpu) {
        pt_trace_begin(trace, "cls_transpose", "bwd", -1);
        transpose(w_t, w->wcls, V, dim);
        pt_trace_end(trace);
    }

    /*
     * Position T-1 has no loss signal (no next token), so d_logit is all zeros.
     * Every operation for that position (matvec, outer product, rmsnorm_backward)
     * produces zeros — pure waste. Skip it. This propagates: d_res[T-1] stays 0,
     * so all per-layer backward work for position T-1 is also zero.
     *
     * Phase 1: compute all d_logit vectors and store in d_logit_all.
     * Phase 2: batched weight gradient — single pass over g->wcls.
     * Phase 3: input gradient + rmsnorm backward for each position.
     */
    float *d_logit_all = b->d_logit_all;

    /* Phase 1: loss backward for all positions → d_logit_all[t*V .. (t+1)*V-1] */
    for (int t = 0; t < T - 1; t++) {
        TLOG("%d", t);
        float *dl = d_logit_all + t * V;
        float *lg = a->logits + t * V;
        int target = tokens[t + 1];
        float mx = lg[0];
        for (int v = 1; v < V; v++) if (lg[v] > mx) mx = lg[v];
        float sm = 0.0f;
        for (int v = 0; v < V; v++) { dl[v] = pt_expf(lg[v] - mx); sm += dl[v]; }
        float inv = 1.0f / sm;
        for (int v = 0; v < V; v++) dl[v] *= inv * inv_T1;
        dl[target] -= inv_T1;
    }

    /* Phase 2: input gradient + rmsnorm backward per position.
     * Must happen before GPU weight grad because the GPU GEMM reuses w_t. */
    for (int t = 0; t < T - 1; t++) {
        float *dl = d_logit_all + t * V;

        if (gpu)
            backward_input_pretrans(d_xb, w_t, dl, V, dim, d_temp, matvec, 0);
        else {
            memset(d_xb, 0, dim * sizeof(float));
            matmul_backward_input(d_xb, w->wcls, dl, V, dim);
        }

        rmsnorm_backward(d_res + t * dim, g->rms_final_weight,
                         d_xb, a->residuals + L * T * dim + t * dim,
                         w->rms_final_weight, dim);
    }

    /* Phase 3: classifier weight gradient.
     * GPU path: rectangular GEMM — dW[V×dim] += D^T[V×(T-1)] × X[(T-1)×dim].
     * w_t is now free (input grads done), reused as GEMM A-transpose scratch.
     * CPU path: cache-tiled batched outer products. */
    pt_trace_begin(trace, "cls_weight_grad", "bwd", -1);
#if defined(__RPI__) && defined(GEMM_BWD_ENABLED)
    if (gpu)
        matmul_backward_weight_gpu(g->wcls, d_logit_all, a->x_final,
                                    V, dim, T - 1, w_t);
    else
#endif
    matmul_backward_weight_batched(g->wcls, d_logit_all, a->x_final, V, dim, T - 1);
    pt_trace_end(trace);
    pt_trace_end(trace);

    TLOG("]");
    /* ── layer backward (L-1 → 0) ─────────────────────────── */
    for (int l = L - 1; l >= 0; l--) {
        TLOG("[bwd:L%d", l);
        pt_trace_begin(trace, "layer", "bwd", l);

        /*
         * Pre-transpose all layer weight matrices into sub-regions of w_t.
         * w_t holds max(V*dim, hdim*dim) floats — the per-layer matrices
         * total ~1M floats, well within bounds.
         * This replaces ~10×T transposes with 7 transposes per layer.
         */
        float *wo_t = NULL, *w2_t = NULL, *w1_t = NULL, *w3_t = NULL;
        float *wq_t = NULL, *wk_t = NULL, *wv_t = NULL;
        if (gpu) {
            pt_trace_begin(trace, "transpose", "bwd", l);
            float *p = w_t;
            wo_t = p; transpose(wo_t, w->wo + l * dim * dim,   dim,  dim);  p += dim * dim;
            w2_t = p; transpose(w2_t, w->w2 + l * dim * hdim,  dim,  hdim); p += dim * hdim;
            w1_t = p; transpose(w1_t, w->w1 + l * hdim * dim,  hdim, dim);  p += hdim * dim;
            w3_t = p; transpose(w3_t, w->w3 + l * hdim * dim,  hdim, dim);  p += hdim * dim;
            wq_t = p; transpose(wq_t, w->wq + l * dim * dim,   dim,  dim);  p += dim * dim;
            wk_t = p; transpose(wk_t, w->wk + l * kvd * dim,   kvd,  dim);  p += kvd * dim;
            wv_t = p; transpose(wv_t, w->wv + l * kvd * dim,   kvd,  dim);
            pt_trace_end(trace);
        }

        /* ── FFN backward ── */
        pt_trace_begin(trace, "ffn", "bwd", l);

        /* Phase 1: per-position input grads + collect vectors for batched weight grads */
        float *sh_all      = b->sh_all;
        float *d_hid_all   = b->d_hid_all;
        float *d_hid2_all  = b->d_hid2_all;
        float *drt_ffn_all = b->drt_ffn_all;

        for (int t = 0; t < T - 1; t++) {
            float *drt  = d_res + t * dim;
            float *hbt  = a->hb  + (l * T + t) * hdim;
            float *hb2t = a->hb2 + (l * T + t) * hdim;

            /* recompute silu(hb) * hb2 for w2 weight gradient */
            float *sh_t = sh_all + t * hdim;
            for (int i = 0; i < hdim; i++) {
                float sig = 1.0f / (1.0f + pt_expf(-hbt[i]));
                sh_t[i] = (hbt[i] * sig) * hb2t[i];
            }

            /* save drt before rmsnorm modifies it (for w2 weight grad batch) */
            memcpy(drt_ffn_all + t * dim, drt, dim * sizeof(float));

            /* w2 backward: input gradient */
            if (gpu)
                backward_input_pretrans(d_hid, w2_t, drt, dim, hdim, d_temp, matvec, 0);
            else {
                memset(d_hid, 0, hdim * sizeof(float));
                matmul_backward_input(d_hid, w->w2 + l * dim * hdim, drt, dim, hdim);
            }

            /* silu * gate backward */
            for (int i = 0; i < hdim; i++) {
                float sig = 1.0f / (1.0f + pt_expf(-hbt[i]));
                float silu_val = hbt[i] * sig;
                float d_silu = d_hid[i] * hb2t[i];
                d_hid2[i] = d_hid[i] * silu_val;
                d_hid[i] = d_silu * sig * (1.0f + hbt[i] * (1.0f - sig));
            }

            /* save d_hid, d_hid2 for batched weight grad */
            memcpy(d_hid_all + t * hdim, d_hid, hdim * sizeof(float));
            memcpy(d_hid2_all + t * hdim, d_hid2, hdim * sizeof(float));

            /* w1 and w3 backward: input gradients */
            if (gpu) {
                backward_input_pretrans(d_xb, w1_t, d_hid,  hdim, dim, d_temp, matvec, 0);
                backward_input_pretrans(d_xb, w3_t, d_hid2, hdim, dim, d_temp, matvec, 1);
            } else {
                memset(d_xb, 0, dim * sizeof(float));
                matmul_backward_input(d_xb, w->w1 + l * hdim * dim, d_hid,  hdim, dim);
                matmul_backward_input(d_xb, w->w3 + l * hdim * dim, d_hid2, hdim, dim);
            }

            /* FFN rmsnorm backward → d_res becomes d_res_mid */
            rmsnorm_backward(drt, g->rms_ffn_weight + l * dim,
                             d_xb, a->residuals_att + (l * T + t) * dim,
                             w->rms_ffn_weight + l * dim, dim);
        }

        /* Phase 2: batched weight gradients for w2, w1, w3
         * Reduces T-1 passes over each gradient matrix to 1 tiled pass. */
        matmul_backward_weight_batched(g->w2 + l * dim * hdim,
            drt_ffn_all, sh_all, dim, hdim, T - 1);
        matmul_backward_weight_batched(g->w1 + l * hdim * dim,
            d_hid_all, a->xb_ffn + l * T * dim, hdim, dim, T - 1);
        matmul_backward_weight_batched(g->w3 + l * hdim * dim,
            d_hid2_all, a->xb_ffn + l * T * dim, hdim, dim, T - 1);
        pt_trace_end(trace);

        TLOG(":ffn:");
        /* ── attention backward ── */

        /* wo backward for all positions */
        pt_trace_begin(trace, "wo", "bwd", l);
        memset(d_aout, 0, (size_t)T * dim * sizeof(float));
        float *drt_wo_all = b->drt_wo_all;
        for (int t = 0; t < T - 1; t++) {
            float *drt = d_res + t * dim;
            if (gpu)
                backward_input_pretrans(d_aout + t * dim, wo_t, drt, dim, dim, d_temp, matvec, 0);
            else
                matmul_backward_input(d_aout + t * dim, w->wo + l * dim * dim, drt, dim, dim);
            memcpy(drt_wo_all + t * dim, drt, dim * sizeof(float));
        }
        /* Batched wo weight gradient */
        matmul_backward_weight_batched(g->wo + l * dim * dim,
            drt_wo_all, a->att_out + l * T * dim, dim, dim, T - 1);
        pt_trace_end(trace);

        TLOG(":wo");
        /* attention mechanism backward */
        pt_trace_begin(trace, "attention", "bwd", l);
        memset(d_q, 0, (size_t)T * dim * sizeof(float));
        memset(d_k, 0, (size_t)T * kvd * sizeof(float));
        memset(d_v, 0, (size_t)T * kvd * sizeof(float));

        for (int t = 0; t < T - 1; t++) {
            for (int h = 0; h < nh; h++) {
                float *d_ao_h = d_aout + t * dim + h * hd;
                float *att_h  = a->att + ((l * nh + h) * T + t) * T;
                int kv_off    = (h / kv_mul) * hd;

                /* value weighting backward */
                memset(d_att_s, 0, T * sizeof(float));
                for (int t2 = 0; t2 <= t; t2++) {
                    float *v_h = a->v + (l * T + t2) * kvd + kv_off;
                    float aw   = att_h[t2];
                    for (int i = 0; i < hd; i++) {
                        d_att_s[t2] += d_ao_h[i] * v_h[i];
                        d_v[t2 * kvd + kv_off + i] += aw * d_ao_h[i];
                    }
                }

                /* softmax backward */
                memset(d_pre_sm, 0, T * sizeof(float));
                softmax_backward(d_pre_sm, d_att_s, att_h, t + 1);

                /* score backward: score = q·k * scale */
                float *q_h = a->q + (l * T + t) * dim + h * hd;
                for (int t2 = 0; t2 <= t; t2++) {
                    float *k_h = a->k + (l * T + t2) * kvd + kv_off;
                    float ds   = d_pre_sm[t2] * scale;
                    for (int i = 0; i < hd; i++) {
                        d_q[t * dim + h * hd + i] += ds * k_h[i];
                        d_k[t2 * kvd + kv_off + i] += ds * q_h[i];
                    }
                }
            }
        }
        pt_trace_end(trace);

        TLOG(":attn");
        /* RoPE backward + QKV backward + attention rmsnorm backward */
        pt_trace_begin(trace, "qkv", "bwd", l);
        float *d_qpre_all = b->d_qpre_all;
        float *d_kpre_all = b->d_kpre_all;
        for (int t = 0; t < T - 1; t++) {
            memset(d_qpre, 0, dim * sizeof(float));
            memset(d_kpre, 0, kvd * sizeof(float));
            rope_backward(d_qpre, d_kpre,
                          d_q + t * dim, d_k + t * kvd, dim, hd, t);

            if (gpu) {
                backward_input_pretrans(d_xb, wq_t, d_qpre, dim, dim, d_temp, matvec, 0);
                backward_input_pretrans(d_xb, wk_t, d_kpre, kvd, dim, d_temp, matvec, 1);
                backward_input_pretrans(d_xb, wv_t, d_v + t * kvd, kvd, dim, d_temp, matvec, 1);
            } else {
                memset(d_xb, 0, dim * sizeof(float));
                matmul_backward_input(d_xb, w->wq + l * dim * dim, d_qpre, dim, dim);
                matmul_backward_input(d_xb, w->wk + l * kvd * dim, d_kpre, kvd, dim);
                matmul_backward_input(d_xb, w->wv + l * kvd * dim, d_v + t * kvd, kvd, dim);
            }

            /* Save pre-RoPE grads for batched weight update */
            memcpy(d_qpre_all + t * dim, d_qpre, dim * sizeof(float));
            memcpy(d_kpre_all + t * kvd, d_kpre, kvd * sizeof(float));

            /* attention rmsnorm backward → d_res becomes d_res_in */
            rmsnorm_backward(d_res + t * dim, g->rms_att_weight + l * dim,
                             d_xb, a->residuals + (l * T + t) * dim,
                             w->rms_att_weight + l * dim, dim);
        }
        /* Batched QKV weight gradients */
        matmul_backward_weight_batched(g->wq + l * dim * dim,
            d_qpre_all, a->xb_att + l * T * dim, dim, dim, T - 1);
        matmul_backward_weight_batched(g->wk + l * kvd * dim,
            d_kpre_all, a->xb_att + l * T * dim, kvd, dim, T - 1);
        matmul_backward_weight_batched(g->wv + l * kvd * dim,
            d_v, a->xb_att + l * T * dim, kvd, dim, T - 1);
        pt_trace_end(trace);
        pt_trace_end(trace); /* end layer */
    }

    TLOG(":rope+qkv]");

    /* ── embedding backward ── */
    TLOG("[bwd:emb]");
    pt_trace_begin(trace, "embedding", "bwd", -1);
    for (int t = 0; t < T - 1; t++)
        embedding_backward(g->token_embedding, d_res + t * dim, dim, tokens[t]);
    pt_trace_end(trace);
    TLOG("\n");
}

/* ── SGD update ───────────────────────────────────────────── */

static void sgd_step(float *w, const float *g, int n, float lr) {
    for (int i = 0; i < n; i++)
        w[i] -= lr * g[i];
}

void pt_sgd_update(pt_weights_t *w, const pt_grads_t *g,
                   float lr, const pt_config_t *cfg) {
    int hd  = cfg->dim / cfg->n_heads;
    int kvd = cfg->n_kv_heads * hd;

    sgd_step(w->token_embedding, g->token_embedding,
             cfg->vocab_size * cfg->dim, lr);
    sgd_step(w->rms_att_weight, g->rms_att_weight,
             cfg->n_layers * cfg->dim, lr);
    sgd_step(w->wq, g->wq, cfg->n_layers * cfg->dim * cfg->dim, lr);
    sgd_step(w->wk, g->wk, cfg->n_layers * kvd * cfg->dim, lr);
    sgd_step(w->wv, g->wv, cfg->n_layers * kvd * cfg->dim, lr);
    sgd_step(w->wo, g->wo, cfg->n_layers * cfg->dim * cfg->dim, lr);
    sgd_step(w->rms_ffn_weight, g->rms_ffn_weight,
             cfg->n_layers * cfg->dim, lr);
    sgd_step(w->w1, g->w1, cfg->n_layers * cfg->hidden_dim * cfg->dim, lr);
    sgd_step(w->w2, g->w2, cfg->n_layers * cfg->dim * cfg->hidden_dim, lr);
    sgd_step(w->w3, g->w3, cfg->n_layers * cfg->hidden_dim * cfg->dim, lr);
    sgd_step(w->rms_final_weight, g->rms_final_weight, cfg->dim, lr);
    if (w->wcls != w->token_embedding)
        sgd_step(w->wcls, g->wcls, cfg->vocab_size * cfg->dim, lr);
}

/*
 * SGD update for layers [l_start, l_end) only.
 * Does NOT update embedding, final rmsnorm, or classifier.
 */
void pt_sgd_update_layers(pt_weights_t *w, const pt_grads_t *g,
                          float lr, const pt_config_t *cfg,
                          int l_start, int l_end) {
    int dim  = cfg->dim;
    int hdim = cfg->hidden_dim;
    int hd   = dim / cfg->n_heads;
    int kvd  = cfg->n_kv_heads * hd;
    int n    = l_end - l_start;

    sgd_step(w->rms_att_weight + l_start * dim,
             g->rms_att_weight + l_start * dim, n * dim, lr);
    sgd_step(w->wq + l_start * dim * dim,
             g->wq + l_start * dim * dim, n * dim * dim, lr);
    sgd_step(w->wk + l_start * kvd * dim,
             g->wk + l_start * kvd * dim, n * kvd * dim, lr);
    sgd_step(w->wv + l_start * kvd * dim,
             g->wv + l_start * kvd * dim, n * kvd * dim, lr);
    sgd_step(w->wo + l_start * dim * dim,
             g->wo + l_start * dim * dim, n * dim * dim, lr);
    sgd_step(w->rms_ffn_weight + l_start * dim,
             g->rms_ffn_weight + l_start * dim, n * dim, lr);
    sgd_step(w->w1 + l_start * hdim * dim,
             g->w1 + l_start * hdim * dim, n * hdim * dim, lr);
    sgd_step(w->w2 + l_start * dim * hdim,
             g->w2 + l_start * dim * hdim, n * dim * hdim, lr);
    sgd_step(w->w3 + l_start * hdim * dim,
             g->w3 + l_start * hdim * dim, n * hdim * dim, lr);
}

/*
 * SGD update for embedding + final rmsnorm + classifier.
 * Complements pt_sgd_update_layers.
 */
void pt_sgd_update_head(pt_weights_t *w, const pt_grads_t *g,
                        float lr, const pt_config_t *cfg) {
    sgd_step(w->token_embedding, g->token_embedding,
             cfg->vocab_size * cfg->dim, lr);
    sgd_step(w->rms_final_weight, g->rms_final_weight, cfg->dim, lr);
    if (w->wcls != w->token_embedding)
        sgd_step(w->wcls, g->wcls, cfg->vocab_size * cfg->dim, lr);
}

/* ── Staged forward training (for pipeline-parallel) ─────────── */

void pt_forward_train_embed(const pt_weights_t *w, pt_activations_t *a,
                            const int *tokens, int T, int dim) {
    a->T = T;
    for (int t = 0; t < T; t++)
        embedding_lookup(a->residuals + t * dim,
                         w->token_embedding, dim, tokens[t]);
}

void pt_forward_train_layers_range(const pt_config_t *cfg, const pt_weights_t *w,
                                    pt_activations_t *a, int T,
                                    int l_start, int l_end,
                                    pt_matvec_fn matvec, pt_trace_t *trace) {
    int dim      = cfg->dim;
    int hdim     = cfg->hidden_dim;
    int nh       = cfg->n_heads;
    int hd       = dim / nh;
    int kvd      = cfg->n_kv_heads * hd;
    int kv_mul   = nh / cfg->n_kv_heads;
    float scale  = 1.0f / pt_sqrtf((float)hd);

    for (int l = l_start; l < l_end; l++) {
        TLOG("[fwd:L%d", l);
        pt_trace_begin(trace, "layer", "fwd", l);
        float *res_in  = a->residuals     + l * T * dim;
        float *res_mid = a->residuals_att + l * T * dim;
        float *res_out = a->residuals     + (l + 1) * T * dim;

        /* QKV projections + RoPE */
        pt_trace_begin(trace, "qkv", "fwd", l);
        for (int t = 0; t < T; t++) {
            float *xb = a->xb_att + (l * T + t) * dim;
            rmsnorm(xb, res_in + t * dim,
                    w->rms_att_weight + l * dim, dim);

            float *qt = a->q + (l * T + t) * dim;
            float *kt = a->k + (l * T + t) * kvd;
            float *vt = a->v + (l * T + t) * kvd;

            matvec(w->wq + l * dim * dim,   xb, qt, dim, dim);
            matvec(w->wk + l * kvd * dim,   xb, kt, kvd, dim);
            matvec(w->wv + l * kvd * dim,   xb, vt, kvd, dim);

            rope(qt, kt, dim, hd, t);
        }
        pt_trace_end(trace);

        TLOG(":qkv");
        /* Multi-head causal attention + output projection */
        pt_trace_begin(trace, "attention", "fwd", l);
        for (int t = 0; t < T; t++) {
            float *qt   = a->q       + (l * T + t) * dim;
            float *ao_t = a->att_out + (l * T + t) * dim;

            for (int h = 0; h < nh; h++) {
                float *q_h   = qt + h * hd;
                float *att_h = a->att + ((l * nh + h) * T + t) * T;

                for (int t2 = 0; t2 <= t; t2++) {
                    float *k_t2 = a->k + (l * T + t2) * kvd
                                  + (h / kv_mul) * hd;
                    float score = 0.0f;
                    for (int i = 0; i < hd; i++)
                        score += q_h[i] * k_t2[i];
                    att_h[t2] = score * scale;
                }
                for (int t2 = t + 1; t2 < T; t2++)
                    att_h[t2] = 0.0f;

                softmax(att_h, t + 1);

                float *out_h = ao_t + h * hd;
                memset(out_h, 0, hd * sizeof(float));
                for (int t2 = 0; t2 <= t; t2++) {
                    float *v_t2 = a->v + (l * T + t2) * kvd
                                  + (h / kv_mul) * hd;
                    float a_val = att_h[t2];
                    for (int i = 0; i < hd; i++)
                        out_h[i] += a_val * v_t2[i];
                }
            }

            /* Wo projection + attention residual */
            matvec(w->wo + l * dim * dim, ao_t, a->scratch, dim, dim);
            vec_add(res_mid + t * dim, res_in + t * dim, a->scratch, dim);
        }
        pt_trace_end(trace);

        TLOG(":att");
        /* Feed-forward (SwiGLU) */
        pt_trace_begin(trace, "ffn", "fwd", l);
        for (int t = 0; t < T; t++) {
            float *xb  = a->xb_ffn + (l * T + t) * dim;
            rmsnorm(xb, res_mid + t * dim,
                    w->rms_ffn_weight + l * dim, dim);

            float *hbt  = a->hb  + (l * T + t) * hdim;
            float *hb2t = a->hb2 + (l * T + t) * hdim;

            matvec(w->w1 + l * hdim * dim, xb, hbt,  hdim, dim);
            matvec(w->w3 + l * hdim * dim, xb, hb2t, hdim, dim);

            float *sh = a->scratch;
            for (int i = 0; i < hdim; i++) {
                float sig = 1.0f / (1.0f + pt_expf(-hbt[i]));
                sh[i] = (hbt[i] * sig) * hb2t[i];
            }

            matvec(w->w2 + l * dim * hdim, sh, res_out + t * dim, dim, hdim);
            vec_add(res_out + t * dim, res_out + t * dim,
                    res_mid + t * dim, dim);
        }
        pt_trace_end(trace);
        pt_trace_end(trace); /* end layer */
        TLOG(":ffn]");
    }
}

float pt_forward_train_head(const pt_config_t *cfg, const pt_weights_t *w,
                            pt_activations_t *a, const int *tokens, int T,
                            pt_matvec_fn matvec, pt_trace_t *trace) {
    int dim = cfg->dim;
    int L   = cfg->n_layers;
    int V   = cfg->vocab_size;

    /* Final rmsnorm + classifier */
    pt_trace_begin(trace, "classifier", "fwd", -1);
    float *res_final = a->residuals + L * T * dim;
    for (int t = 0; t < T; t++) {
        rmsnorm(a->x_final + t * dim, res_final + t * dim,
                w->rms_final_weight, dim);
        matvec(w->wcls, a->x_final + t * dim,
               a->logits + t * V, V, dim);
    }
    pt_trace_end(trace);

    /* Cross-entropy loss */
    pt_trace_begin(trace, "loss", "fwd", -1);
    float loss = 0.0f;
    for (int t = 0; t < T - 1; t++) {
        float *lg = a->logits + t * V;
        int target = tokens[t + 1];
        float max_lg = lg[0];
        for (int v = 1; v < V; v++)
            if (lg[v] > max_lg) max_lg = lg[v];
        float sum = 0.0f;
        for (int v = 0; v < V; v++)
            sum += pt_expf(lg[v] - max_lg);
        loss += -(lg[target] - max_lg - pt_logf(sum));
    }
    loss /= (float)(T - 1);
    pt_trace_end(trace);

    return loss;
}

/* ── Staged backward (for pipeline-parallel) ─────────────────── */

void pt_backward_head(const pt_config_t *cfg, const pt_weights_t *w,
                      pt_grads_t *g, const pt_activations_t *a,
                      pt_backward_buf_t *b,
                      const int *tokens, int T,
                      pt_matvec_fn matvec, pt_trace_t *trace) {
    int dim = cfg->dim;
    int L   = cfg->n_layers;
    int V   = cfg->vocab_size;
    float *d_res   = b->d_res;
    float *d_xb    = b->d_xb;
    float *w_t     = b->w_transpose;
    float *d_temp  = b->d_temp;
    int gpu = (matvec && w_t && d_temp);

    memset(d_res, 0, (size_t)T * dim * sizeof(float));

    /* Loss + classifier + final rmsnorm backward */
    pt_trace_begin(trace, "classifier", "bwd", -1);
    float inv_T1 = 1.0f / (float)(T - 1);

    if (gpu) {
        pt_trace_begin(trace, "cls_transpose", "bwd", -1);
        transpose(w_t, w->wcls, V, dim);
        pt_trace_end(trace);
    }

    float *d_logit_all = b->d_logit_all;

    /* Phase 1: loss backward */
    for (int t = 0; t < T - 1; t++) {
        float *dl = d_logit_all + t * V;
        float *lg = a->logits + t * V;
        int target = tokens[t + 1];
        float mx = lg[0];
        for (int v = 1; v < V; v++) if (lg[v] > mx) mx = lg[v];
        float sm = 0.0f;
        for (int v = 0; v < V; v++) { dl[v] = pt_expf(lg[v] - mx); sm += dl[v]; }
        float inv = 1.0f / sm;
        for (int v = 0; v < V; v++) dl[v] *= inv * inv_T1;
        dl[target] -= inv_T1;
    }

    /* Phase 2: input gradient + rmsnorm backward (uses w_t for transposed wcls) */
    for (int t = 0; t < T - 1; t++) {
        float *dl = d_logit_all + t * V;
        if (gpu)
            backward_input_pretrans(d_xb, w_t, dl, V, dim, d_temp, matvec, 0);
        else {
            memset(d_xb, 0, dim * sizeof(float));
            matmul_backward_input(d_xb, w->wcls, dl, V, dim);
        }
        rmsnorm_backward(d_res + t * dim, g->rms_final_weight,
                         d_xb, a->residuals + L * T * dim + t * dim,
                         w->rms_final_weight, dim);
    }

    /* Phase 3: classifier weight gradient (w_t now free, reused as GEMM scratch) */
    pt_trace_begin(trace, "cls_weight_grad", "bwd", -1);
#if defined(__RPI__) && defined(GEMM_BWD_ENABLED)
    if (gpu)
        matmul_backward_weight_gpu(g->wcls, d_logit_all, a->x_final,
                                    V, dim, T - 1, w_t);
    else
#endif
    matmul_backward_weight_batched(g->wcls, d_logit_all, a->x_final, V, dim, T - 1);
    pt_trace_end(trace);

    pt_trace_end(trace);
}

void pt_backward_layers_range(const pt_config_t *cfg, const pt_weights_t *w,
                               pt_grads_t *g, const pt_activations_t *a,
                               pt_backward_buf_t *b, int T,
                               int l_start, int l_end,
                               pt_matvec_fn matvec, pt_trace_t *trace) {
    int dim    = cfg->dim;
    int hdim   = cfg->hidden_dim;
    int nh     = cfg->n_heads;
    int hd     = dim / nh;
    int kvd    = cfg->n_kv_heads * hd;
    int kv_mul = nh / cfg->n_kv_heads;
    float scale = 1.0f / pt_sqrtf((float)hd);

    float *d_res    = b->d_res;
    float *d_aout   = b->d_aout;
    float *d_q      = b->d_q;
    float *d_k      = b->d_k;
    float *d_v      = b->d_v;
    float *d_xb     = b->d_xb;
    float *d_qpre   = b->d_qpre;
    float *d_kpre   = b->d_kpre;
    float *d_hid    = b->d_hid;
    float *d_hid2   = b->d_hid2;
    float *d_att_s  = b->d_att_s;
    float *d_pre_sm = b->d_pre_sm;
    float *w_t      = b->w_transpose;
    float *d_temp   = b->d_temp;
    int gpu = (matvec && w_t && d_temp);

    for (int l = l_end - 1; l >= l_start; l--) {
        TLOG("[bwd:L%d", l);
        pt_trace_begin(trace, "layer", "bwd", l);

        /* Pre-transpose layer weight matrices */
        float *wo_t = NULL, *w2_t = NULL, *w1_t = NULL, *w3_t = NULL;
        float *wq_t = NULL, *wk_t = NULL, *wv_t = NULL;
        if (gpu) {
            pt_trace_begin(trace, "transpose", "bwd", l);
            float *p = w_t;
            wo_t = p; transpose(wo_t, w->wo + l * dim * dim,   dim,  dim);  p += dim * dim;
            w2_t = p; transpose(w2_t, w->w2 + l * dim * hdim,  dim,  hdim); p += dim * hdim;
            w1_t = p; transpose(w1_t, w->w1 + l * hdim * dim,  hdim, dim);  p += hdim * dim;
            w3_t = p; transpose(w3_t, w->w3 + l * hdim * dim,  hdim, dim);  p += hdim * dim;
            wq_t = p; transpose(wq_t, w->wq + l * dim * dim,   dim,  dim);  p += dim * dim;
            wk_t = p; transpose(wk_t, w->wk + l * kvd * dim,   kvd,  dim);  p += kvd * dim;
            wv_t = p; transpose(wv_t, w->wv + l * kvd * dim,   kvd,  dim);
            pt_trace_end(trace);
        }

        /* FFN backward — Phase 1: per-position input grads + collect batch vectors */
        pt_trace_begin(trace, "ffn", "bwd", l);
        float *sh_all2      = b->sh_all;
        float *d_hid_all2   = b->d_hid_all;
        float *d_hid2_all2  = b->d_hid2_all;
        float *drt_ffn_all2 = b->drt_ffn_all;
        for (int t = 0; t < T - 1; t++) {
            float *drt  = d_res + t * dim;
            float *hbt  = a->hb  + (l * T + t) * hdim;
            float *hb2t = a->hb2 + (l * T + t) * hdim;

            /* recompute silu(hb) * hb2 directly into batch buffer */
            float *sh_t2 = sh_all2 + t * hdim;
            for (int i = 0; i < hdim; i++) {
                float sig = 1.0f / (1.0f + pt_expf(-hbt[i]));
                sh_t2[i] = (hbt[i] * sig) * hb2t[i];
            }

            memcpy(drt_ffn_all2 + t * dim, drt, dim * sizeof(float));

            if (gpu)
                backward_input_pretrans(d_hid, w2_t, drt, dim, hdim, d_temp, matvec, 0);
            else {
                memset(d_hid, 0, hdim * sizeof(float));
                matmul_backward_input(d_hid, w->w2 + l * dim * hdim, drt, dim, hdim);
            }

            for (int i = 0; i < hdim; i++) {
                float sig = 1.0f / (1.0f + pt_expf(-hbt[i]));
                float silu_val = hbt[i] * sig;
                float d_silu = d_hid[i] * hb2t[i];
                d_hid2[i] = d_hid[i] * silu_val;
                d_hid[i] = d_silu * sig * (1.0f + hbt[i] * (1.0f - sig));
            }

            if (gpu) {
                backward_input_pretrans(d_xb, w1_t, d_hid,  hdim, dim, d_temp, matvec, 0);
                backward_input_pretrans(d_xb, w3_t, d_hid2, hdim, dim, d_temp, matvec, 1);
            } else {
                memset(d_xb, 0, dim * sizeof(float));
                matmul_backward_input(d_xb, w->w1 + l * hdim * dim, d_hid,  hdim, dim);
                matmul_backward_input(d_xb, w->w3 + l * hdim * dim, d_hid2, hdim, dim);
            }
            memcpy(d_hid_all2 + t * hdim, d_hid, hdim * sizeof(float));
            memcpy(d_hid2_all2 + t * hdim, d_hid2, hdim * sizeof(float));

            rmsnorm_backward(drt, g->rms_ffn_weight + l * dim,
                             d_xb, a->residuals_att + (l * T + t) * dim,
                             w->rms_ffn_weight + l * dim, dim);
        }
        /* Phase 2: batched weight gradients for w2, w1, w3 */
        matmul_backward_weight_batched(g->w2 + l * dim * hdim,
            drt_ffn_all2, sh_all2, dim, hdim, T - 1);
        matmul_backward_weight_batched(g->w1 + l * hdim * dim,
            d_hid_all2, a->xb_ffn + l * T * dim, hdim, dim, T - 1);
        matmul_backward_weight_batched(g->w3 + l * hdim * dim,
            d_hid2_all2, a->xb_ffn + l * T * dim, hdim, dim, T - 1);
        pt_trace_end(trace);

        TLOG(":ffn:");
        /* Attention backward — wo */
        pt_trace_begin(trace, "wo", "bwd", l);
        memset(d_aout, 0, (size_t)T * dim * sizeof(float));
        float *drt_wo_all2 = b->drt_wo_all;
        for (int t = 0; t < T - 1; t++) {
            float *drt = d_res + t * dim;
            if (gpu)
                backward_input_pretrans(d_aout + t * dim, wo_t, drt, dim, dim, d_temp, matvec, 0);
            else
                matmul_backward_input(d_aout + t * dim, w->wo + l * dim * dim, drt, dim, dim);
            memcpy(drt_wo_all2 + t * dim, drt, dim * sizeof(float));
        }
        matmul_backward_weight_batched(g->wo + l * dim * dim,
            drt_wo_all2, a->att_out + l * T * dim, dim, dim, T - 1);
        pt_trace_end(trace);

        TLOG(":wo");
        /* Attention mechanism backward */
        pt_trace_begin(trace, "attention", "bwd", l);
        memset(d_q, 0, (size_t)T * dim * sizeof(float));
        memset(d_k, 0, (size_t)T * kvd * sizeof(float));
        memset(d_v, 0, (size_t)T * kvd * sizeof(float));

        for (int t = 0; t < T - 1; t++) {
            for (int h = 0; h < nh; h++) {
                float *d_ao_h = d_aout + t * dim + h * hd;
                float *att_h  = a->att + ((l * nh + h) * T + t) * T;
                int kv_off    = (h / kv_mul) * hd;

                memset(d_att_s, 0, T * sizeof(float));
                for (int t2 = 0; t2 <= t; t2++) {
                    float *v_h = a->v + (l * T + t2) * kvd + kv_off;
                    float aw   = att_h[t2];
                    for (int i = 0; i < hd; i++) {
                        d_att_s[t2] += d_ao_h[i] * v_h[i];
                        d_v[t2 * kvd + kv_off + i] += aw * d_ao_h[i];
                    }
                }

                memset(d_pre_sm, 0, T * sizeof(float));
                softmax_backward(d_pre_sm, d_att_s, att_h, t + 1);

                float *q_h = a->q + (l * T + t) * dim + h * hd;
                for (int t2 = 0; t2 <= t; t2++) {
                    float *k_h = a->k + (l * T + t2) * kvd + kv_off;
                    float ds   = d_pre_sm[t2] * scale;
                    for (int i = 0; i < hd; i++) {
                        d_q[t * dim + h * hd + i] += ds * k_h[i];
                        d_k[t2 * kvd + kv_off + i] += ds * q_h[i];
                    }
                }
            }
        }
        pt_trace_end(trace);

        TLOG(":attn");
        /* RoPE backward + QKV backward + attention rmsnorm backward */
        pt_trace_begin(trace, "qkv", "bwd", l);
        float *d_qpre_all2 = b->d_qpre_all;
        float *d_kpre_all2 = b->d_kpre_all;
        for (int t = 0; t < T - 1; t++) {
            memset(d_qpre, 0, dim * sizeof(float));
            memset(d_kpre, 0, kvd * sizeof(float));
            rope_backward(d_qpre, d_kpre,
                          d_q + t * dim, d_k + t * kvd, dim, hd, t);

            if (gpu) {
                backward_input_pretrans(d_xb, wq_t, d_qpre, dim, dim, d_temp, matvec, 0);
                backward_input_pretrans(d_xb, wk_t, d_kpre, kvd, dim, d_temp, matvec, 1);
                backward_input_pretrans(d_xb, wv_t, d_v + t * kvd, kvd, dim, d_temp, matvec, 1);
            } else {
                memset(d_xb, 0, dim * sizeof(float));
                matmul_backward_input(d_xb, w->wq + l * dim * dim, d_qpre, dim, dim);
                matmul_backward_input(d_xb, w->wk + l * kvd * dim, d_kpre, kvd, dim);
                matmul_backward_input(d_xb, w->wv + l * kvd * dim, d_v + t * kvd, kvd, dim);
            }

            memcpy(d_qpre_all2 + t * dim, d_qpre, dim * sizeof(float));
            memcpy(d_kpre_all2 + t * kvd, d_kpre, kvd * sizeof(float));

            rmsnorm_backward(d_res + t * dim, g->rms_att_weight + l * dim,
                             d_xb, a->residuals + (l * T + t) * dim,
                             w->rms_att_weight + l * dim, dim);
        }
        /* Batched QKV weight gradients */
        matmul_backward_weight_batched(g->wq + l * dim * dim,
            d_qpre_all2, a->xb_att + l * T * dim, dim, dim, T - 1);
        matmul_backward_weight_batched(g->wk + l * kvd * dim,
            d_kpre_all2, a->xb_att + l * T * dim, kvd, dim, T - 1);
        matmul_backward_weight_batched(g->wv + l * kvd * dim,
            d_v, a->xb_att + l * T * dim, kvd, dim, T - 1);
        pt_trace_end(trace);
        pt_trace_end(trace); /* end layer */
    }
    TLOG(":rope+qkv]");
}

void pt_backward_embed(pt_grads_t *g, const pt_backward_buf_t *b,
                       const int *tokens, int T, int dim) {
    for (int t = 0; t < T - 1; t++)
        embedding_backward(g->token_embedding, b->d_res + t * dim, dim, tokens[t]);
}
