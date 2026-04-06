#include <string.h>
#ifndef __RPI__
#include <stdlib.h>
#endif
#include "llama2.h"
#include "pt_ops.h"
#include "pt_math.h"

void pt_load_config(pt_config_t *cfg, const void *data) {
    const int *p = (const int *)data;
    cfg->dim        = p[0];
    cfg->hidden_dim = p[1];
    cfg->n_layers   = p[2];
    cfg->n_heads    = p[3];
    cfg->n_kv_heads = p[4];
    int raw_vocab   = p[5];
    cfg->vocab_size = raw_vocab > 0 ? raw_vocab : -raw_vocab;
    cfg->seq_len    = p[6];
}

unsigned pt_file_size(const void *data) {
    const int *h = (const int *)data;
    int dim = h[0], hidden_dim = h[1], n_layers = h[2], n_heads = h[3];
    int n_kv_heads = h[4], raw_vocab = h[5], seq_len = h[6];
    int vocab_size = raw_vocab > 0 ? raw_vocab : -raw_vocab;
    int shared = raw_vocab > 0;
    int head_dim = dim / n_heads;
    int kv_dim = n_kv_heads * head_dim;

    unsigned s = 7 * 4;                                     /* header */
    s += (unsigned)(vocab_size * dim) * 4;                  /* token_embedding */
    s += (unsigned)(n_layers * dim) * 4;                    /* rms_att_weight */
    s += (unsigned)(n_layers * dim * dim) * 4;              /* wq */
    s += (unsigned)(n_layers * kv_dim * dim) * 4;           /* wk */
    s += (unsigned)(n_layers * kv_dim * dim) * 4;           /* wv */
    s += (unsigned)(n_layers * dim * dim) * 4;              /* wo */
    s += (unsigned)(n_layers * dim) * 4;                    /* rms_ffn_weight */
    s += (unsigned)(n_layers * hidden_dim * dim) * 4;       /* w1 */
    s += (unsigned)(n_layers * dim * hidden_dim) * 4;       /* w2 */
    s += (unsigned)(n_layers * hidden_dim * dim) * 4;       /* w3 */
    s += (unsigned)dim * 4;                                 /* rms_final_weight */
    s += (unsigned)(seq_len * head_dim / 2) * 4;            /* freq_cis_real */
    s += (unsigned)(seq_len * head_dim / 2) * 4;            /* freq_cis_imag */
    if (!shared)
        s += (unsigned)(vocab_size * dim) * 4;              /* wcls */
    return s;
}

void pt_load_weights(pt_weights_t *w, const pt_config_t *cfg, void *data) {
    int shared_weights = ((const int *)data)[5] > 0;
    float *p = (float *)((char *)data + 7 * sizeof(int));

    int head_dim = cfg->dim / cfg->n_heads;
    int kv_dim   = cfg->n_kv_heads * head_dim;

    w->token_embedding = p; p += cfg->vocab_size * cfg->dim;
    w->rms_att_weight  = p; p += cfg->n_layers * cfg->dim;
    w->wq = p; p += cfg->n_layers * cfg->dim * cfg->dim;
    w->wk = p; p += cfg->n_layers * kv_dim * cfg->dim;
    w->wv = p; p += cfg->n_layers * kv_dim * cfg->dim;
    w->wo = p; p += cfg->n_layers * cfg->dim * cfg->dim;
    w->rms_ffn_weight = p; p += cfg->n_layers * cfg->dim;
    w->w1 = p; p += cfg->n_layers * cfg->hidden_dim * cfg->dim;
    w->w2 = p; p += cfg->n_layers * cfg->dim * cfg->hidden_dim;
    w->w3 = p; p += cfg->n_layers * cfg->hidden_dim * cfg->dim;
    w->rms_final_weight = p; p += cfg->dim;
    p += cfg->seq_len * head_dim / 2;  /* skip freq_cis_real */
    p += cfg->seq_len * head_dim / 2;  /* skip freq_cis_imag */
    w->wcls = shared_weights ? w->token_embedding : p;
}

#ifndef __RPI__
void pt_alloc_state(pt_state_t *s, const pt_config_t *cfg) {
    int head_dim = cfg->dim / cfg->n_heads;
    int kv_dim   = cfg->n_kv_heads * head_dim;

    s->x      = (float *)malloc(cfg->dim * sizeof(float));
    s->xb     = (float *)malloc(cfg->dim * sizeof(float));
    s->xb2    = (float *)malloc(cfg->dim * sizeof(float));
    s->q      = (float *)malloc(cfg->dim * sizeof(float));
    s->k      = (float *)malloc(kv_dim * sizeof(float));
    s->v      = (float *)malloc(kv_dim * sizeof(float));
    s->hb     = (float *)malloc(cfg->hidden_dim * sizeof(float));
    s->hb2    = (float *)malloc(cfg->hidden_dim * sizeof(float));
    s->att    = (float *)malloc(cfg->n_heads * cfg->seq_len * sizeof(float));
    s->logits = (float *)malloc(cfg->vocab_size * sizeof(float));
    s->key_cache   = (float *)calloc((size_t)cfg->n_layers * cfg->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *)calloc((size_t)cfg->n_layers * cfg->seq_len * kv_dim, sizeof(float));
}

void pt_free_state(pt_state_t *s) {
    free(s->x);     free(s->xb);    free(s->xb2);
    free(s->q);     free(s->k);     free(s->v);
    free(s->hb);    free(s->hb2);
    free(s->att);   free(s->logits);
    free(s->key_cache);  free(s->value_cache);
}
#else
void pt_scratch_alloc_state(pt_state_t *s, const pt_config_t *cfg,
                            pt_bump_alloc_fn alloc) {
    int hd  = cfg->dim / cfg->n_heads;
    int kvd = cfg->n_kv_heads * hd;
    s->x      = alloc(cfg->dim * sizeof(float));
    s->xb     = alloc(cfg->dim * sizeof(float));
    s->xb2    = alloc(cfg->dim * sizeof(float));
    s->q      = alloc(cfg->dim * sizeof(float));
    s->k      = alloc(kvd * sizeof(float));
    s->v      = alloc(kvd * sizeof(float));
    s->hb     = alloc(cfg->hidden_dim * sizeof(float));
    s->hb2    = alloc(cfg->hidden_dim * sizeof(float));
    s->att    = alloc(cfg->n_heads * cfg->seq_len * sizeof(float));
    s->logits = alloc(cfg->vocab_size * sizeof(float));
    unsigned kv_bytes = cfg->n_layers * cfg->seq_len * kvd * sizeof(float);
    s->key_cache   = alloc(kv_bytes);
    s->value_cache = alloc(kv_bytes);
    memset(s->key_cache,   0, kv_bytes);
    memset(s->value_cache, 0, kv_bytes);
}
#endif

/* ── Staged forward pass ── */

void pt_forward_embed(const pt_weights_t *w, float *x, int dim, int token) {
    embedding_lookup(x, w->token_embedding, dim, token);
}

void pt_forward_layers_range(const pt_config_t *cfg, const pt_weights_t *w,
                              pt_state_t *s, int pos,
                              int l_start, int l_end,
                              pt_matvec_fn matvec) {
    int dim        = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    int n_heads    = cfg->n_heads;
    int head_dim   = dim / n_heads;
    int kv_dim     = cfg->n_kv_heads * head_dim;
    int kv_mul     = n_heads / cfg->n_kv_heads;
    float att_scale = 1.0f / pt_sqrtf((float)head_dim);

    for (int l = l_start; l < l_end; l++) {
        /* ── attention ── */
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);

        matvec(w->wq + l * dim * dim,    s->xb, s->q, dim,    dim);
        matvec(w->wk + l * kv_dim * dim, s->xb, s->k, kv_dim, dim);
        matvec(w->wv + l * kv_dim * dim, s->xb, s->v, kv_dim, dim);

        rope(s->q, s->k, dim, head_dim, pos);

        int loff = l * cfg->seq_len * kv_dim;
        memcpy(s->key_cache   + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        for (int h = 0; h < n_heads; h++) {
            float *q_h   = s->q   + h * head_dim;
            float *att_h = s->att + h * cfg->seq_len;

            for (int t = 0; t <= pos; t++) {
                float *k_t = s->key_cache + loff + t * kv_dim
                             + (h / kv_mul) * head_dim;
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++)
                    score += q_h[i] * k_t[i];
                att_h[t] = score * att_scale;
            }

            softmax(att_h, pos + 1);

            float *xb_h = s->xb + h * head_dim;
            memset(xb_h, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v_t = s->value_cache + loff + t * kv_dim
                             + (h / kv_mul) * head_dim;
                float a = att_h[t];
                for (int i = 0; i < head_dim; i++)
                    xb_h[i] += a * v_t[i];
            }
        }

        matvec(w->wo + l * dim * dim, s->xb, s->xb2, dim, dim);
        vec_add(s->x, s->x, s->xb2, dim);

        /* ── feed-forward (SwiGLU) ── */
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

        matvec(w->w1 + l * hidden_dim * dim, s->xb, s->hb,  hidden_dim, dim);
        matvec(w->w3 + l * hidden_dim * dim, s->xb, s->hb2, hidden_dim, dim);
        silu(s->hb, hidden_dim);
        vec_mul(s->hb, s->hb, s->hb2, hidden_dim);
        matvec(w->w2 + l * dim * hidden_dim, s->hb, s->xb, dim, hidden_dim);
        vec_add(s->x, s->x, s->xb, dim);
    }
}

void pt_forward_head(const pt_config_t *cfg, const pt_weights_t *w,
                     pt_state_t *s, pt_matvec_fn matvec) {
    rmsnorm(s->x, s->x, w->rms_final_weight, cfg->dim);
    matvec(w->wcls, s->x, s->logits, cfg->vocab_size, cfg->dim);
}

/* Monolithic forward pass — thin wrapper over staged functions. */
void pt_forward(const pt_config_t *cfg, const pt_weights_t *w,
                pt_state_t *s, int token, int pos, pt_matvec_fn matvec) {
    pt_forward_embed(w, s->x, cfg->dim, token);
    pt_forward_layers_range(cfg, w, s, pos, 0, cfg->n_layers, matvec);
    pt_forward_head(cfg, w, s, matvec);
}
