#include "pt_shard.h"
#include "pt_ops.h"
#include <string.h>

#ifdef __RPI__
#include "rpi.h"
#define SLOG(...) printk(__VA_ARGS__)
#else
#include <stdio.h>
#include <stdlib.h>
#define SLOG(...) printf(__VA_ARGS__)
#define panic(...) do { fprintf(stderr, __VA_ARGS__); exit(1); } while(0)
#endif

void pt_load_shard_header(pt_config_t *global_cfg, pt_shard_info_t *shard,
                          const void *data) {
    /* Standard header (28 bytes) */
    pt_load_config(global_cfg, data);

    /* Shard extension (bytes 28-55) */
    const int *ext = (const int *)((const char *)data + 28);
    if ((unsigned)ext[0] != PT_SHARD_MAGIC)
        panic("not a shard file (magic=0x%x, expected 0x%x)\n",
              (unsigned)ext[0], PT_SHARD_MAGIC);

    shard->rank       = ext[1];
    shard->world_size = ext[2];
    shard->l_start    = ext[3];
    shard->l_end      = ext[4];
    shard->has_embed  = ext[5];
    shard->has_head   = ext[6];
    shard->n_local    = shard->l_end - shard->l_start;
    shard->n_layers_global = global_cfg->n_layers;

    SLOG("shard: rank %d/%d layers [%d,%d) embed=%d head=%d\n",
         shard->rank, shard->world_size,
         shard->l_start, shard->l_end,
         shard->has_embed, shard->has_head);
}

void pt_load_shard_weights(pt_weights_t *w, const pt_config_t *global_cfg,
                           const pt_shard_info_t *shard, void *data) {
    int shared = ((const int *)data)[5] > 0;
    int dim = global_cfg->dim;
    int hidden_dim = global_cfg->hidden_dim;
    int head_dim = dim / global_cfg->n_heads;
    int kv_dim = global_cfg->n_kv_heads * head_dim;
    int n_local = shard->n_local;
    int vocab_size = global_cfg->vocab_size;
    int seq_len = global_cfg->seq_len;

    float *p = (float *)((char *)data + 56);  /* skip 56-byte header */
    memset(w, 0, sizeof(*w));

    /* Embed */
    if (shard->has_embed) {
        w->token_embedding = p;  p += vocab_size * dim;
    }

    /* Per-layer weights (local layers only) */
    if (n_local > 0) {
        w->rms_att_weight = p;  p += n_local * dim;
        w->wq = p;             p += n_local * dim * dim;
        w->wk = p;             p += n_local * kv_dim * dim;
        w->wv = p;             p += n_local * kv_dim * dim;
        w->wo = p;             p += n_local * dim * dim;
        w->rms_ffn_weight = p; p += n_local * dim;
        w->w1 = p;             p += n_local * hidden_dim * dim;
        w->w2 = p;             p += n_local * dim * hidden_dim;
        w->w3 = p;             p += n_local * hidden_dim * dim;
    }

    /* Head */
    if (shard->has_head) {
        w->rms_final_weight = p;  p += dim;
    }

    /* freq_cis (always present) */
    /* Skip — they're accessed via global pointers set by pt_forward.
     * The forward functions read freq_cis from the weight file directly.
     * We just need to know where they are. */
    float *freq_real = p;  p += seq_len * head_dim / 2;
    float *freq_imag = p;  p += seq_len * head_dim / 2;
    /* Store in unused weight slots — forward functions need these */
    (void)freq_real;
    (void)freq_imag;

    /* wcls */
    if (shard->has_head) {
        w->wcls = shared ? w->token_embedding : p;
    }
}

unsigned pt_shard_file_size(const void *data) {
    const int *h = (const int *)data;
    int dim = h[0], hidden_dim = h[1], n_heads = h[3];
    int n_kv_heads = h[4], raw_vocab = h[5], seq_len = h[6];
    int vocab_size = raw_vocab > 0 ? raw_vocab : -raw_vocab;
    int shared = raw_vocab > 0;
    int head_dim = dim / n_heads;
    int kv_dim = n_kv_heads * head_dim;

    /* Shard extension */
    const int *ext = (const int *)((const char *)data + 28);
    int l_start = ext[3], l_end = ext[4];
    int has_embed = ext[5], has_head = ext[6];
    int n_local = l_end - l_start;

    unsigned s = 56;  /* header */
    if (has_embed)
        s += (unsigned)(vocab_size * dim) * 4;
    if (n_local > 0) {
        s += (unsigned)(n_local * dim) * 4;              /* rms_att */
        s += (unsigned)(n_local * dim * dim) * 4;        /* wq */
        s += (unsigned)(n_local * kv_dim * dim) * 4;     /* wk */
        s += (unsigned)(n_local * kv_dim * dim) * 4;     /* wv */
        s += (unsigned)(n_local * dim * dim) * 4;        /* wo */
        s += (unsigned)(n_local * dim) * 4;              /* rms_ffn */
        s += (unsigned)(n_local * hidden_dim * dim) * 4; /* w1 */
        s += (unsigned)(n_local * dim * hidden_dim) * 4; /* w2 */
        s += (unsigned)(n_local * hidden_dim * dim) * 4; /* w3 */
    }
    if (has_head)
        s += (unsigned)dim * 4;                          /* rms_final */
    s += (unsigned)(seq_len * head_dim / 2) * 4;         /* freq_cis_real */
    s += (unsigned)(seq_len * head_dim / 2) * 4;         /* freq_cis_imag */
    if (has_head && !shared)
        s += (unsigned)(vocab_size * dim) * 4;           /* wcls */
    return s;
}
