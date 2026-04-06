#include "pt.h"
#include "pt_ops.h"
#include <string.h>
#ifndef __RPI__
#include <stdlib.h>
#include <stdio.h>
#endif

#ifdef __RPI__
#include "rpi.h"
#include "gpu.h"
#include "arena.h"
#include "mailbox.h"
#include "profiler.h"
#include "matvec.h"

/* ── Pi globals for GPU matvec wrapper ──────────────────────── */

static int g_num_qpus;

static void matvec_gpu(const float *W, const float *x, float *y,
                       int out_dim, int in_dim) {
    gpu_arena_reset();
    smatvec_tmu(W, x, y, out_dim, in_dim, g_num_qpus, NULL);
}

/* ── Pi bump allocator (base set dynamically in pt_pi_init) ── */

static unsigned state_base;
static unsigned scratch_off;

static void *scratch_alloc(unsigned bytes) {
    void *p = (void *)(state_base + scratch_off);
    scratch_off += (bytes + 15u) & ~15u;
    return p;
}

void pt_pi_init(pt_context_t *ctx, void *weight_data,
                int num_qpus, int max_T, unsigned arena_bytes) {
    memset(ctx, 0, sizeof(*ctx));

    qpu_enable();
    perf_init();
    gpu_arena_init(arena_bytes);

    g_num_qpus = num_qpus;
    ctx->num_qpus = num_qpus;

    /* validate weight data */
    int dim0 = *(volatile int *)weight_data;
    if (dim0 <= 0 || dim0 > 65536)
        panic("no weights at 0x%x (dim=%d)\n"
              "SD card: initramfs weights/<model>.bin 0x2000000\n",
              (unsigned)weight_data, dim0);

    pt_load_config(&ctx->cfg, weight_data);
    ctx->shared_weights = ((const int *)weight_data)[5] > 0;
    pt_load_weights(&ctx->w, &ctx->cfg, weight_data);
    ctx->matvec = matvec_gpu;
    ctx->max_T  = max_T;

    /* place buffers after the weights (1MB aligned) */
    unsigned weight_base = (unsigned)weight_data;
    unsigned file_size = pt_file_size(weight_data);
    state_base = (weight_base + file_size + 0x100000) & ~0xFFFFF;
    scratch_off = 0;

    if (max_T > 0) {
        /* ── training mode ── */
        /* All scratch allocations must come before grad_base to avoid overlap */
        pt_scratch_alloc_activations(&ctx->acts, &ctx->cfg, max_T, scratch_alloc);
        pt_scratch_alloc_backward_buf(&ctx->bb, &ctx->cfg, max_T, scratch_alloc);

        int tmp_sz = ctx->cfg.dim > ctx->cfg.hidden_dim
                   ? ctx->cfg.dim : ctx->cfg.hidden_dim;
        ctx->bb.d_temp = scratch_alloc(tmp_sz * 4);

        pt_scratch_alloc_state(&ctx->state, &ctx->cfg, scratch_alloc);

        /* grads: 1MB-aligned, placed after all scratch allocations */
        unsigned grad_base = (state_base + scratch_off + 0xFFFFF) & ~0xFFFFF;
        pt_scratch_alloc_grads(&ctx->grads, &ctx->cfg, ctx->shared_weights, grad_base);

        /* w_transpose: placed after grads in memory */
        unsigned wt_size = ctx->cfg.vocab_size * ctx->cfg.dim;
        unsigned ht_size = ctx->cfg.hidden_dim * ctx->cfg.dim;
        unsigned tb_elems = wt_size > ht_size ? wt_size : ht_size;
        unsigned tb_base = grad_base + (unsigned)ctx->grads._n_params * 4;
        tb_base = (tb_base + 15u) & ~15u;
        ctx->bb.w_transpose = (float *)tb_base;

        /* verify firmware gave us enough ARM RAM */
        uint32_t ram_end = arm_ram_end();
        uint32_t state_need = state_base + scratch_off;
        uint32_t grad_need  = tb_base + (unsigned)tb_elems * 4;
        uint32_t need = state_need > grad_need ? state_need : grad_need;
        printk("memory: state=0x%x grads=0x%x end=0x%x ram=0x%x (%dMB)\n",
               state_base, grad_base, need, ram_end, ram_end >> 20);
        if (need > ram_end)
            panic("need 0x%x but firmware gave 0x%x — check fixup.dat\n", need, ram_end);
    } else {
        /* ── inference only ── */
        pt_scratch_alloc_state(&ctx->state, &ctx->cfg, scratch_alloc);

        uint32_t ram_end = arm_ram_end();
        uint32_t need = state_base + scratch_off;
        printk("memory: state=0x%x end=0x%x ram=0x%x (%dMB)\n",
               state_base, need, ram_end, ram_end >> 20);
        if (need > ram_end)
            panic("need 0x%x but firmware gave 0x%x — check fixup.dat\n", need, ram_end);
    }

    pt_print_config(ctx);
}

void pt_pi_init_shard(pt_context_t *ctx, pt_shard_info_t *shard_out,
                      void *shard_data, int num_qpus, int max_T,
                      unsigned arena_bytes) {
    memset(ctx, 0, sizeof(*ctx));
    memset(shard_out, 0, sizeof(*shard_out));

    qpu_enable();
    perf_init();
    gpu_arena_init(arena_bytes);

    g_num_qpus = num_qpus;
    ctx->num_qpus = num_qpus;

    /* validate shard data */
    int dim0 = *(volatile int *)shard_data;
    if (dim0 <= 0 || dim0 > 65536)
        panic("no shard at 0x%x (dim=%d)\n", (unsigned)shard_data, dim0);

    /* Read global config + shard info */
    pt_config_t global_cfg;
    pt_load_shard_header(&global_cfg, shard_out, shard_data);

    ctx->shared_weights = ((const int *)shard_data)[5] > 0;

    /* Load shard weights */
    pt_load_shard_weights(&ctx->w, &global_cfg, shard_out, shard_data);

    ctx->matvec = matvec_gpu;
    ctx->max_T  = max_T;

    /* Set cfg.n_layers = n_local for allocation sizing.
     * All other fields (dim, hidden, vocab, seq_len) stay global. */
    ctx->cfg = global_cfg;
    ctx->cfg.n_layers = shard_out->n_local;

    /* Place buffers after the shard (1 MB aligned) */
    unsigned shard_base = (unsigned)shard_data;
    unsigned file_size = pt_shard_file_size(shard_data);
    state_base = (shard_base + file_size + 0x100000) & ~0xFFFFF;
    scratch_off = 0;

    if (max_T > 0) {
        /* Activations and backward buffers sized for LOCAL layers */
        pt_scratch_alloc_activations(&ctx->acts, &ctx->cfg, max_T, scratch_alloc);
        pt_scratch_alloc_backward_buf(&ctx->bb, &ctx->cfg, max_T, scratch_alloc);

        int tmp_sz = ctx->cfg.dim > ctx->cfg.hidden_dim
                   ? ctx->cfg.dim : ctx->cfg.hidden_dim;
        ctx->bb.d_temp = scratch_alloc(tmp_sz * 4);

        /* No KV cache needed for training (skip pt_scratch_alloc_state) */

        /* Grads: only for owned layers + embed/head.
         * We modify cfg temporarily to control what pt_scratch_alloc_grads
         * allocates, then restore it. */
        unsigned grad_base = (state_base + scratch_off + 0xFFFFF) & ~0xFFFFF;

        /* Build a grad config: only include embed if has_embed */
        pt_config_t grad_cfg = ctx->cfg;  /* n_layers already = n_local */
        if (!shard_out->has_embed) {
            /* No embed grads needed — set vocab_size = 0 for allocation.
             * We'll fix it back after. */
            grad_cfg.vocab_size = 0;
        }
        pt_scratch_alloc_grads(&ctx->grads, &grad_cfg, ctx->shared_weights, grad_base);

        /* If no embed, null out token_embedding grad pointer */
        if (!shard_out->has_embed) {
            ctx->grads.token_embedding = 0;
            ctx->grads.wcls = 0;
        }
        /* If no head, null out rms_final grad pointer */
        if (!shard_out->has_head)
            ctx->grads.rms_final_weight = 0;

        /* w_transpose: sized for owned components only */
        unsigned wt_elems;
        if (shard_out->has_head || shard_out->has_embed) {
            unsigned vt = (unsigned)global_cfg.vocab_size * global_cfg.dim;
            unsigned ht = (unsigned)global_cfg.hidden_dim * global_cfg.dim;
            wt_elems = vt > ht ? vt : ht;
        } else {
            /* Layer-only rank: just need max layer weight dim */
            unsigned dt = (unsigned)global_cfg.dim * global_cfg.dim;
            unsigned ht = (unsigned)global_cfg.hidden_dim * global_cfg.dim;
            wt_elems = dt > ht ? dt : ht;
        }
        unsigned tb_base = grad_base + (unsigned)ctx->grads._n_params * 4;
        tb_base = (tb_base + 15u) & ~15u;
        ctx->bb.w_transpose = (float *)tb_base;

        /* Verify RAM */
        uint32_t ram_end = arm_ram_end();
        uint32_t need = tb_base + (unsigned)wt_elems * 4;
        printk("shard memory: state=0x%x grads=0x%x wt_end=0x%x ram=0x%x (%dMB)\n",
               state_base, grad_base, need, ram_end, ram_end >> 20);
        printk("  shard=%dMB acts+bwd=%dKB grads=%dMB wt=%dMB\n",
               file_size >> 20, scratch_off >> 10,
               (ctx->grads._n_params * 4) >> 20, (wt_elems * 4) >> 20);
        if (need > ram_end)
            panic("need 0x%x but firmware gave 0x%x\n", need, ram_end);
    } else {
        /* Inference-only with shard */
        pt_scratch_alloc_state(&ctx->state, &ctx->cfg, scratch_alloc);

        uint32_t ram_end = arm_ram_end();
        uint32_t need = state_base + scratch_off;
        if (need > ram_end)
            panic("need 0x%x but firmware gave 0x%x\n", need, ram_end);
    }

    pt_print_config(ctx);
}

#else /* Mac host */

void pt_host_init(pt_context_t *ctx, void *weight_data, int max_T) {
    memset(ctx, 0, sizeof(*ctx));

    pt_load_config(&ctx->cfg, weight_data);
    ctx->shared_weights = ((const int *)weight_data)[5] > 0;
    pt_load_weights(&ctx->w, &ctx->cfg, weight_data);
    ctx->matvec = smatvec_cpu;
    ctx->max_T  = max_T;

    if (max_T > 0) {
        pt_alloc_activations(&ctx->acts, &ctx->cfg, max_T);
        pt_alloc_grads(&ctx->grads, &ctx->cfg, ctx->shared_weights);
        pt_alloc_backward_buf(&ctx->bb, &ctx->cfg, max_T);
    }

    pt_alloc_state(&ctx->state, &ctx->cfg);
}

void pt_free(pt_context_t *ctx) {
    if (ctx->max_T > 0) {
        pt_free_activations(&ctx->acts);
        pt_free_grads(&ctx->grads);
        pt_free_backward_buf(&ctx->bb);
    }
    pt_free_state(&ctx->state);
    if (ctx->trace) {
        free(ctx->trace);
        ctx->trace = NULL;
    }
}

#endif

/* ── inference helpers ──────────────────────────────────────── */

int pt_forward_step(pt_context_t *ctx, int token) {
    pt_forward(&ctx->cfg, &ctx->w, &ctx->state, token, ctx->pos, ctx->matvec);
    ctx->pos++;
    return argmax(ctx->state.logits, ctx->cfg.vocab_size);
}

void pt_reset_kv(pt_context_t *ctx) {
    int head_dim = ctx->cfg.dim / ctx->cfg.n_heads;
    int kv_dim   = ctx->cfg.n_kv_heads * head_dim;
    unsigned kv_bytes = ctx->cfg.n_layers * ctx->cfg.seq_len * kv_dim * sizeof(float);
    memset(ctx->state.key_cache,   0, kv_bytes);
    memset(ctx->state.value_cache, 0, kv_bytes);
    ctx->pos = 0;
}

void pt_print_config(const pt_context_t *ctx) {
    const pt_config_t *c = &ctx->cfg;
#ifdef __RPI__
    printk("\npitorch (%dd %dL %dKv) | %d QPUs\n",
           c->dim, c->n_layers, c->vocab_size / 1000, ctx->num_qpus);
    printk("  dim=%d hidden=%d L=%d H=%d kv=%d V=%d seq=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads,
           c->n_kv_heads, c->vocab_size, c->seq_len);
    if (ctx->max_T > 0)
        printk("  training: max_T=%d\n", ctx->max_T);
#else
    printf("pitorch (%dd %dL %dKv) | CPU\n",
           c->dim, c->n_layers, c->vocab_size / 1000);
    printf("  dim=%d hidden=%d L=%d H=%d kv=%d V=%d seq=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads,
           c->n_kv_heads, c->vocab_size, c->seq_len);
    if (ctx->max_T > 0)
        printf("  training: max_T=%d\n", ctx->max_T);
#endif
}

/* ── host utilities ─────────────────────────────────────────── */

#ifndef __RPI__
void *pt_read_file(const char *path, long *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);
    if (out_size) *out_size = sz;
    return buf;
}
#endif

/* ── profiling ──────────────────────────────────────────────── */

void pt_enable_trace(pt_context_t *ctx, pt_trace_t *preallocated) {
    if (preallocated) {
        ctx->trace = preallocated;
    } else {
#ifndef __RPI__
        ctx->trace = (pt_trace_t *)malloc(sizeof(pt_trace_t));
#endif
    }
    if (ctx->trace) pt_trace_init(ctx->trace);
}

void pt_disable_trace(pt_context_t *ctx) {
    if (ctx->trace) ctx->trace->enabled = 0;
}

/* ── convenience training step ──────────────────────────────── */

float pt_train_step(pt_context_t *ctx, const int *tokens, int T, float lr) {
    pt_trace_t *tr = ctx->trace;

    pt_trace_begin(tr, "train_step", "train", -1);

    pt_trace_begin(tr, "zero_grads", "train", -1);
    pt_zero_grads(&ctx->grads);
    pt_trace_end(tr);

    pt_trace_begin(tr, "forward", "fwd", -1);
    float loss = pt_forward_train(&ctx->cfg, &ctx->w, &ctx->acts,
                                  tokens, T, ctx->matvec, tr);
    pt_trace_end(tr);

    pt_trace_begin(tr, "backward", "bwd", -1);
    pt_backward(&ctx->cfg, &ctx->w, &ctx->grads, &ctx->acts,
                &ctx->bb, tokens, T, ctx->matvec, tr);
    pt_trace_end(tr);

    pt_trace_begin(tr, "sgd", "train", -1);
    pt_sgd_update(&ctx->w, &ctx->grads, lr, &ctx->cfg);
    pt_trace_end(tr);

    pt_trace_end(tr);
    return loss;
}
