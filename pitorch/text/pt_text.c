#include <string.h>
#include "pt_text.h"
#include "pt_ops.h"
#include "pt_math.h"

#ifdef __RPI__
#include "rpi.h"
#else
#include <stdio.h>
#include <stdint.h>
#define printk printf
static inline uint32_t timer_get_usec(void) { return 0; }
#endif

#define MAX_VOCAB 32000

/* ── static pools (BSS, ~1.3 MB total) ──
 * ARM binary is ~25 KB at 0x8000. BSS extends to ~1.3 MB.
 * Nothing else lives between the binary and STATE_BASE (32 MB). */

static char              vocab_pool[512 * 1024];
static float             scores_pool[MAX_VOCAB];
static char             *vocab_ptrs[MAX_VOCAB];
static pt_token_index_t  sorted_pool[MAX_VOCAB];

/* ═══════════════════════════════════════════════════════════════════
 *  TOKENIZER
 * ═══════════════════════════════════════════════════════════════════ */

void pt_load_tokenizer(pt_tokenizer_t *t, const void *combined_file, int vocab_size) {
    unsigned model_bytes = pt_file_size(combined_file);
    const void *tok_data = (const char *)combined_file + model_bytes;
    pt_tokenizer_init(t, tok_data, vocab_size);
}

void pt_tokenizer_init(pt_tokenizer_t *t, const void *data, int vocab_size) {
    const unsigned char *p = (const unsigned char *)data;

    unsigned max_len;
    memcpy(&max_len, p, 4);
    p += 4;

    t->vocab_size       = vocab_size;
    t->max_token_length = (int)max_len;
    t->vocab            = vocab_ptrs;
    t->vocab_scores     = scores_pool;
    t->sorted_vocab     = sorted_pool;
    t->sorted_ready     = 0;

    char *pool = vocab_pool;

    for (int i = 0; i < vocab_size; i++) {
        float score;
        memcpy(&score, p, 4);
        t->vocab_scores[i] = score;
        p += 4;

        int len;
        memcpy(&len, p, 4);
        p += 4;

        memcpy(pool, p, len);
        pool[len] = '\0';      /* binary doesn't null-terminate */
        t->vocab[i] = pool;
        pool += len + 1;
        p += len;
    }
}

/* ── decode (id → string) ── */

static unsigned char byte_val_buf[2];

const char *pt_decode(pt_tokenizer_t *t, int prev_token, int token) {
    char *piece = t->vocab[token];

    /* strip leading space after BOS (sentencepiece convention) */
    if (prev_token == 1 && piece[0] == ' ')
        piece++;

    /* handle <0xHH> byte tokens (no sscanf on bare metal) */
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x'
        && piece[5] == '>' && strlen(piece) == 6) {
        unsigned val = 0;
        for (int i = 3; i < 5; i++) {
            char c = piece[i];
            val <<= 4;
            if      (c >= '0' && c <= '9') val |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') val |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') val |= (unsigned)(c - 'A' + 10);
        }
        byte_val_buf[0] = (unsigned char)val;
        byte_val_buf[1] = '\0';
        return (const char *)byte_val_buf;
    }

    return piece;
}

/* ── sort + search for encode ── */

static void shellsort_vocab(pt_token_index_t *arr, int n) {
    for (int gap = n / 2; gap > 0; gap /= 2)
        for (int i = gap; i < n; i++) {
            pt_token_index_t tmp = arr[i];
            int j = i;
            while (j >= gap && strcmp(arr[j - gap].str, tmp.str) > 0) {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = tmp;
        }
}

static int str_lookup(const char *str, const pt_token_index_t *sorted, int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(str, sorted[mid].str);
        if (cmp == 0) return sorted[mid].id;
        if (cmp < 0)  hi = mid - 1;
        else           lo = mid + 1;
    }
    return -1;
}

static void ensure_sorted(pt_tokenizer_t *t) {
    if (t->sorted_ready) return;
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id  = i;
    }
    shellsort_vocab(t->sorted_vocab, t->vocab_size);
    t->sorted_ready = 1;
}

/* ── encode (string → token ids) ── */

void pt_encode(pt_tokenizer_t *t, const char *text, int bos, int eos,
               int *tokens, int *n_tokens) {
    ensure_sorted(t);

    int n = 0;

    if (bos) tokens[n++] = 1;

    if (text[0] != '\0') {
        /* sentencepiece: prepend dummy space so first word merges with ▁ */
        int sp_id = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (sp_id != -1)
            tokens[n++] = sp_id;
    }

    /* initial encoding: one token per UTF-8 codepoint (or byte fallback) */
    for (const char *c = text; *c; ) {
        int cplen = 1;
        if      ((*c & 0x80) == 0)    cplen = 1;
        else if ((*c & 0xE0) == 0xC0) cplen = 2;
        else if ((*c & 0xF0) == 0xE0) cplen = 3;
        else if ((*c & 0xF8) == 0xF0) cplen = 4;

        char cp_buf[8];
        memcpy(cp_buf, c, cplen);
        cp_buf[cplen] = '\0';

        int id = str_lookup(cp_buf, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[n++] = id;
        } else {
            for (int i = 0; i < cplen; i++)
                tokens[n++] = (unsigned char)c[i] + 3;
        }
        c += cplen;
    }

    /* iterative BPE merge */
    char merge_buf[256];
    while (1) {
        float best_score = -1e10f;
        int best_idx = -1, best_id = -1;

        for (int i = 0; i < n - 1; i++) {
            strcpy(merge_buf, t->vocab[tokens[i]]);
            strcat(merge_buf, t->vocab[tokens[i + 1]]);
            int id = str_lookup(merge_buf, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_idx   = i;
                best_id    = id;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n - 1; i++)
            tokens[i] = tokens[i + 1];
        n--;
    }

    if (eos) tokens[n++] = 2;
    *n_tokens = n;
}

/* ═══════════════════════════════════════════════════════════════════
 *  SAMPLER
 * ═══════════════════════════════════════════════════════════════════ */

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float rng_float(uint64_t *state) {
    return (float)(rng_next(state) >> 40) * 5.9604644775390625e-08f;
}

void pt_sampler_init(pt_sampler_t *s, int vocab_size,
                     float temp, float topp, uint64_t seed) {
    s->vocab_size  = vocab_size;
    s->temperature = temp;
    s->topp        = topp;
    s->rng_state   = seed ? seed : 42;
}

typedef struct { float prob; int index; } prob_index_t;

static prob_index_t pi_pool[MAX_VOCAB];

static void sift_down(prob_index_t *a, int n, int i) {
    while (1) {
        int largest = i, l = 2*i + 1, r = 2*i + 2;
        if (l < n && a[l].prob > a[largest].prob) largest = l;
        if (r < n && a[r].prob > a[largest].prob) largest = r;
        if (largest == i) break;
        prob_index_t tmp = a[i]; a[i] = a[largest]; a[largest] = tmp;
        i = largest;
    }
}

static void heapsort_pi(prob_index_t *a, int n) {
    for (int i = n/2 - 1; i >= 0; i--) sift_down(a, n, i);
    for (int i = n - 1; i > 0; i--) {
        prob_index_t tmp = a[0]; a[0] = a[i]; a[i] = tmp;
        sift_down(a, i, 0);
    }
}

int pt_sample(pt_sampler_t *s, float *logits) {
    int n = s->vocab_size;

    if (s->temperature == 0.0f)
        return argmax(logits, n);

    /* temperature */
    float inv_t = 1.0f / s->temperature;
    for (int i = 0; i < n; i++)
        logits[i] *= inv_t;

    softmax(logits, n);

    float coin = rng_float(&s->rng_state);

    /* top-p (nucleus) sampling */
    if (s->topp > 0.0f && s->topp < 1.0f) {
        int n0 = 0;
        float cutoff = (1.0f - s->topp) / (float)(n - 1);
        for (int i = 0; i < n; i++) {
            if (logits[i] >= cutoff) {
                pi_pool[n0].index = i;
                pi_pool[n0].prob  = logits[i];
                n0++;
            }
        }

        if (n0 > 0) {
            /* heapsort gives ascending; walk from end for descending */
            heapsort_pi(pi_pool, n0);

            float cumprob = 0.0f;
            int last_idx = 0;
            for (int i = n0 - 1; i >= 0; i--) {
                cumprob += pi_pool[i].prob;
                if (cumprob > s->topp) { last_idx = i; break; }
            }

            float r = coin * cumprob;
            float cdf = 0.0f;
            for (int i = n0 - 1; i >= last_idx; i--) {
                cdf += pi_pool[i].prob;
                if (r < cdf) return pi_pool[i].index;
            }
            return pi_pool[last_idx].index;
        }
    }

    /* sample from full distribution */
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += logits[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

/* ═══════════════════════════════════════════════════════════════════
 *  GENERATION LOOP
 * ═══════════════════════════════════════════════════════════════════ */

void pt_generate(const pt_config_t *cfg, const pt_weights_t *w, pt_state_t *s,
                 pt_tokenizer_t *tok, pt_sampler_t *sampler,
                 const char *prompt, int max_tokens, pt_matvec_fn matvec) {
    int prompt_tokens[512];
    int n_prompt;
    pt_encode(tok, prompt, /*bos=*/1, /*eos=*/0, prompt_tokens, &n_prompt);

    if (max_tokens > cfg->seq_len)
        max_tokens = cfg->seq_len;

    /* zero KV cache */
    int head_dim = cfg->dim / cfg->n_heads;
    int kv_dim   = cfg->n_kv_heads * head_dim;
    unsigned kv_bytes = (unsigned)cfg->n_layers * cfg->seq_len
                        * kv_dim * sizeof(float);
    memset(s->key_cache,   0, kv_bytes);
    memset(s->value_cache, 0, kv_bytes);

    /* prompt text is already displayed by the caller */

    int token = prompt_tokens[0];
    int n_decode = 0;
    uint32_t prefill_us = 0, decode_us = 0;

    for (int pos = 0; pos < max_tokens; pos++) {
        uint32_t t0 = timer_get_usec();
        pt_forward(cfg, w, s, token, pos, matvec);
        uint32_t elapsed = timer_get_usec() - t0;

        int next;
        if (pos < n_prompt - 1) {
            /* prefill: force next prompt token */
            next = prompt_tokens[pos + 1];
            prefill_us += elapsed;
        } else {
            /* decode: sample */
            next = pt_sample(sampler, s->logits);
            decode_us += elapsed;
            n_decode++;

            if (next == 1 || next == 2) break;  /* BOS or EOS */

            const char *piece = pt_decode(tok, token, next);
            printk("%s", piece);
        }

        token = next;
    }

    printk("\n");

    /* timing summary */
    unsigned total_tok = (unsigned)(n_prompt - 1) + (unsigned)n_decode;
    printk("[%d tokens | prefill %d.%ds",
           total_tok,
           prefill_us / 1000000, (prefill_us / 100000) % 10);

    printk(" | decode %d.%ds",
           decode_us / 1000000, (decode_us / 100000) % 10);

    if (n_decode > 0 && decode_us > 0) {
        unsigned tps10 = (unsigned)((uint64_t)n_decode * 10000000 / decode_us);
        printk(" | %d.%d tok/s", tps10 / 10, tps10 % 10);
    }
    printk("]\n");
}
