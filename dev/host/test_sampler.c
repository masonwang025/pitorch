/*
 * Quick sampler sanity check: greedy vs temperature vs top-p.
 * Run after test_generate_host to verify sampling modes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pt_text.h"
#include "pt_ops.h"

static void *read_file(const char *path, long *out_size) {
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

static void generate_n(const char *label, pt_config_t *cfg, pt_weights_t *w,
                       pt_state_t *s, pt_tokenizer_t *tok,
                       float temp, float topp, uint64_t seed, int n_tokens) {
    pt_sampler_t sampler;
    pt_sampler_init(&sampler, cfg->vocab_size, temp, topp, seed);

    int head_dim = cfg->dim / cfg->n_heads;
    int kv_dim = cfg->n_kv_heads * head_dim;
    unsigned kv_bytes = (unsigned)cfg->n_layers * cfg->seq_len * kv_dim * sizeof(float);
    memset(s->key_cache, 0, kv_bytes);
    memset(s->value_cache, 0, kv_bytes);

    int token = 1;
    printf("[%s temp=%.1f topp=%.1f] ", label, temp, topp);

    for (int pos = 0; pos < n_tokens; pos++) {
        pt_forward(cfg, w, s, token, pos, smatvec_cpu);
        int next = pt_sample(&sampler, s->logits);
        if (next == 1 || next == 2) break;
        const char *piece = pt_decode(tok, token, next);
        printf("%s", piece);
        token = next;
    }
    printf("\n\n");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.bin> <tokenizer.bin>\n", argv[0]);
        return 1;
    }

    long model_sz;
    void *model_data = read_file(argv[1], &model_sz);
    pt_config_t cfg;
    pt_load_config(&cfg, model_data);
    pt_weights_t w;
    pt_load_weights(&w, &cfg, model_data);
    pt_state_t s;
    pt_alloc_state(&s, &cfg);

    long tok_sz;
    void *tok_data = read_file(argv[2], &tok_sz);
    pt_tokenizer_t tok;
    pt_tokenizer_init(&tok, tok_data, cfg.vocab_size);

    printf("=== sampler test (15 tokens from BOS) ===\n\n");

    generate_n("greedy",  &cfg, &w, &s, &tok, 0.0f, 1.0f, 42, 15);
    generate_n("temp0.8", &cfg, &w, &s, &tok, 0.8f, 1.0f, 42, 15);
    generate_n("topp0.9", &cfg, &w, &s, &tok, 0.8f, 0.9f, 42, 15);
    generate_n("seed1",   &cfg, &w, &s, &tok, 0.8f, 0.9f, 1,  15);
    generate_n("seed2",   &cfg, &w, &s, &tok, 0.8f, 0.9f, 2,  15);

    pt_free_state(&s);
    free(model_data);
    free(tok_data);
    return 0;
}
