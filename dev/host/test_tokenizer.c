/*
 * Host-side tokenizer verification.
 * Loads tokenizer.bin, tests encode/decode against known reference.
 *
 * Build:  cc -O2 -Wall -I../text -I../model -I../ops/core \
 *         -o test_tokenizer test_tokenizer.c ../text/pt_text.c \
 *         ../ops/core/pt_ops.c ../ops/core/pt_math.c
 * Run:    ./test_tokenizer ../weights/tokenizer.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pt_text.h"

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

static int check_encode(pt_tokenizer_t *tok, const char *text,
                        const int *expected, int n_expected) {
    int tokens[512], n;
    pt_encode(tok, text, /*bos=*/1, /*eos=*/0, tokens, &n);

    printf("encode(\"%s\")\n  got:      [", text);
    for (int i = 0; i < n; i++) printf("%d%s", tokens[i], i < n-1 ? ", " : "");
    printf("]\n  expected: [");
    for (int i = 0; i < n_expected; i++) printf("%d%s", expected[i], i < n_expected-1 ? ", " : "");
    printf("]\n");

    if (n != n_expected) {
        printf("  FAIL (length: got %d, expected %d)\n\n", n, n_expected);
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (tokens[i] != expected[i]) {
            printf("  FAIL (mismatch at index %d: got %d, expected %d)\n\n",
                   i, tokens[i], expected[i]);
            return 0;
        }
    }
    printf("  PASS\n\n");
    return 1;
}

static void test_decode(pt_tokenizer_t *tok, const int *ids, int n) {
    printf("decode: ");
    int prev = 0;
    for (int i = 0; i < n; i++) {
        const char *piece = pt_decode(tok, prev, ids[i]);
        printf("%s", piece);
        prev = ids[i];
    }
    printf("\n\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <tokenizer.bin>\n", argv[0]);
        return 1;
    }

    long sz;
    void *data = read_file(argv[1], &sz);

    pt_tokenizer_t tok;
    pt_tokenizer_init(&tok, data, 32000);
    printf("loaded tokenizer: %d tokens, max_len=%d, file=%ld bytes\n\n",
           tok.vocab_size, tok.max_token_length, sz);

    int pass = 0, total = 0;

    /* reference encodings from sentencepiece (with BOS prepended) */
    int e1[] = {1, 9038, 2501, 263, 931};
    total++; pass += check_encode(&tok, "Once upon a time", e1, 5);

    int e2[] = {1, 319, 6635, 3290, 373};
    total++; pass += check_encode(&tok, "A cat sat on", e2, 5);

    int e3[] = {1, 15043};
    total++; pass += check_encode(&tok, "Hello", e3, 2);

    /* decode the Phase 3 greedy sequence */
    int phase3[] = {1, 9038, 2501, 263, 931, 29892};
    printf("decode Phase 3 sequence [1, 9038, 2501, 263, 931, 29892]:\n");
    test_decode(&tok, phase3, 6);

    /* round-trip: encode then decode */
    const char *rt_text = "Once upon a time in a city made of glass,";
    int rt_tokens[512], rt_n;
    pt_encode(&tok, rt_text, 1, 0, rt_tokens, &rt_n);
    printf("round-trip \"%s\":\n  tokens (%d): [", rt_text, rt_n);
    for (int i = 0; i < rt_n; i++) printf("%d%s", rt_tokens[i], i < rt_n-1 ? ", " : "");
    printf("]\n  decode: ");
    int prev = 0;
    for (int i = 0; i < rt_n; i++) {
        const char *piece = pt_decode(&tok, prev, rt_tokens[i]);
        printf("%s", piece);
        prev = rt_tokens[i];
    }
    printf("\n\n");

    printf("=== %d/%d encode tests passed ===\n", pass, total);

    free(data);
    return pass == total ? 0 : 1;
}
