/*
 * generate-distributed.c — Interactive distributed inference across 4 Pi Zeros.
 *
 * Pipeline-parallel greedy decoding. Each token flows through the ring:
 * head rank embeds → layer ranks process → head rank classifies → next token.
 *
 * Topology (42M model, 8 layers):
 *   R3: embed + head    (interactive prompt + generates tokens)
 *   R0: layers [0,3)
 *   R1: layers [3,6)
 *   R2: layers [6,8)
 *
 *   Forward: R3 embed → R0 → R1 → R2 → R3 head → argmax
 *
 * ── How to run ──────────────────────────────────────────────────────
 *
 *   cd examples && ./run.sh generate-distributed
 *
 *   Per-Pi logs are written to examples/logs/pi{0,1,2,3}.log in real-time.
 *   Console shows only the interactive generation (from R3/head rank).
 *
 * ── SD cards ────────────────────────────────────────────────────────
 *
 *   PIE0: initramfs weights/shards/42M/rank0.bin 0x2000000
 *   PIE1: initramfs weights/shards/42M/rank1.bin 0x2000000
 *   PIE2: initramfs weights/shards/42M/rank2.bin 0x2000000
 *   PIE3: initramfs weights/shards/42M/rank3_full.bin 0x2000000
 *
 * ════════════════════════════════════════════════════════════════════
 */

#include "rpi.h"
#include "mmu.h"
#include "pt.h"
#include "pt_ops.h"
#include "pt_text.h"
#include "pt_proto.h"
#include "pt_dist_pipeline.h"

#ifndef RANK
#error "compile with -DRANK=0/1/2/3"
#endif

#define WEIGHT_ADDR  ((void *)0x02000000)
#define NUM_QPUS     12
#define ARENA_SIZE   (1 * 1024 * 1024)
#define MAX_TOKENS   256

/* Read a line from UART.
 * No local echo — the host terminal already displays typed characters.
 * (Single-Pi generate.c needs uart_put8 echo because exec replaces
 *  the shell, but here the tee pipeline preserves terminal echo.) */
static int read_prompt(char *buf, int max_len) {
    int i = 0;
    while (i < max_len - 1) {
        int c = uart_get8();
        if (c == '\n' || c == '\r') break;
        if (c == 0x7f || c == '\b') {
            if (i > 0) i--;
            continue;
        }
        buf[i++] = (char)c;
    }
    buf[i] = '\0';
    return i;
}

void notmain(void) {
    mmu_init_and_enable();

    /* ── Load tokenizer BEFORE init (head rank only) ──
     *    Tokenizer is appended after the shard file on the head rank's SD card.
     *    Must load before pt_pi_init_shard, which places buffers over that region.
     *    pt_tokenizer_init copies into static arrays, so source can be overwritten. */
    pt_tokenizer_t tok;
    int has_tok = 0;
    {
        pt_config_t tmp;
        pt_shard_info_t tmp_shard;
        pt_load_shard_header(&tmp, &tmp_shard, WEIGHT_ADDR);
        if (tmp_shard.has_head) {
            unsigned shard_bytes = pt_shard_file_size(WEIGHT_ADDR);
            const void *tok_data = (const char *)WEIGHT_ADDR + shard_bytes;
            pt_tokenizer_init(&tok, tok_data, tmp.vocab_size);
            has_tok = 1;
        }
    }

    /* ── Initialize model from weight shard ── */
    pt_context_t ctx;
    pt_shard_info_t shard;
    pt_pi_init_shard(&ctx, &shard, WEIGHT_ADDR, NUM_QPUS, 0, ARENA_SIZE);

    /* ── Initialize distributed pipeline ── */
    pt_dist_t dist;
    pt_dist_setup(&dist, &shard, RANK, 4);
    /* Layer ranks: verbose logging to UART (captured in log files).
     * Head rank: quiet — only user-facing text on console. */
    if (dist.has_head && dist.has_embed) {
        pt_dist_set_verbose(&dist, 0);
    } else {
        pt_dist_set_verbose(&dist, 1);
        pt_dist_print(&dist);
    }

    /* ── Synchronize the ring ── */
    pt_dist_ring_sync(&dist);

    if (dist.has_head && dist.has_embed)
        printk("\nready.\n\n");

    /* ── Prompt + generation ── */
    int prompt_tokens[512];
    int n_prompt = 1;       /* default: BOS only */
    int total_steps;

    if (dist.has_head && dist.has_embed) {
        /* R3: read prompt from user */
        char prompt[256];
        printk("> ");
        int len = read_prompt(prompt, sizeof(prompt));

        if (len > 0) {
            printk("\n...");
            pt_encode(&tok, prompt, /*bos=*/1, /*eos=*/0,
                      prompt_tokens, &n_prompt);
        } else {
            printk("\n...");
            prompt_tokens[0] = 1;  /* BOS */
            n_prompt = 1;
        }

        total_steps = (n_prompt - 1) + MAX_TOKENS;

        /* Broadcast total_steps around the ring so layer ranks know */
        uint32_t count = (uint32_t)total_steps;
        pt_proto_send(&dist.downstream.base, PT_OP_PING,
                      &count, sizeof(count));
        uint32_t op, plen, dummy;
        pt_proto_recv(&dist.upstream.base, &op, &dummy,
                      sizeof(dummy), &plen);

    } else {
        /* Layer ranks: receive total_steps from upstream, forward */
        uint32_t op, plen;
        uint32_t count;
        if (pt_proto_recv(&dist.upstream.base, &op, &count,
                          sizeof(count), &plen) < 0) {
            printk("FAIL: recv step count\n");
            clean_reboot();
        }
        total_steps = (int)count;
        pt_proto_send(&dist.downstream.base, PT_OP_PING,
                      &count, sizeof(count));
    }

    /* ── Prefill + decode ── */
    int token = prompt_tokens[0];
    int prev = prompt_tokens[0];
    uint32_t t0 = timer_get_usec();
    int n_decode = 0;
    int done = 0;

    for (int i = 0; i < total_steps; i++) {
        int next = pt_dist_forward_step(&ctx, &dist, token);

        if (dist.has_head && dist.has_embed) {
            if (i < n_prompt - 1) {
                /* Prefill: force next prompt token */
                next = prompt_tokens[i + 1];
            } else if (!done) {
                /* Decode: print generated token */
                n_decode++;
                if (next == 1 || next == 2) {
                    done = 1;  /* stop printing but keep loop running for layer ranks */
                } else if (has_tok) {
                    const char *piece = pt_decode(&tok, prev, next);
                    printk("%s", piece);
                }
            }
            prev = token;
            token = next;
        }
    }

    if (dist.has_head && dist.has_embed) {
        uint32_t elapsed = (timer_get_usec() - t0) / 1000;
        unsigned s  = elapsed / 1000;
        unsigned ds = (elapsed % 1000) / 100;
        unsigned ms_per = n_decode > 0 ? elapsed / (unsigned)n_decode : 0;
        printk("\n[%d tokens | %d.%ds | %d ms/tok]\n",
               n_decode, s, ds, ms_per);
    }

    clean_reboot();
}
