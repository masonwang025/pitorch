/*
 * Inference speed benchmark.
 * Runs greedy generation for 32 tokens with per-token timing.
 * No UART input needed — fully automatic.
 */
#include "rpi.h"
#include <string.h>
#include "gpu.h"
#include "arena.h"
#include "mmu.h"
#include "profiler.h"
#include "pt_ops.h"
#include "pt_math.h"
#include "matvec.h"
#include "llama2.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (1 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u
#define GEN_TOKENS   32

static unsigned state_base;
static unsigned scratch_off;
static void *scratch_alloc(unsigned bytes) {
    void *p = (void *)(state_base + scratch_off);
    scratch_off += (bytes + 15u) & ~15u;
    return p;
}

static unsigned gpu_us_accum;
static int gpu_call_count;

static void matvec_gpu(const float *W, const float *x, float *y,
                       int out_dim, int in_dim) {
    unsigned t0 = timer_get_usec();
    gpu_arena_reset();
    smatvec_tmu(W, x, y, out_dim, in_dim, NUM_QPUS, NULL);
    gpu_us_accum += timer_get_usec() - t0;
    gpu_call_count++;
}

void notmain(void) {
    mmu_init_and_enable();
    qpu_enable();
    perf_init();
    gpu_arena_init(ARENA_SIZE);

    void *data = (void *)WEIGHT_BASE;
    if (*(volatile int *)data <= 0 || *(volatile int *)data > 65536)
        panic("no weights at 0x%x\n", WEIGHT_BASE);

    pt_config_t cfg;
    pt_load_config(&cfg, data);
    pt_weights_t w;
    pt_load_weights(&w, &cfg, data);

    unsigned file_size = pt_file_size(data);
    state_base = (WEIGHT_BASE + file_size + 0x100000) & ~0xFFFFF;
    scratch_off = 0;
    pt_state_t s;
    pt_scratch_alloc_state(&s, &cfg, scratch_alloc);

    printk("\n=== inference benchmark ===\n");
    printk("%dd %dL %dH %dKv | %d QPUs | %d tokens\n\n",
           cfg.dim, cfg.n_layers, cfg.n_heads, cfg.vocab_size / 1000,
           NUM_QPUS, GEN_TOKENS);

    int token = 1; /* BOS */
    unsigned total_us = 0;
    unsigned total_gpu_us = 0;
    int total_gpu_calls = 0;

    for (int pos = 0; pos < GEN_TOKENS; pos++) {
        gpu_us_accum = 0;
        gpu_call_count = 0;

        unsigned t0 = timer_get_usec();
        pt_forward(&cfg, &w, &s, token, pos, matvec_gpu);
        unsigned t1 = timer_get_usec();
        unsigned elapsed = t1 - t0;
        total_us += elapsed;
        total_gpu_us += gpu_us_accum;
        total_gpu_calls += gpu_call_count;

        int next = argmax(s.logits, cfg.vocab_size);
        printk("[%d] tok=%d -> %d  (%d ms, gpu=%d ms/%d calls, cpu=%d ms)\n",
               pos, token, next, elapsed / 1000,
               gpu_us_accum / 1000, gpu_call_count,
               (elapsed - gpu_us_accum) / 1000);
        token = next;
    }

    unsigned avg_ms = total_us / GEN_TOKENS / 1000;
    unsigned tps10 = (unsigned)((uint64_t)GEN_TOKENS * 10000000 / total_us);
    unsigned gpu_pct = total_gpu_us * 100 / total_us;
    unsigned cpu_ms = (total_us - total_gpu_us) / 1000;
    printk("\n%d tokens in %d ms  (avg %d ms/tok, %d.%d tok/s)\n",
           GEN_TOKENS, total_us / 1000, avg_ms, tps10 / 10, tps10 % 10);
    printk("gpu: %d ms (%d pct)  cpu: %d ms  calls: %d\n",
           total_gpu_us / 1000, gpu_pct, cpu_ms, total_gpu_calls);
}
