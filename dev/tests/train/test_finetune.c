/*
 * Fine-tuning demo: train stories15M on a small hardcoded dataset,
 * generate before and after to show the model learned something.
 *
 * SD card: initramfs weights/stories15M_full.bin 0x2000000
 *   (combined: model weights + tokenizer.bin)
 */
#include "rpi.h"
#include "mmu.h"
#include "pt.h"
#include "pt_text.h"
#include "profiler.h"

#define NUM_QPUS     12
#define ARENA_SIZE   (100 * 1024 * 1024)
#define WEIGHT_BASE  0x02000000u

#define T_SEQ        16
#define N_EPOCHS     5
#define LR           0.001f
#define GEN_TOKENS   48

static const char *train_data[] = {
    "Mason is a student who builds computers.",
    "The Pi Zero costs five dollars and runs code.",
    "Twelve GPU cores do math for neural networks.",
    "PiTorch trains language models on bare metal.",
    "A tiny chip learned to write stories by itself.",
};
#define N_EXAMPLES (sizeof(train_data) / sizeof(train_data[0]))

static const char *gen_prompt = "Once upon a time";

void notmain(void) {
    mmu_init_and_enable();

    /* Load tokenizer before pt_pi_init — state buffers overlap tokenizer
     * region in the combined file. */
    pt_config_t tmp_cfg;
    pt_load_config(&tmp_cfg, (void *)WEIGHT_BASE);

    pt_tokenizer_t tok;
    pt_load_tokenizer(&tok, (void *)WEIGHT_BASE, tmp_cfg.vocab_size);

    pt_context_t ctx;
    pt_pi_init(&ctx, (void *)WEIGHT_BASE, NUM_QPUS, T_SEQ, ARENA_SIZE);

    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, 0.0f, 0.9f, timer_get_usec());

    /* ── tokenize all training data ── */
    int all_tokens[N_EXAMPLES][T_SEQ + 1];
    int all_lengths[N_EXAMPLES];

    printk("\ndataset:\n");
    for (int i = 0; i < (int)N_EXAMPLES; i++) {
        int raw[128];
        int n;
        pt_encode(&tok, train_data[i], 1, 0, raw, &n);
        if (n > T_SEQ) n = T_SEQ;
        for (int j = 0; j < n; j++) all_tokens[i][j] = raw[j];
        all_lengths[i] = n;
        printk("  [%d] %d tok: \"%s\"\n", i, n, train_data[i]);
    }

    /* ── generate BEFORE training ── */
    printk("\n--- before training ---\n");
    pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                gen_prompt, GEN_TOKENS, ctx.matvec);
    printk("\n");

    /* ── fine-tuning loop ── */
    printk("\n--- training ---\n");
    unsigned t_start = timer_get_usec();

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        for (int i = 0; i < (int)N_EXAMPLES; i++) {
            float loss = pt_train_step(&ctx, all_tokens[i],
                                       all_lengths[i], LR);
            epoch_loss += loss;
        }
        epoch_loss /= (float)N_EXAMPLES;
        unsigned elapsed = (timer_get_usec() - t_start) / 1000000;
        printk("epoch %d: avg_loss=", epoch); pt_pf(epoch_loss, 4);
        printk("  (%ds)\n", elapsed);
    }

    unsigned total_s = (timer_get_usec() - t_start) / 1000000;
    printk("training done: %d steps in %ds\n",
           N_EPOCHS * (int)N_EXAMPLES, total_s);

    /* ── generate AFTER training ── */
    printk("\n--- after training ---\n");
    pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                gen_prompt, GEN_TOKENS, ctx.matvec);
    printk("\n");

    /* ── memorization check ── */
    printk("\n--- memorization check ---\n");
    for (int i = 0; i < (int)N_EXAMPLES; i++) {
        pt_reset_kv(&ctx);
        int token = all_tokens[i][0];
        int match = 1;
        for (int t = 0; t < all_lengths[i] - 1; t++) {
            int next = pt_forward_step(&ctx, token);
            if (next != all_tokens[i][t + 1]) match = 0;
            token = next;
        }
        printk("[%d] %s\n", i, match ? "MATCH" : "partial");
    }
}
