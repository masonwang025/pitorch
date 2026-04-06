# 🥧 pitorch

LLaMA-2 inference and training on bare-metal (no operating system) Raspberry Pi Zero. Optimized with custom assembly kernels for the Raspberry Pi's GPU (VideoCore IV QPUs), achieving a 210x speedup over CPU.

Supports distributing work across four Pis, scaling comfortably to 110M parameters (in theory, ~400M fp32). Check out the full writeup [here](https://masonjwang.com/projects/pitorch)!

**⭐ This repo has the full PiTorch framework (GPU kernels, tools, and examples). Once you have a Pi Zero, a cable, and an SD card, you should be able to generate text in ~10 minutes ([setup guide](docs/setup.md)).**

## Running

```bash
cd examples
./run.sh generate                  # single-Pi inference
./run.sh train                     # single-Pi training
./run.sh generate-distributed      # 4-Pi inference
./run.sh train-distributed         # 4-Pi training
```

See [docs/setup.md](docs/setup.md) for full setup (toolchain, SD card, weights).
See [docs/hardware.md](docs/hardware.md) for 4-Pi wiring.

## Inference

`examples/generate.c` — the complete single-Pi inference program:

```c
void notmain(void) {
    mmu_init_and_enable();               // enable ARM D-cache (4x speedup)

    pt_tokenizer_t tok;
    pt_load_tokenizer(&tok, WEIGHT_ADDR, tmp_cfg.vocab_size);

    pt_context_t ctx;
    pt_pi_init(&ctx, WEIGHT_ADDR, NUM_QPUS, /*max_T=*/0, ARENA_SIZE);

    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, TEMPERATURE, TOPP, seed);

    while (1) {
        read_prompt(prompt, sizeof(prompt));
        pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                    prompt, MAX_TOKENS, ctx.matvec);
    }
}
```

`pt_pi_init` enables QPUs, maps the weight file from SD-card RAM into the struct, and initializes the GPU arena allocator. `pt_generate` handles BPE tokenization, KV-cache prefill, temperature/top-p sampling, and streams tokens back over UART.

## Training

Pass `max_T > 0` to `pt_pi_init` to allocate training buffers:

```c
pt_pi_init(&ctx, WEIGHT_ADDR, NUM_QPUS, SEQ_LEN, ARENA_SIZE);

for (int step = 0; step < N_STEPS; step++) {
    float loss = pt_train_step(&ctx, target, SEQ_LEN, LR);
    printk("step %d: loss=%f\n", step, loss);
}
```

`pt_train_step` is `zero_grads → forward (saving activations) → backward (7 per-operator passes) → SGD update`. Convergence on stories15M: loss 7.4 → 0.03 in 18 steps.

## 4-Pi distributed

`examples/generate-distributed.c` splits the model across 4 Pis. Each Pi loads only its weight shard:

```c
pt_pi_init_shard(&ctx, &shard, WEIGHT_ADDR, NUM_QPUS, /*max_T=*/0, ARENA_SIZE);

pt_dist_t dist;
pt_dist_setup(&dist, &shard, RANK, /*world_size=*/4);
pt_dist_ring_sync(&dist);

for (int i = 0; i < total_steps; i++) {
    int next = pt_dist_forward_step(&ctx, &dist, token);
    // R3 (head rank): print token. Layer ranks: forward layers, pass activation onward.
}
```

Activations flow ring: R3 (embed+head) → R0 → R1 → R2 → R3. The inter-Pi link is a custom 8-bit parallel GPIO bus (~8 MB/s), making communication less than 1% of step time for all tested models.

## Memory layout

```
Physical memory (512 MB)
            ┌──────────────────────────┐
0x00000000  │  code + stack            │  ~1 MB
            ├──────────────────────────┤
0x02000000  │  model weights  (58 MB)  │  ← start.elf loads
            │  activations             │    before ARM boots
            │  KV cache                │
            │  gradients               │
            ├──────────────────────────┤
0x1C000000  │  GPU memory   (64 MB)    │  gpu_mem=64 in config.txt
            └──────────────────────────┘
0x20000000
```

## Library layout

The library lives in `pitorch/`. The hardware-facing layers (runtime, kernels, ops) sit at the bottom; the model, training, and distributed logic build on top.

| Module          |                                                                |
| --------------- | -------------------------------------------------------------- |
| `pt.h` / `pt.c` | Context API (`pt_pi_init`, `pt_train_step`, `pt_forward_step`) |
| `model/`        | LLaMA-2 forward pass and weight layout                         |
| `ops/core/`     | CPU ops (rmsnorm, softmax, silu, rope) and math (no libm)      |
| `ops/matvec/`   | GPU matrix-vector multiply (QPU assembly + C wrapper)          |
| `ops/gemm/`     | GPU GEMM for training backward pass                            |
| `runtime/`      | QPU enable, GPU memory, D-cache                                |
| `train/`        | Forward + backward pass, SGD                                   |
| `text/`         | Tokenizer, sampler, generation                                 |
| `dist/`         | Distributed pipeline (GPIO ring, allreduce)                    |
| `profiler/`     | V3D counters, Chrome Tracing                                   |

`libpi/` — GPIO, UART, timer, ARM startup.

## Models

| Model       | Params | Size   | Single Pi        | 4 Pi      |
| ----------- | ------ | ------ | ---------------- | --------- |
| stories15M  | 15M    | 58 MB  | 12 tok/s         | 12 tok/s  |
| stories42M  | 42M    | 167 MB | 0.8 tok/s        | 0.8 tok/s |
| stories110M | 110M   | 438 MB | won't fit in RAM | 0.3 tok/s |

Download:

```bash
bash tools/download.sh
```
