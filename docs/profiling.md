# Profiling & Tracing

> **Scope:** How PiTorch measures performance — what gets instrumented, what the output looks like, and how to use it in your own code.
>
> **Code:** `pitorch/profiler/` — `trace.h` (Chrome Tracing) and `profiler.h` (V3D hardware counters).

---

## TL;DR

PiTorch has two profiling systems:

1. **Chrome Tracing** (`pt_trace_t`) — records wall-clock timing for every op in the forward and backward pass, outputs JSON viewable in [Perfetto](https://ui.perfetto.dev). Works on both Mac and Pi.

2. **V3D hardware counters** (`perf_t`) — reads the GPU's built-in counters for QPU execution cycles, idle cycles, TMU stalls, and VPM stalls. Pi only.

```
┌─────────────────────────────────────────────────────────┐
│  Chrome Tracing (trace.h)                               │
│                                                         │
│  What:  Wall-clock timing per op                        │
│  Where: Mac (trace.json file) or Pi (JSON over UART)    │
│  View:  Perfetto (ui.perfetto.dev) or traces/viewer.html│
│                                                         │
│  Instruments:                                           │
│    Forward:  embedding, per-layer (qkv, attention, ffn),│
│              classifier, loss                           │
│    Backward: classifier, cls_weight_grad, per-layer     │
│              (transpose, ffn, wo, attention, qkv),      │
│              embedding                                  │
│    Training: step (wraps fwd+bwd+sgd)                   │
│    Distributed: fwd_send, fwd_recv, bwd_send, bwd_recv  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  V3D Hardware Counters (profiler.h)                     │
│                                                         │
│  What:  GPU pipeline metrics per matvec call            │
│  Where: Pi only (reads V3D PCTR registers)              │
│                                                         │
│  Counters:                                              │
│    qpu_exec_cyc  — QPU execution cycles                 │
│    qpu_idle_cyc  — QPU idle cycles (waiting for work)   │
│    qpu_tmu_stall — TMU stall cycles (memory latency)    │
│    qpu_vpm_stall — VPM stall cycles (DMA latency)       │
│    wall_us       — wall-clock microseconds              │
└─────────────────────────────────────────────────────────┘
```

---

## Chrome Tracing: the main profiling tool

### Quick start

```c
// 1. Declare and init
pt_trace_t trace;
pt_trace_init(&trace);

// 2. Attach to context (instruments forward_train + backward automatically)
ctx.trace = &trace;

// 3. Run your workload
pt_train_step(&ctx, tokens, T, lr);

// 4. Export
pt_trace_write_json(&trace, "trace.json");   // Mac — writes file
pt_trace_emit_uart(&trace);                   // Pi  — prints JSON over UART
```

Open `trace.json` in [Perfetto](https://ui.perfetto.dev) or drag it into `traces/viewer.html`.

### What gets instrumented

When `ctx.trace` is non-NULL, the forward and backward passes automatically record timing for every operation. Here's what a single training step looks like:

```
step ─────────────────────────────────────────────────────────────────
│
├── embedding                          (token lookup)
├── L0_layer ─┬── L0_qkv               (query/key/value projections + RoPE)
│             ├── L0_attention          (scaled dot-product attention)
│             └── L0_ffn               (SiLU gate + up/down projections)
├── L1_layer ─┬── L1_qkv
│             ├── L1_attention
│             └── L1_ffn
├── ...
├── classifier                         (final rmsnorm + vocab projection)
├── loss                               (cross-entropy)
│
├── classifier [bwd]                   (d_logits → d_residual)
│   ├── cls_transpose                  (transpose wcls for backward matvec)
│   └── cls_weight_grad                (outer product: d_logits × x_final)
├── L5_layer [bwd] ─┬── L5_transpose  (transpose w1/w2/w3 for backward)
│                    ├── L5_ffn        (FFN backward: SiLU grad, gate grad)
│                    ├── L5_wo         (output projection backward)
│                    ├── L5_attention  (attention backward)
│                    └── L5_qkv       (QKV projection backward + RoPE backward)
├── ...
├── embedding [bwd]                    (scatter-add into embedding grad)
└── sgd                                (weight update)
```

For distributed training, communication events are also traced:

```
step ────────────────────────────────────────────
├── fwd_recv       (GPIO: receive activations from upstream)
├── fwd_layers     (compute local layers)
├── fwd_send       (GPIO: send activations to downstream)
├── bwd_recv       (GPIO: receive gradients from downstream)
├── bwd_layers     (compute local backward)
├── bwd_send       (GPIO: send gradients to upstream)
└── sgd            (update local weights)
```

### Using trace in your own code

Wrap any region with `pt_trace_begin` / `pt_trace_end`:

```c
pt_trace_begin(&trace, "my_operation", "custom", -1);
// ... your code ...
pt_trace_end(&trace);
```

Parameters:
- `name` — event name (shows up in the trace viewer)
- `cat` — category: `"fwd"`, `"bwd"`, `"sgd"`, `"comm"`, `"train"`, or any string
- `layer` — transformer layer index (0-based), or `-1` for non-layer events. When >= 0, the event name becomes `L{layer}_{name}` in the JSON.

Calls nest — begin/end pairs form a stack (max depth 32):

```c
pt_trace_begin(&trace, "outer", "fwd", -1);
  pt_trace_begin(&trace, "inner", "fwd", 0);
  pt_trace_end(&trace);   // closes "inner"
pt_trace_end(&trace);     // closes "outer"
```

All functions are **no-ops** when the trace pointer is NULL or `trace.enabled == 0`, so you can leave instrumentation in production code with zero overhead.

### Capacity

- **4096 events** max (`PT_TRACE_MAX_EVENTS`) — enough for ~10 training steps of a 6-layer model, or ~3 steps of a 12-layer model
- **32 nesting depth** (`PT_TRACE_MAX_DEPTH`)
- Call `pt_trace_reset(&trace)` to discard events and reuse the buffer

### Output format

Chrome Tracing JSON (`"ph":"X"` = complete events):

```json
[
  {"name":"embedding","cat":"fwd","ph":"X","ts":1234567,"dur":500,"pid":0,"tid":0},
  {"name":"L0_qkv","cat":"fwd","ph":"X","ts":1235067,"dur":12000,"pid":0,"tid":0},
  ...
]
```

On Pi, this is wrapped in sentinels for `capture-trace.sh` to extract:

```
---TRACE_BEGIN---
[...json...]
---TRACE_END---
```

---

## V3D Hardware Counters (Pi only)

For GPU-level profiling of individual matvec calls:

```c
perf_init();                    // configure V3D counter sources (once)

perf_start();                   // zero counters, start timer
smatvec_tmu(W, x, y, M, K, num_qpus, NULL);
perf_t p = perf_stop();         // read counters

perf_print("my_matvec", M, K, 1, &p);
```

Output:

```
my_matvec  M=768 K=768 N=1  wall=1234us  exec=45000cyc  idle=200cyc  tmu_stall=12000cyc  vpm_stall=50cyc
```

### What the counters tell you

| Counter | What it measures | What high values mean |
|---------|-----------------|----------------------|
| `qpu_exec_cyc` | Cycles QPUs spent executing instructions | Baseline — this is useful work |
| `qpu_idle_cyc` | Cycles QPUs spent waiting for dispatch | Load imbalance or insufficient parallelism |
| `qpu_tmu_stall` | Cycles stalled waiting for TMU (texture memory reads) | Memory-bound — weights don't fit in L2 cache |
| `qpu_vpm_stall` | Cycles stalled waiting for VPM (vertex pipe memory DMA) | Result writeback bottleneck |
| `wall_us` | Wall-clock microseconds | End-to-end latency |

The TMU stall ratio (`tmu_stall / exec_cyc`) is the key metric — it tells you how memory-bound your matvec is. For large models (42M+), this is typically > 50%.

---

## Capturing traces from Pi

```
┌──────────┐   UART    ┌──────────────┐  extract   ┌──────────────────────────┐
│  Pi Zero │ ────────▶ │  capture-    │ ─────────▶ │ traces/runs/<name>/      │
│  (runs   │  printk   │  trace.sh    │  sentinels │   trace.json  → Perfetto │
│  binary) │  output   │ (my-install) │            │   meta.json   → metadata │
└──────────┘           └──────────────┘            │   uart.log    → raw log  │
                                                   └──────────────────────────┘
```

```bash
# Single Pi — captures UART output, extracts trace.json + meta.json
./tools/scripts/capture-trace.sh demo/tests/train/test_train.bin "baseline" "before optimization"
# Output: traces/runs/20260314_153000_baseline/
#   ├── trace.json   — open in Perfetto
#   ├── meta.json    — run metadata (model, config, results)
#   └── uart.log     — raw UART output
```

### Viewing traces

**Option 1: Perfetto** — go to [ui.perfetto.dev](https://ui.perfetto.dev), drag in `trace.json`. Best for zooming into individual ops.

**Option 2: Local viewer** — open `traces/viewer.html` in a browser, drag in trace files. Shows sidebar with run metadata.

---

## Saved traces

All profiling runs are saved in `traces/runs/` with `meta.json` describing the setup:

| Run | Model | What it captures |
|-----|-------|-----------------|
| `test_phase6/` | 15M | Baseline training (pre-optimization) |
| `exp1_hoist_transpose/` | 15M | After hoisting weight transposes out of backward loop |
| `exp2_skip_last_pos/` | 15M | After skipping backward at position T-1 |
| `exp3_batched_cls_grad/` | 15M | After batching classifier weight gradient |
| `exp3b_tiled_cls_grad/` | 15M | After cache-tiling the classifier gradient (BW_TILE=6) |
| `scaling_15M_gpu64/` | 15M | Scaling baseline (gpu_mem=64) |
| `scaling_15M_gpu32/` | 15M | Scaling with gpu_mem=32 |
| `scaling_42M_gpu64/` | 42M | 42M inference (gpu_mem=64) |
| `scaling_42M_gpu32/` | 42M | 42M inference (gpu_mem=32) |
| `scaling_110M_single_attempt/` | 110M | 110M single-Pi crash (no trace — silent death) |
| `dist_42M_4pi_train/` | 42M | 4-Pi distributed training |
| `dist_110M_4pi_infer/` | 110M | 4-Pi distributed inference |
| `dist_110M_4pi_train/` | 110M | 4-Pi distributed training |

---

## What we actually measure (and why)

### Inference profiling

The key question: **where does each token's time go?**

```
15M single Pi (82 ms/tok)
├───────────────────────────────────────────────────────┤ GPU matvec (72%)
├──────────────┤ attention (15%)
├──────────┤ other (13%)

110M 4-Pi pipeline (3274 ms/tok)
├──────────────────────────────────┤ stage 0: fwd layers [0,4)  (880 ms)
├──────────────────────────────────┤ stage 1: fwd layers [4,8)  (880 ms)
├──────────────────────────────────┤ stage 2: fwd layers [8,12) (880 ms)
├────────────────────┤ head (classifier 768→32K)                (617 ms)
├┤ GPIO × 4 transfers                                          (12 ms)
```

Inference is **compute-bound** on matvec — the GPU spends most cycles waiting for TMU to fetch weight data from SDRAM through the L2 cache. Optimization levers: cache tiling, reducing memory traffic.

### Training profiling

The key question: **what dominates backward pass time?**

```
15M single Pi (23.5 s/step, T=8)
│◀── forward (5%) ──▶│◀────────────── backward (84%) ─────────────▶│◀ sgd (11%) ▶│
│       1.2 s         │                  19.7 s                     │    2.6 s     │
                      │                                             │
                      ├─ per-layer: 2.2 s × 6 layers = 13.2 s      │
                      └─ classifier weight grad: 8.5 s ← bottleneck│
```

```
110M 4-Pi distributed (184 s/step, T=8)

R3  │emb│fs│   fwd_recv (21s)   │head_fwd│    head_bwd (30s)    │bs│      bwd_recv (123s)       │sgd│
    │   │  │                    │ (5.3s) │                      │  │                            │   │
R0  │fr │  fwd (7s)   │fs│            bwd_recv (131s)            │   bwd_layers (41s)   │bs│ sgd │
    │   │             │  │                                       │                      │  │     │
R1  │   │fr│  fwd (7s) │fs│           bwd_recv (131s)            │   bwd_layers (41s)   │bs│ sgd │
    │   │  │           │  │                                      │                      │  │     │
R2  │   │  │fr│ fwd (7s)│fs│          bwd_recv (131s)            │   bwd_layers (41s)   │bs│ sgd │

    time ──────────────────────────────────────────────────────────────────────────────────────────▶
    fr/fs/bs = GPIO recv/send (~3ms)
```

The backward pass dominates because every matvec in the forward pass becomes **three** matvecs in backward (input gradient + weight gradient + the original forward), plus transpose overhead. The classifier weight gradient is especially expensive: it's an outer product of `[vocab_size × dim]` — for 110M that's 32000 × 768 = 24.6M multiply-adds per token per position.

Most of each rank's time is spent in `bwd_recv` — waiting for the backward pipeline to drain. R0 can't start its backward until R3→R2→R1 have all finished theirs. This is the fundamental cost of sequential pipeline parallelism without microbatching.
