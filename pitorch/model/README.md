# model/ — Llama2 weight loader and forward pass

Portable C, no Pi-specific dependencies. Compiled natively on Mac (`tools/host/`) and cross-compiled for Pi (`tests/forward/`).

## Files

- `llama2.h` — `pt_config_t`, `pt_weights_t`, `pt_state_t`, `pt_matvec_fn`, `pt_forward()`
- `llama2.c` — weight parsing, state allocation (host only), full transformer forward pass

## Weight format

karpathy/llama2.c `.bin` checkpoints. 28-byte header (7 ints: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len) followed by raw fp32 weights in a fixed order. Positive `vocab_size` means the classifier shares the embedding weights.

## Forward pass

`pt_forward()` takes a `pt_matvec_fn` function pointer for all linear layers, so callers choose the backend:

```c
pt_forward(&cfg, &w, &s, token, pos, smatvec_cpu);   // CPU reference
pt_forward(&cfg, &w, &s, token, pos, matvec_gpu);     // GPU (12 QPUs)
```

The forward pass follows llama2.c closely: embedding lookup → per-layer (RMSNorm, QKV projection, RoPE, multi-head attention with KV cache, output projection + residual, SwiGLU FFN + residual) → final RMSNorm → classifier.

## State allocation

On Mac: `pt_alloc_state()` uses malloc. On Pi (`__RPI__`): not available — the test allocates state buffers from a scratch region in high physical RAM (see `tests/forward/test_forward.c`).
