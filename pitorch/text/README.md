# text/ — Text-layer concerns

Everything between "user types a string" and "model produces a string."

## Files

- `pt_text.h` — tokenizer, sampler, and generation API
- `pt_text.c` — all text-layer implementation

## What's here

- **Tokenizer**: load `tokenizer.bin`, BPE encode (string → token ids), decode (token id → string)
- **Sampler**: greedy argmax, temperature scaling, top-p (nucleus) sampling
- **Generate**: prefill + decode loop with streaming UART output

## Dependencies

- `model/llama2.h` — config, weights, state, forward pass types
- `ops/core/pt_ops.h` — argmax, softmax (used by sampler)
- `ops/core/pt_math.h` — expf (used by softmax inside sampler)
- `libpi/` — printk, timer_get_usec, string functions

## Testing

```bash
cd tests/generate && ./run.sh
```

One integration test covers encode → prefill → decode → stream.
