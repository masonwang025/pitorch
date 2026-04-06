# tests/forward/ — End-to-end forward pass on Pi

Runs `stories15M.bin` (or any llama2.c-format model) through the full transformer forward pass on the Pi Zero, with both CPU and GPU matvec backends. Verifies token-for-token match against the Mac reference.

## SD card setup (one-time)

The GPU firmware loads weights from the SD card into RAM at boot — no UART transfer, no SD driver.

1. Download weights: `cd weights && ./download.sh` (or `./download.sh all`)
2. Create `weights/` on the SD card's boot partition
3. Copy `.bin` files into it
4. Add to `config.txt`: `initramfs weights/stories15M.bin 0x2000000`

Or use the setup script: `scripts/setup-sd.sh 15M` (auto-detects mounted SD card).

To switch models, change the filename in `config.txt` and reboot.

## Run

```bash
./run.sh
```

Assembles the matvec QPU kernel, builds the Pi binary (~19 KB), sends it over UART, and prints inference output.

## Memory map

```
0x00008000   test binary (~19 KB, sent over UART)
0x02000000   state buffers (~3.6 MB: KV cache, logits, activations)
0x02000000   weight data (loaded by GPU firmware from SD card)
```

## Files

- `test_forward.c` — Pi-side test: parses weights from RAM, runs 5 autoregressive steps with CPU and GPU matvec, prints comparison
- `host.py` — optional automated verifier: boots binary via Python, compares output against `host/expected.txt`
- `Makefile` — links model + ops + matvec + runtime for ARM cross-compilation
- `run.sh` — assembles QPU kernel, builds, runs via `my-install`

## Results (stories15M)

```
CPU:  ~19s/token   (ARM1176, smatvec_cpu)
GPU:  ~0.36s/token (12 QPUs, smatvec_tmu)  →  53x speedup, ~2.8 tok/s
```

All 5 tokens match the Mac/PyTorch reference exactly.
