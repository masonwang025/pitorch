# host/ — Mac-side forward pass validation

Runs stories15M through a CPU-only llama2 forward pass compiled with the native Mac compiler, using the exact same C ops from `pitorch/ops/core/`. Validates against a PyTorch reference.

## Usage

```bash
./run.sh    # downloads model, runs PyTorch reference, builds + runs C test
make        # build only
make clean  # remove artifacts
```

## Files

- `test_forward.c` — loads stories15M.bin, runs 5 autoregressive steps, compares against expected.txt
- `reference.py` — PyTorch reference: same forward pass, generates expected.txt with top-5 indices
- `run.sh` — orchestration: download → reference → build → test
