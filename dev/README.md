# dev/

Host-side validation tools and on-Pi test suite. Not needed for normal use — see `examples/` for runnable demos.

- `host/` — Mac-side reference implementations: `reference.py` (Python inference reference), `test_train*` (SGD validation, interactive fine-tune, pipeline simulation). Build with `make` from that directory.
- `tests/` — On-device test suite covering ops, matvec, forward pass, training, and distributed transport. Each subdirectory has a `run.sh`.
