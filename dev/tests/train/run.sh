#!/bin/bash
# Build and run Pi-side training tests.
#
# usage: bash run.sh [test]
#
#   test = train (default), finetune, interactive
#
# Examples:
#   bash run.sh              # build + run test_train
#   bash run.sh finetune     # build + run test_finetune
#   bash run.sh interactive  # build + run test_train_interactive
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
PITORCH_ROOT="$TEST_DIR/../../.."
KERNEL_DIR="$PITORCH_ROOT/pitorch/kernels"
OPS_DIR="$PITORCH_ROOT/pitorch/ops/matvec"
MY_INSTALL="$PITORCH_ROOT/tools/bin/my-install"

TEST="${1:-train}"

case "$TEST" in
    train)       MAKEFILE="Makefile.pi"          BIN="test_train.bin" ;;
    finetune)    MAKEFILE="Makefile.finetune"     BIN="test_finetune.bin" ;;
    interactive) MAKEFILE="Makefile.interactive"  BIN="test_train_interactive.bin" ;;
    *)           echo "unknown test: $TEST (try: train, finetune, interactive)" && exit 1 ;;
esac

echo "=== assembling QPU kernels ==="
out=$(cd "$KERNEL_DIR/matvec" && \
    vc4asm -I"../include/" -i vc4.qinc \
    -c "$OPS_DIR/matvec_tmu_shader.c" -h "$OPS_DIR/matvec_tmu_shader.h" \
    "matvec_tmu.qasm" 2>&1)
if [[ -n $out ]]; then
    echo "assembly warnings/errors:" && printf '%s\n' "$out"
fi

echo "=== building $BIN ==="
make -C "$TEST_DIR" -f "$MAKEFILE"

echo "=== running on Pi ==="
"$MY_INSTALL" "$TEST_DIR/$BIN"
