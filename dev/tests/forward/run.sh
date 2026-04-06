#!/bin/bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
PITORCH_ROOT="$TEST_DIR/../../.."
KERNEL_DIR="$PITORCH_ROOT/pitorch/kernels"
OPS_DIR="$PITORCH_ROOT/pitorch/ops/matvec"

# ── assemble matvec kernel ──
echo "ASSEMBLING matvec_tmu.qasm"
out=$(cd "$KERNEL_DIR/matvec" && \
    vc4asm -I"../include/" -i vc4.qinc \
    -c "$OPS_DIR/matvec_tmu_shader.c" -h "$OPS_DIR/matvec_tmu_shader.h" \
    "matvec_tmu.qasm" 2>&1)
if [[ -n $out ]]; then
    echo "ASSEMBLY FAILED:" && printf '%s\n' "$out"
    exit 1
fi

echo "RUNNING MAKE" && make -C "$TEST_DIR"
