#!/bin/bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
PITORCH_ROOT="$TEST_DIR/../../.."
KERNEL_DIR="$PITORCH_ROOT/pitorch/kernels"
OPS_DIR="$PITORCH_ROOT/pitorch/ops/gemm"

for pair in "gemm_tmu.qasm:gemm_tmu_shader"; do
    QASM="${pair%%:*}"
    SHADER="${pair##*:}"
    echo "ASSEMBLING $QASM"
    out=$(cd "$KERNEL_DIR/gemm" && \
        vc4asm -I"../include/" -i vc4.qinc \
        -c "$OPS_DIR/${SHADER}.c" -h "$OPS_DIR/${SHADER}.h" \
        "$QASM" 2>&1)
    if [[ -n $out ]]; then
        echo "ASSEMBLY FAILED:" && printf '%s\n' "$out"
        exit 1
    fi
done

echo "RUNNING MAKE" && make -C "$TEST_DIR"
