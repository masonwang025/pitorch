#!/bin/bash
#
# Usage: run-test.sh <test_dir> <qasm_file> <shader_name>
#
# Assembles a .qasm kernel into <shader_name>.c/.h in <test_dir>,
# then runs make in that directory.
#
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <test_dir> <qasm_file> <shader_name>" >&2
    exit 1
fi

TEST_DIR="$1"
QASM="$2"
SHADER="$3"

PITORCH_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KERNEL_DIR="$PITORCH_ROOT/pitorch/kernels"

echo "ASSEMBLING QASM"
out=$(cd "$KERNEL_DIR/examples" && \
    vc4asm -I"../include/" -i vc4.qinc \
    -c "$TEST_DIR/${SHADER}.c" -h "$TEST_DIR/${SHADER}.h" \
    "$QASM" 2>&1)

if [[ -n $out ]]; then
    echo "ASSEMBLY FAILED:" && printf '%s\n' "$out"
    exit 1
else
    echo "RUNNING MAKE" && make -C "$TEST_DIR"
fi
