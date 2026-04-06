#!/bin/bash
QASM=parallel_add.qasm
SHADER=add_shader

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$TEST_DIR/../../../../tools/scripts/run-test.sh" "$TEST_DIR" "$QASM" "$SHADER"
