#!/bin/bash
set -e
cd "$(dirname "$0")"

WEIGHTS=../../weights/stories15M.bin

# ── download weights if needed ──
if [ ! -f "$WEIGHTS" ]; then
    ../../weights/download.sh 15M
    echo ""
fi

# ── PyTorch reference ──
echo "=== PyTorch reference ==="
python3 reference.py "$WEIGHTS" expected.txt
echo ""

# ── build C test ──
echo "=== building C forward pass ==="
make -s
echo ""

# ── run C test ──
echo "=== C forward pass ==="
./test_forward "$WEIGHTS" expected.txt
