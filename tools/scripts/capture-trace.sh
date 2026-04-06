#!/bin/bash
#
# capture-trace.sh — capture a pitorch trace from Pi over UART.
#
# Runs the binary via my-install, captures UART output, extracts
# trace JSON and metadata between sentinel markers, and saves to
# traces/runs/<timestamp>_<name>/.
#
# Usage:
#   ./capture-trace.sh <binary.bin> [name] [notes]
#
# Example:
#   ./capture-trace.sh ../../demo/tests/train/test_train.bin "baseline" "before transpose caching"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRACES_DIR="$ROOT/traces/runs"
BOOTLOADER="$ROOT/tools/bin/my-install"
TTYUSB="${TTYUSB:-/dev/ttyUSB0}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <binary.bin> [name] [notes]"
    exit 1
fi

BINARY="$1"
NAME="${2:-trace}"
NOTES="${3:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$TRACES_DIR/${TIMESTAMP}_${NAME}"

mkdir -p "$RUN_DIR"

echo "=== pitorch trace capture ==="
echo "binary: $BINARY"
echo "output: $RUN_DIR"
echo ""

UART_LOG="$RUN_DIR/uart.log"

"$BOOTLOADER" "$TTYUSB" "$BINARY" 2>&1 | tee "$UART_LOG" &
INSTALL_PID=$!

wait $INSTALL_PID || true

echo ""
echo "=== extracting trace ==="

# Extract trace JSON between sentinels
if grep -q "---TRACE_BEGIN---" "$UART_LOG"; then
    sed -n '/---TRACE_BEGIN---/,/---TRACE_END---/p' "$UART_LOG" \
        | grep -v '---TRACE' \
        > "$RUN_DIR/trace.json"
    EVENT_COUNT=$(grep -c '"ph"' "$RUN_DIR/trace.json" || echo 0)
    echo "extracted trace.json ($EVENT_COUNT events)"
else
    echo "warning: no trace sentinels found in output"
fi

# Extract metadata between sentinels
if grep -q "---META_BEGIN---" "$UART_LOG"; then
    sed -n '/---META_BEGIN---/,/---META_END---/p' "$UART_LOG" \
        | grep -v '---META' \
        > "$RUN_DIR/meta.json"
    echo "extracted meta.json"
else
    # Generate minimal meta.json from command line args
    cat > "$RUN_DIR/meta.json" <<EOF
{
    "name": "$NAME",
    "device": "pi_zero",
    "notes": "$NOTES",
    "timestamp": "$TIMESTAMP"
}
EOF
    echo "generated meta.json from args"
fi

echo ""
echo "done: $RUN_DIR"
echo "  trace.json  — open in Perfetto (ui.perfetto.dev) or traces/viewer.html"
echo "  meta.json   — run metadata"
echo "  uart.log    — full UART output"
