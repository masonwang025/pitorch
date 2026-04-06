#!/bin/bash
# examples/run.sh — Build, deploy, and run PiTorch examples.
#
# Usage:
#   ./run.sh generate                        # single Pi (default Pi 0)
#   ./run.sh train                           # single Pi
#   PI_DEVICE=2 ./run.sh generate            # choose which Pi
#   ./run.sh train-distributed               # 4 Pis, logs to examples/logs/
#   ./run.sh generate-distributed            # 4 Pis, interactive generation
#
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
PITORCH_ROOT="$(cd "$DEMO_DIR/.." && pwd)"
BOOTLOADER="$PITORCH_ROOT/tools/bin/my-install"
KERNEL_DIR="$PITORCH_ROOT/pitorch/kernels"
OPS_DIR="$PITORCH_ROOT/pitorch/ops/matvec"
LOG_DIR="$DEMO_DIR/logs"

# ── Load device mapping from devices.conf ──
declare -a PORT_MAP NAME_MAP
while read -r idx suffix label desc; do
    [[ "$idx" =~ ^#.*$ || -z "$idx" ]] && continue
    PORT_MAP[$idx]="$suffix"
    NAME_MAP[$idx]="$desc"
done < "$PITORCH_ROOT/devices.conf"

get_port() { echo "/dev/cu.usbserial-${PORT_MAP[$1]}"; }

# ── Validate arguments ──
if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh <demo>"
    echo ""
    echo "Single-Pi demos (use PI_DEVICE=N to select Pi, default 0):"
    echo "  generate                 Interactive text generation"
    echo "  train                    SGD training + verification"
    echo ""
    echo "Distributed demos (4 Pis, logs written to demo/logs/):"
    echo "  train-distributed        Pipeline-parallel training"
    echo "  generate-distributed     Pipeline-parallel inference"
    exit 1
fi

DEMO="$1"

# ── Assemble QPU shaders ──
echo ">>> assembling QPU shaders..."
(cd "$KERNEL_DIR/matvec" && \
    vc4asm -I"../include/" -i vc4.qinc \
    -c "$OPS_DIR/matvec_tmu_shader.c" -h "$OPS_DIR/matvec_tmu_shader.h" \
    "matvec_tmu.qasm" 2>&1) || { echo "MATVEC ASSEMBLY FAILED (install vc4asm)"; exit 1; }

GEMM_OPS_DIR="$PITORCH_ROOT/pitorch/ops/gemm"
(cd "$KERNEL_DIR/gemm" && \
    vc4asm -I"../include/" -i vc4.qinc \
    -c "$GEMM_OPS_DIR/gemm_rect_tmu_shader.c" -h "$GEMM_OPS_DIR/gemm_rect_tmu_shader.h" \
    "gemm_rect_tmu.qasm" 2>&1) || { echo "GEMM ASSEMBLY FAILED"; exit 1; }

# ── Build ──
echo ">>> building demos..."
make -C "$DEMO_DIR" 2>&1 | tail -5

# ── Helper functions for distributed deployment ──

check_distributed_ports() {
    MISSING=0
    for d in 0 1 2 3; do
        PORT=$(get_port "$d")
        if [ ! -e "$PORT" ]; then
            echo "WARNING: Pi $d port $PORT not found" >&2
            MISSING=1
        fi
    done
    if [ "$MISSING" -eq 1 ]; then
        echo ""
        echo "Available ports:"
        ls /dev/cu.usbserial-* 2>/dev/null || echo "  (none)"
        echo ""
    fi
}

check_distributed_bins() {
    for r in 0 1 2 3; do
        BIN="$DEMO_DIR/ranks/${DEMO}-rank${r}.bin"
        if [ ! -f "$BIN" ]; then
            echo "ERROR: $BIN not found (build failed?)" >&2
            exit 1
        fi
    done
}

print_distributed_banner() {
    echo "╔══════════════════════════════════════════╗"
    echo "║  demo: $DEMO (4 Pis)                     "
    echo "╠══════════════════════════════════════════╣"
    for r in 0 1 2 3; do
        echo "║  rank $r → $(get_port $r)  ← ranks/${DEMO}-rank${r}.bin"
    done
    echo "╠══════════════════════════════════════════╣"
    echo "║  logs: examples/logs/pi{0,1,2,3}.log      "
    echo "╚══════════════════════════════════════════╝"
}

# ── Dispatch ──
case "$DEMO" in
    generate|train)
        # Single-Pi deployment
        PI_DEVICE="${PI_DEVICE:-0}"
        PORT=$(get_port "$PI_DEVICE")
        BIN="$DEMO_DIR/$DEMO.bin"

        if [ ! -f "$BIN" ]; then
            echo "ERROR: $BIN not found (build failed?)" >&2
            exit 1
        fi

        echo ""
        echo "╔══════════════════════════════════════════╗"
        echo "║  demo: $DEMO"
        echo "║  Pi $PI_DEVICE → $PORT"
        echo "╚══════════════════════════════════════════╝"
        echo ""
        exec "$BOOTLOADER" "$PORT" "$BIN"
        ;;

    train-distributed)
        # 4-Pi deployment with per-rank logging (all output visible)
        mkdir -p "$LOG_DIR"
        check_distributed_ports
        check_distributed_bins

        echo ""
        print_distributed_banner
        echo ""

        # Clear old logs
        for r in 0 1 2 3; do
            > "$LOG_DIR/pi${r}.log"
        done

        # Deploy in REVERSE order (rank 3 first, rank 0 last).
        PIDS=()
        for r in 3 2 1; do
            PORT=$(get_port "$r")
            BIN="$DEMO_DIR/ranks/${DEMO}-rank${r}.bin"
            echo ">>> deploying rank $r (background)..."
            "$BOOTLOADER" "$PORT" "$BIN" 2>&1 | tee "$LOG_DIR/pi${r}.log" &
            PIDS+=($!)
            sleep 2
        done

        # Rank 0 in foreground, also logged
        PORT=$(get_port 0)
        BIN="$DEMO_DIR/ranks/${DEMO}-rank0.bin"
        echo ">>> deploying rank 0 (foreground)..."
        echo ""
        "$BOOTLOADER" "$PORT" "$BIN" 2>&1 | tee "$LOG_DIR/pi0.log"

        # Wait for background ranks
        for pid in "${PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        echo ""
        echo "=== all ranks complete ==="
        echo "logs saved to: $LOG_DIR/pi{0,1,2,3}.log"
        ;;

    generate-distributed)
        # 4-Pi interactive generation — only head rank (R3) on console
        mkdir -p "$LOG_DIR"
        check_distributed_ports
        check_distributed_bins

        echo ""
        print_distributed_banner
        echo ""

        # Clear old logs
        for r in 0 1 2 3; do
            > "$LOG_DIR/pi${r}.log"
        done

        # Deploy layer ranks 0, 1, 2 in background (logs only, no console).
        # Layer ranks have verbose=1 — detailed timing goes to their log files.
        PIDS=()
        for r in 0 1 2; do
            PORT=$(get_port "$r")
            BIN="$DEMO_DIR/ranks/${DEMO}-rank${r}.bin"
            echo ">>> deploying rank $r..."
            "$BOOTLOADER" "$PORT" "$BIN" > "$LOG_DIR/pi${r}.log" 2>&1 &
            PIDS+=($!)
            sleep 1
        done

        # Deploy head rank (R3) — interactive foreground.
        # R3 has verbose=0: only user-facing text on console.
        # tee copies everything to pi3.log. No grep — it breaks
        # character-at-a-time streaming (buffers until newline).
        PORT=$(get_port 3)
        BIN="$DEMO_DIR/ranks/${DEMO}-rank3.bin"
        echo ">>> deploying head rank (R3)..."
        echo ""
        "$BOOTLOADER" "$PORT" "$BIN" 2>&1 | tee "$LOG_DIR/pi3.log"

        # Wait for background ranks
        for pid in "${PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        echo ""
        echo "logs saved to: $LOG_DIR/pi{0,1,2,3}.log"
        ;;

    *)
        echo "ERROR: unknown demo '$DEMO'" >&2
        echo "Available: generate, train, train-distributed, generate-distributed" >&2
        exit 1
        ;;
esac
