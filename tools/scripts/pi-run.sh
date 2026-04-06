#!/bin/bash
# pi-run.sh — unified Pi deployment script
#
# Analogous to CUDA_VISIBLE_DEVICES for Raspberry Pi Zeros.
#
# Environment:
#   PI_DEVICES  Comma-separated device indices (default: "0")
#               e.g. PI_DEVICES=0,1,2,3 for all 4 Pis
#               e.g. PI_DEVICES=0 for single-Pi (default)
#
# Usage:
#   Single-Pi:  pi-run.sh test.bin
#   Multi-Pi:   PI_DEVICES=0,1,2,3 pi-run.sh test_rank0.bin test_rank1.bin test_rank2.bin test_rank3.bin
#               (binaries are matched to devices in order)
#
# Device index → USB serial port mapping:
#   0 → /dev/cu.usbserial-1330   (Pi 0, ring rank 0, SD=PIE0)
#   1 → /dev/cu.usbserial-1320   (Pi 1, ring rank 1, SD=PIE1)
#   2 → /dev/cu.usbserial-1310   (Pi 2, ring rank 2, SD=PIE2)
#   3 → /dev/cu.usbserial-1340   (Pi 3, ring rank 3, SD=PIE3)
#
# For distributed tests, binaries are deployed in REVERSE order
# (highest rank first, rank 0 last) with 2s delays between each,
# matching the GPIO ring protocol requirements.
#
set -euo pipefail

PITORCH_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BOOTLOADER="$PITORCH_ROOT/tools/bin/my-install"

# ── device mapping (from devices.conf) ──
declare -a PORT_MAP NAME_MAP
while read -r idx suffix label desc; do
    [[ "$idx" =~ ^#.*$ || -z "$idx" ]] && continue
    PORT_MAP[$idx]="$suffix"
    NAME_MAP[$idx]="$desc"
done < "$PITORCH_ROOT/devices.conf"

# ── parse PI_DEVICES ──
PI_DEVICES="${PI_DEVICES:-0}"
IFS=',' read -ra DEVICES <<< "$PI_DEVICES"
NUM_DEVICES=${#DEVICES[@]}

# ── validate devices ──
for d in "${DEVICES[@]}"; do
    if [[ "$d" -lt 0 || "$d" -gt 3 ]]; then
        echo "ERROR: invalid device index '$d' (must be 0-3)" >&2
        exit 1
    fi
done

# ── resolve serial ports ──
get_port() {
    local idx="$1"
    echo "/dev/cu.usbserial-${PORT_MAP[$idx]}"
}

# ── check ports exist ──
check_ports() {
    local missing=0
    for d in "${DEVICES[@]}"; do
        local port
        port=$(get_port "$d")
        if [ ! -e "$port" ]; then
            echo "WARNING: ${NAME_MAP[$d]} port $port not found" >&2
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        echo "" >&2
        echo "Available ports:" >&2
        ls /dev/cu.usbserial-* 2>/dev/null || echo "  (none)" >&2
        echo "" >&2
    fi
}

# ── validate args ──
if [ $# -lt 1 ]; then
    echo "Usage: pi-run.sh <binary> [binary2 binary3 ...]" >&2
    echo "" >&2
    echo "Environment:" >&2
    echo "  PI_DEVICES=0         single Pi (default)" >&2
    echo "  PI_DEVICES=0,1       two Pis" >&2
    echo "  PI_DEVICES=0,1,2,3   four Pis (full ring)" >&2
    echo "" >&2
    echo "Device mapping:" >&2
    for i in 0 1 2 3; do
        echo "  $i → /dev/cu.usbserial-${PORT_MAP[$i]}  ${NAME_MAP[$i]}" >&2
    done
    exit 1
fi

BINARIES=("$@")

# ── single-Pi mode ──
if [ "$NUM_DEVICES" -eq 1 ]; then
    if [ ${#BINARIES[@]} -ne 1 ]; then
        echo "ERROR: single-Pi mode expects 1 binary, got ${#BINARIES[@]}" >&2
        exit 1
    fi
    check_ports
    port=$(get_port "${DEVICES[0]}")
    echo "╔══════════════════════════════════════════╗"
    echo "║  pi-run: ${NAME_MAP[${DEVICES[0]}]}  (single)          ║"
    echo "╚══════════════════════════════════════════╝"
    echo "  port: $port"
    echo "  binary: ${BINARIES[0]}"
    echo ""
    exec "$BOOTLOADER" "$port" "${BINARIES[0]}"
fi

# ── multi-Pi mode ──
if [ ${#BINARIES[@]} -ne "$NUM_DEVICES" ]; then
    echo "ERROR: ${NUM_DEVICES} devices but ${#BINARIES[@]} binaries provided" >&2
    echo "  devices: ${DEVICES[*]}" >&2
    echo "  binaries: ${BINARIES[*]}" >&2
    exit 1
fi

check_ports

echo "╔══════════════════════════════════════════╗"
echo "║  pi-run: ${NUM_DEVICES}-Pi deployment                  ║"
echo "╚══════════════════════════════════════════╝"
for i in $(seq 0 $((NUM_DEVICES - 1))); do
    d="${DEVICES[$i]}"
    echo "  rank $i: ${NAME_MAP[$d]} → $(get_port "$d") ← ${BINARIES[$i]}"
done
echo ""

# Deploy in REVERSE order (highest rank first, rank 0 last)
# This is required because higher ranks block on GPIO recv,
# and rank 0 triggers the execution chain.
PIDS=()
for i in $(seq $((NUM_DEVICES - 1)) -1 0); do
    d="${DEVICES[$i]}"
    port=$(get_port "$d")
    bin="${BINARIES[$i]}"

    if [ "$i" -eq 0 ]; then
        # Rank 0 runs in foreground (last to deploy)
        echo ">>> deploying rank $i to ${NAME_MAP[$d]} (foreground)..."
        "$BOOTLOADER" "$port" "$bin"
    else
        echo ">>> deploying rank $i to ${NAME_MAP[$d]} (background)..."
        "$BOOTLOADER" "$port" "$bin" &
        PIDS+=($!)
        sleep 2
    fi
done

# Wait for background processes
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "=== all ranks complete ==="
