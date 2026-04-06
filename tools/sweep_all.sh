#!/bin/bash
# Pin sweep all 4 links in the ring.
# Tests each link independently (re-flashes both Pis per link).
#
# Usage: ./tools/sweep_all.sh [link]
#   No args: test all 4 links sequentially
#   1: link 1→2    2: link 2→3    3: link 3→4    4: link 4→1

BIN_DIR="dev/tests/dist/gpio_test"
DRIVER="$BIN_DIR/test_pin_sweep_rank0.bin"
READER="$BIN_DIR/test_pin_sweep_rank1.bin"
INSTALL="tools/bin/my-install"

# Load device mapping from devices.conf
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
declare -a PORT_SUFFIXES
while read -r idx suffix label desc; do
    [[ "$idx" =~ ^#.*$ || -z "$idx" ]] && continue
    PORT_SUFFIXES[$idx]="$suffix"
done < "$ROOT_DIR/devices.conf"

PI1="/dev/cu.usbserial-${PORT_SUFFIXES[0]}"
PI2="/dev/cu.usbserial-${PORT_SUFFIXES[1]}"
PI3="/dev/cu.usbserial-${PORT_SUFFIXES[2]}"
PI4="/dev/cu.usbserial-${PORT_SUFFIXES[3]}"

LINK_SENDER=("$PI1" "$PI2" "$PI3" "$PI4")
LINK_RECVER=("$PI2" "$PI3" "$PI4" "$PI1")
LINK_NAMES=("Pi1→Pi2" "Pi2→Pi3" "Pi3→Pi4" "Pi4→Pi1")

sweep_link() {
    local idx=$1
    local sender=${LINK_SENDER[$idx]}
    local recver=${LINK_RECVER[$idx]}
    local name=${LINK_NAMES[$idx]}
    local log="/tmp/sweep_${name}.log"

    echo ""
    echo "=== Link $((idx+1)): $name  (sender=$sender  reader=$recver) ==="

    # Flash reader first (blocks waiting for pin changes)
    $INSTALL "$recver" "$READER" > "$log" 2>&1 &
    local rpid=$!
    sleep 4

    # Flash driver (starts toggling pins after 2s delay)
    $INSTALL "$sender" "$DRIVER" > /dev/null 2>&1 &
    local dpid=$!

    # Wait for both to finish (reader runs 25s, driver ~13s)
    wait $rpid 2>/dev/null
    wait $dpid 2>/dev/null

    # Show transitions (skip the t=0 floating-pin reading)
    local transitions=$(grep -E "^t=" "$log" | grep -v "^t=   0:")
    if [ -n "$transitions" ]; then
        echo "$transitions" | sed 's/^/  /'
    else
        echo "  (no transitions seen)"
    fi

    # Parse: check which pins toggled (appeared in any non-t=0 transition line)
    local seen=0
    local missing=""
    for pin in "D0" "D1" "D2" "D3" "D4" "D5" "D6" "D7" "CLK" "ACK"; do
        if echo "$transitions" | grep -qE "$pin\("; then
            seen=$((seen + 1))
        else
            missing="$missing $pin"
        fi
    done

    if [ "$seen" -ge 10 ]; then
        echo "  ✓ all 10 pins"
    else
        echo "  MISSING $((10 - seen))/10:$missing"
    fi

    # Let Pis fully reboot before next test (needs >5s for reliable re-flash)
    sleep 10
}

if [ -n "$1" ]; then
    sweep_link $(($1 - 1))
else
    for i in 0 1 2 3; do
        sweep_link $i
    done
fi
echo ""
echo "=== DONE ==="
