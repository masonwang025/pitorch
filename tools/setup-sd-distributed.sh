#!/bin/bash
#
# Set up an SD card for distributed pipeline-parallel training.
# Copies the weight shard for the given rank and configures initramfs.
#
# Usage: ./tools/setup-sd-distributed.sh <rank> <volume_name> <model>
#   rank:   0-3
#   volume: PIE0, PIE1, PIE2, PIE3
#   model:  42M (looks for weights/shards/42M/rank<N>.bin)
#
# Example:
#   ./tools/setup-sd-distributed.sh 0 PIE0 42M
#   ./tools/setup-sd-distributed.sh 3 PIE3 42M
#
set -euo pipefail

PITORCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <rank> <volume_name> <model>"
    echo "  rank:   0-3"
    echo "  volume: PIE0..PIE3"
    echo "  model:  42M, 110M, etc."
    exit 1
fi

RANK="$1"
VOLUME="$2"
MODEL="$3"
MOUNT="/Volumes/$VOLUME"

if [ ! -d "$MOUNT" ]; then
    echo "ERROR: $MOUNT not mounted" >&2
    exit 1
fi

SHARD_DIR="$PITORCH_ROOT/weights/shards/$MODEL"
SHARD_FILE="$SHARD_DIR/rank${RANK}.bin"

if [ ! -f "$SHARD_FILE" ]; then
    echo "ERROR: shard not found: $SHARD_FILE" >&2
    echo "Run: python3 tools/shard_weights.py weights/stories${MODEL}.bin 4 $SHARD_DIR" >&2
    exit 1
fi

SHARD_SIZE=$(du -h "$SHARD_FILE" | cut -f1)
echo "Setting up $VOLUME for rank $RANK ($MODEL, $SHARD_SIZE shard)"

# Create weights directory
mkdir -p "$MOUNT/weights/shards/${MODEL}"

# Copy shard
echo "  copying rank${RANK}.bin ($SHARD_SIZE)..."
cp "$SHARD_FILE" "$MOUNT/weights/shards/${MODEL}/rank${RANK}.bin"

# Update config.txt
CONFIG="$MOUNT/config.txt"
SHARD_PATH="weights/shards/${MODEL}/rank${RANK}.bin"

if [ -f "$CONFIG" ]; then
    # Update gpu_mem
    if grep -q "^gpu_mem=" "$CONFIG"; then
        sed -i '' "s/^gpu_mem=.*/gpu_mem=32/" "$CONFIG"
    else
        echo "gpu_mem=32" >> "$CONFIG"
    fi
    # Update initramfs
    if grep -q "^initramfs" "$CONFIG"; then
        sed -i '' "s|^initramfs.*|initramfs $SHARD_PATH 0x2000000|" "$CONFIG"
    else
        echo "initramfs $SHARD_PATH 0x2000000" >> "$CONFIG"
    fi
else
    echo "gpu_mem=32" > "$CONFIG"
    echo "initramfs $SHARD_PATH 0x2000000" >> "$CONFIG"
fi

echo "  config.txt updated:"
grep -E "^(gpu_mem|initramfs)" "$CONFIG" | sed 's/^/    /'

# Verify
echo "  verifying..."
DEST_SIZE=$(wc -c < "$MOUNT/weights/shards/${MODEL}/rank${RANK}.bin" | tr -d ' ')
SRC_SIZE=$(wc -c < "$SHARD_FILE" | tr -d ' ')
if [ "$DEST_SIZE" = "$SRC_SIZE" ]; then
    echo "  OK: $DEST_SIZE bytes"
else
    echo "  ERROR: size mismatch (src=$SRC_SIZE dest=$DEST_SIZE)" >&2
    exit 1
fi

echo "Done. Eject $VOLUME and insert into Pi $RANK."
