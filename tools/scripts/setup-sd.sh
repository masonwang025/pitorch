#!/bin/bash
# Set up the SD card for PiTorch.
#
# Single-Pi mode (default):
#   Downloads weights + tokenizer, creates combined binary, copies to SD.
#   Usage: ./setup-sd.sh [model]    (default: stories15M)
#     e.g. ./setup-sd.sh 15M
#
# Distributed mode (head rank tokenizer):
#   Appends tokenizer to a shard file for distributed text generation.
#   Usage: ./setup-sd.sh --shard <model> <rank>
#     e.g. ./setup-sd.sh --shard 42M 3
#     Only rank 3 (head rank) gets the tokenizer appended.
#     Other ranks just get the plain shard copied.
set -euo pipefail

PITORCH_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEIGHTS_DIR="$PITORCH_ROOT/weights"
HOST_DIR="$PITORCH_ROOT/tools/host"
SHARD_MODE=0

if [ "${1:-}" = "--shard" ]; then
    SHARD_MODE=1
    shift
    MODEL="${1:?usage: setup-sd.sh --shard <model> <rank>}"
    SHARD_RANK="${2:?usage: setup-sd.sh --shard <model> <rank>}"
    shift 2
else
    MODEL="${1:-15M}"
fi

BIN_FILE="stories${MODEL}.bin"
COMBINED_FILE="stories${MODEL}_full.bin"

# ── find SD card ──
SD=""
for v in /Volumes/boot /Volumes/NO\ NAME /Volumes/BOOT /Volumes/PI /Volumes/PITORCH; do
    if [ -d "$v" ]; then
        SD="$v"
        break
    fi
done

# fallback: any non-system volume that has kernel.img
if [ -z "$SD" ]; then
    for v in /Volumes/*/; do
        v="${v%/}"
        [ "$v" = "/Volumes/Macintosh HD" ] && continue
        if [ -f "$v/kernel.img" ]; then
            SD="$v"
            break
        fi
    done
fi

if [ -z "$SD" ]; then
    echo "ERROR: no SD card found. Mount it and try again." >&2
    echo "  looked in /Volumes/ for a partition with kernel.img" >&2
    exit 1
fi

echo "SD card: $SD"
echo ""

# ── check kernel.img ──
if [ ! -f "$SD/kernel.img" ]; then
    echo "WARNING: no kernel.img on SD card — bootloader not installed?"
fi

# ── check/install firmware files ──
SDCARD_DIR="$PITORCH_ROOT/tools/sdcard"
for f in bootcode.bin start.elf kernel.img; do
    if [ ! -f "$SD/$f" ]; then
        if [ -f "$SDCARD_DIR/$f" ]; then
            echo "installing $f from tools/sdcard/..."
            cp "$SDCARD_DIR/$f" "$SD/$f"
        else
            echo "WARNING: $f not found on SD card or in tools/sdcard/"
        fi
    fi
done

# install matched fixup files (fixup.dat MUST accompany start.elf)
for f in fixup.dat fixup_cd.dat fixup_db.dat fixup_x.dat \
         start_cd.elf start_db.elf start_x.elf; do
    if [ -f "$SDCARD_DIR/$f" ] && [ ! -f "$SD/$f" ]; then
        echo "installing $f from tools/sdcard/..."
        cp "$SDCARD_DIR/$f" "$SD/$f"
    fi
done

if [ ! -f "$SD/fixup.dat" ]; then
    echo "ERROR: fixup.dat is missing from SD card and tools/sdcard/." >&2
    echo "  Without fixup.dat, start.elf cannot relocate itself and the" >&2
    echo "  ARM/GPU memory split will be wrong (e.g. 128/128 instead of 448/64)." >&2
    echo "" >&2
    echo "  To fix: identify your start.elf version and get the matching fixup.dat:" >&2
    echo "    strings \"\$SD/start.elf\" | grep VC_BUILD" >&2
    echo "    # then fetch the matching fixup.dat from raspberrypi/firmware" >&2
    exit 1
fi

# ── ensure tokenizer ──
ensure_tokenizer() {
    if [ ! -f "$WEIGHTS_DIR/tokenizer.model" ]; then
        echo "downloading tokenizer.model..."
        (cd "$WEIGHTS_DIR" && ./download.sh tokenizer)
    fi
    if [ ! -f "$WEIGHTS_DIR/tokenizer.bin" ]; then
        echo "exporting tokenizer.bin..."
        python3 "$HOST_DIR/export_tokenizer.py" \
            --tokenizer-model "$WEIGHTS_DIR/tokenizer.model" \
            --output "$WEIGHTS_DIR/tokenizer.bin"
    fi
}

# ── fix config.txt ──
update_config() {
    local initramfs_path="$1"
    CONFIG="$SD/config.txt"
    if [ ! -f "$CONFIG" ]; then
        echo "creating config.txt..."
        touch "$CONFIG"
    fi
    if grep -q "^initramfs" "$CONFIG" 2>/dev/null; then
        echo "updating initramfs line in config.txt..."
        sed -i '' '/^initramfs/d' "$CONFIG"
    fi
    echo "initramfs $initramfs_path 0x2000000" >> "$CONFIG"
    echo ""
    echo "config.txt:"
    cat "$CONFIG"
}

if [ "$SHARD_MODE" -eq 1 ]; then
    # ═══════════════════════════════════════════════
    # Shard mode: set up one rank's SD card
    # ═══════════════════════════════════════════════
    SHARD_DIR="$WEIGHTS_DIR/shards/$MODEL"
    SHARD_FILE="$SHARD_DIR/rank${SHARD_RANK}.bin"

    if [ ! -f "$SHARD_FILE" ]; then
        echo "ERROR: shard not found: $SHARD_FILE" >&2
        echo "  Generate shards with: cd tools/host && make test_shard_train" >&2
        exit 1
    fi

    SD_SHARD_DIR="$SD/weights/shards/$MODEL"
    mkdir -p "$SD_SHARD_DIR"

    if [ "$SHARD_RANK" = "3" ]; then
        # Head rank: append tokenizer for distributed text generation
        ensure_tokenizer
        FULL_FILE="$SD_SHARD_DIR/rank3_full.bin"
        echo ""
        echo "creating rank3_full.bin (shard + tokenizer)..."
        cat "$SHARD_FILE" "$WEIGHTS_DIR/tokenizer.bin" > "$FULL_FILE"
        echo "  shard:     $(du -h "$SHARD_FILE" | cut -f1)"
        echo "  tokenizer: $(du -h "$WEIGHTS_DIR/tokenizer.bin" | cut -f1)"
        echo "  combined:  $(du -h "$FULL_FILE" | cut -f1)"

        update_config "weights/shards/$MODEL/rank3_full.bin"
        echo ""
        echo "DONE. Head rank (rank 3) set up with tokenizer for distributed generation."
    else
        echo ""
        echo "copying rank${SHARD_RANK}.bin to SD card..."
        cp "$SHARD_FILE" "$SD_SHARD_DIR/rank${SHARD_RANK}.bin"
        echo "  shard: $(du -h "$SHARD_FILE" | cut -f1)"

        update_config "weights/shards/$MODEL/rank${SHARD_RANK}.bin"
        echo ""
        echo "DONE. Rank $SHARD_RANK set up."
    fi

else
    # ═══════════════════════════════════════════════
    # Single-Pi mode: combined weights + tokenizer
    # ═══════════════════════════════════════════════
    if [ ! -f "$WEIGHTS_DIR/$BIN_FILE" ]; then
        echo "downloading $BIN_FILE..."
        (cd "$WEIGHTS_DIR" && ./download.sh "$MODEL")
    fi

    ensure_tokenizer

    echo ""
    echo "creating $COMBINED_FILE..."
    cat "$WEIGHTS_DIR/$BIN_FILE" "$WEIGHTS_DIR/tokenizer.bin" \
        > "$WEIGHTS_DIR/$COMBINED_FILE"
    echo "  model:     $(du -h "$WEIGHTS_DIR/$BIN_FILE" | cut -f1)"
    echo "  tokenizer: $(du -h "$WEIGHTS_DIR/tokenizer.bin" | cut -f1)"
    echo "  combined:  $(du -h "$WEIGHTS_DIR/$COMBINED_FILE" | cut -f1)"

    mkdir -p "$SD/weights"
    echo ""
    echo "copying to SD card..."
    cp "$WEIGHTS_DIR/$COMBINED_FILE" "$SD/weights/$COMBINED_FILE"

    echo ""
    ls -lh "$SD/weights/"

    update_config "weights/$COMBINED_FILE"
    echo ""
    echo "DONE. Eject SD card, put it in the Pi, and run:"
    echo "  cd demo && ./run.sh generate"
fi
