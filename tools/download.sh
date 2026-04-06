#!/bin/bash
# Download karpathy/tinyllamas checkpoints and tokenizer.
# Usage:
#   ./download.sh              # downloads stories15M.bin (default)
#   ./download.sh 260K         # downloads stories260K.bin
#   ./download.sh tokenizer    # downloads tokenizer.model
#   ./download.sh 15M 42M      # downloads both
#   ./download.sh all           # downloads all models + tokenizer
set -euo pipefail
cd "$(dirname "$0")"

MODELS_BASE="https://huggingface.co/karpathy/tinyllamas/resolve/main"
TOKENIZER_URL="https://github.com/karpathy/llama2.c/raw/master/tokenizer.model"

url_for() {
    case "$1" in
        260K) echo "$MODELS_BASE/stories260K/stories260K.bin" ;;
        *)    echo "$MODELS_BASE/stories${1}.bin" ;;
    esac
}

download_model() {
    local file="stories${1}.bin"
    if [ -f "$file" ]; then
        echo "$file already exists ($(du -h "$file" | cut -f1))"
    else
        echo "downloading $file..."
        curl -L -o "$file" "$(url_for "$1")"
        echo "  -> $(du -h "$file" | cut -f1)"
    fi
}

download_tokenizer() {
    if [ -f "tokenizer.model" ]; then
        echo "tokenizer.model already exists ($(du -h "tokenizer.model" | cut -f1))"
    else
        echo "downloading tokenizer.model..."
        curl -L -o "tokenizer.model" "$TOKENIZER_URL"
        echo "  -> $(du -h "tokenizer.model" | cut -f1)"
    fi
}

if [ $# -eq 0 ]; then
    set -- 15M
fi

if [ "$1" = "all" ]; then
    set -- 260K 15M 42M 110M tokenizer
fi

for t in "$@"; do
    case "$t" in
        260K|15M|42M|110M) download_model "$t" ;;
        tokenizer) download_tokenizer ;;
        *) echo "unknown: $t (available: 260K 15M 42M 110M tokenizer)" >&2; exit 1 ;;
    esac
done
