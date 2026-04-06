#!/bin/bash
set -euo pipefail
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
make -C "$TEST_DIR"
