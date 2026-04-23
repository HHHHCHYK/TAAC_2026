#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

uv python install 3.13
uv sync --locked --python 3.13
uv run pytest tests/gpu/test_gpu_environment.py -q
