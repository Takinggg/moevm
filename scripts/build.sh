#!/usr/bin/env bash
# Clone the MoEVM llama.cpp fork into external/llama.cpp and build llama-moe-trace.
#
# Environment overrides:
#   MOEVM_LLAMA_REPO  git URL or local path of the fork (default: Takinggg/llama.cpp on GitHub)
#   MOEVM_LLAMA_REF   branch or tag to build (default: codex/moevm-graph-build)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${MOEVM_LLAMA_REPO:-https://github.com/Takinggg/llama.cpp.git}"
REF="${MOEVM_LLAMA_REF:-codex/moevm-graph-build}"
DIR="$ROOT/external/llama.cpp"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "error: MoEVM targets Apple Silicon Macs (Darwin arm64); found $(uname -s) $(uname -m)" >&2
  exit 1
fi
for tool in git cmake c++; do
  command -v "$tool" >/dev/null || { echo "error: '$tool' not found (install Xcode command line tools and CMake)" >&2; exit 1; }
done

if [ -d "$DIR/.git" ]; then
  echo "using existing fork checkout: $DIR ($(git -C "$DIR" rev-parse --short HEAD))"
else
  git clone --branch "$REF" --single-branch "$REPO" "$DIR"
fi

# llama-moe-trace compiles this repository's phase0/moevm_runtime.cpp.
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DMOEVM_PHASE0_DIR="$ROOT/phase0"
cmake --build "$DIR/build" --target llama-moe-trace -j "$(sysctl -n hw.ncpu)"

BIN="$DIR/build/bin/llama-moe-trace"
[ -x "$BIN" ] || { echo "error: build finished but $BIN is missing" >&2; exit 1; }
echo "built $BIN"
