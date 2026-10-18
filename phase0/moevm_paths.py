"""Filesystem locations shared by the MoEVM benchmark harnesses.

Defaults follow the layout described in the top-level README: the llama.cpp
fork is built in ``external/llama.cpp`` and models live in ``~/models``.
Override either one with an environment variable instead of editing scripts:

- ``MOEVM_MODELS_DIR``: directory holding the GGUF models and packed sidecars.
- ``MOEVM_LLAMA_BIN``: path to the fork's ``llama-moe-trace`` binary.

An empty variable is treated as unset.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SAFE_RUN = ROOT / "phase0/safe_run.py"
MODELS_DIR = Path(os.environ.get("MOEVM_MODELS_DIR") or "~/models").expanduser()
LLAMA_BIN = Path(
    os.environ.get("MOEVM_LLAMA_BIN") or ROOT / "external/llama.cpp/build/bin/llama-moe-trace"
).expanduser()
