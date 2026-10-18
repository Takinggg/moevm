"""Tests for scripts/bench.py that need no model and no GPU.

Run from the repository root (macOS, because safe_run.py uses macOS tools):

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench  # noqa: E402  (needs scripts/ on sys.path)

EVIDENCE = ROOT / "runs/evidence"
OFF_LOG = EVIDENCE / "qwen3_next_q4_p0d_parity_stock_ignoreeos_n1024_v2.log"
ON_LOG = EVIDENCE / "qwen3_next_q4_p0d_parity_moevm_cap128_ignoreeos_n1024.log"

# Stands in for llama-moe-trace: prints a recorded log and writes a tiny logits file.
FAKE_LLAMA = f"""#!{sys.executable}
import struct, sys
args = sys.argv[1:]
log = {str(ON_LOG)!r} if "--moevm-sidecar" in args else {str(OFF_LOG)!r}
lines = open(log, errors="replace").read().splitlines()
print("\\n".join(lines[1:]))
with open(args[args.index("--logits-out") + 1], "wb") as f:
    f.write(struct.pack("<8sIIII", b"MOEVMLG1", 1, 3, 1, 0))
    f.write(struct.pack("<Ii", 0, 2) + struct.pack("<3f", 0.5, -1.0, 2.0))
"""


def write_logits(path: Path, rows: list[list[float]]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<8sIIII", b"MOEVMLG1", 1, len(rows[0]), len(rows), 0))
        for step, row in enumerate(rows):
            f.write(struct.pack("<Ii", step, 0) + struct.pack(f"<{len(row)}f", *row))


class ParityTest(unittest.TestCase):
    def test_identical_files_are_bit_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, "a.logits"), Path(tmp, "b.logits")
            write_logits(a, [[1.0, 0.5, -2.0], [0.0, 3.0, 1.0]])
            write_logits(b, [[1.0, 0.5, -2.0], [0.0, 3.0, 1.0]])
            result = bench.logits_parity(a, b)
        self.assertTrue(result["bit_identical"])
        self.assertEqual((result["rows"], result["n_vocab"], result["max_abs"]), (2, 3, 0.0))

    def test_different_logits_report_max_abs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, "a.logits"), Path(tmp, "b.logits")
            write_logits(a, [[1.0, 0.5, -2.0], [0.0, 3.0, 1.0]])
            write_logits(b, [[1.0, 0.5, -2.0], [0.0, 3.0, 1.25]])
            result = bench.logits_parity(a, b)
        self.assertFalse(result["bit_identical"])
        self.assertAlmostEqual(result["max_abs"], 0.25)
        self.assertEqual(result["argmax_mismatch_count"], 0)


class EndToEndTest(unittest.TestCase):
    def test_qwen_pipeline_with_fake_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_dir = tmp / "models/qwen3-coder-next-q4"
            model_dir.mkdir(parents=True)
            for name in ("Qwen3-Coder-Next-Q4_K_M.gguf", "qwen3-next-q4.moevm",
                         "qwen3-next-q4.moevm.idx", "qwen3-next-q4.moevm.cidx"):
                (model_dir / name).write_bytes(b"placeholder")
            fake = tmp / "fake-llama-moe-trace"
            fake.write_text(FAKE_LLAMA)
            fake.chmod(0o755)
            env = {**os.environ, "MOEVM_MODELS_DIR": str(tmp / "models"), "MOEVM_LLAMA_BIN": str(fake),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            out = tmp / "out"
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/bench.py"), "qwen3-coder-next-q4",
                                   "--runs", "2", "--out-dir", str(out), "--allow-docker"],
                                  env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads((out / "summary.json").read_text())
        summary = report["summary"]
        self.assertEqual((summary["off"]["median"], summary["on"]["median"]), (6.41, 19.9))
        self.assertAlmostEqual(summary["speedup"], 19.9 / 6.41)
        self.assertEqual(len(report["parity"]), 2)
        self.assertTrue(all(p["bit_identical"] for p in report["parity"]))
        self.assertIn("Speedup (median on / median off): 3.10x", proc.stdout)


if __name__ == "__main__":
    unittest.main()
