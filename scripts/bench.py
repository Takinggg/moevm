#!/usr/bin/env python3
"""Reproduce the MoEVM headline benchmarks on an Apple Silicon Mac.

    python3 scripts/bench.py qwen3-coder-next-q4 [--runs 3]
    python3 scripts/bench.py gpt-oss-120b [--runs 5]

qwen3-coder-next-q4: llama.cpp --op-offload with MoEVM off vs on, identical
prompt and flags, runs alternated. The logits of every off/on pair are
compared. Prints a summary table and writes summary.json/summary.md.

gpt-oss-120b: MoEVM graph pool at 4096 context, median of N, through the
phase0/gptoss_soak.py harness that produced the published number.

Models are not downloaded automatically; the script prints the command.
A missing sidecar is packed first (it needs about as much disk as the model).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PHASE0 = Path(__file__).resolve().parents[1] / "phase0"
sys.path.insert(0, str(PHASE0))

from compare_decode_logs import parse_log  # noqa: E402  (needs PHASE0 on sys.path)
from compare_logits import compare, read_header  # noqa: E402
from moevm_paths import LLAMA_BIN, MODELS_DIR, RUNS, SAFE_RUN  # noqa: E402


@dataclass(frozen=True)
class Preset:
    model_dir: str
    shards: tuple[str, ...]
    sidecar_stem: str
    download: str
    published: dict


@dataclass(frozen=True)
class Setup:
    paths: dict
    work: Path
    safe_args: tuple  # extra safe_run.py flags chosen on the command line


PRESETS = {
    "qwen3-coder-next-q4": Preset(
        model_dir="qwen3-coder-next-q4",
        shards=("Qwen3-Coder-Next-Q4_K_M.gguf",),
        sidecar_stem="qwen3-next-q4",
        download='hf download unsloth/Qwen3-Coder-Next-GGUF Qwen3-Coder-Next-Q4_K_M.gguf --local-dir "{dir}"',
        published={"off": 6.41, "on": 19.90},
    ),
    "gpt-oss-120b": Preset(
        model_dir="gpt-oss-120b/Q4_K_M",
        shards=("gpt-oss-120b-Q4_K_M-00001-of-00002.gguf", "gpt-oss-120b-Q4_K_M-00002-of-00002.gguf"),
        sidecar_stem="gpt-oss-120b-q4",
        download='hf download unsloth/gpt-oss-120b-GGUF --include "Q4_K_M/*" --local-dir "{parent}"',
        published={"on": 8.72},
    ),
}

QWEN_PROMPT = "Write a Python function that parses log lines into records."
QWEN_COMMON = [
    "-ngl", "99", "-ot", ".*ffn_.*_exps.weight=CPU", "--op-offload", "--no-repack",
    "-c", "1536", "-b", "1", "-ub", "1", "-n", "1024", "--ignore-eos", "--temp", "0", "-s", "1",
    "-p", QWEN_PROMPT,
]
QWEN_MOEVM = [
    "--moevm-arena-gib", "1", "--moevm-inflight-mib", "64", "--moevm-threads", "4",
    "--moevm-prefetch-topk", "10", "--moevm-copy-mode", "memcpy", "--moevm-cache",
    "--moevm-pool-layer", "all", "--moevm-pool-capacity", "128",
]
RUNTIME_ERRORS_RE = re.compile(r"runtime_errors=(\d+)")
GIB = 1024**3


def model_paths(preset: Preset) -> dict:
    model_dir = MODELS_DIR / preset.model_dir
    sidecar = model_dir / f"{preset.sidecar_stem}.moevm"
    return {
        "dir": model_dir,
        "shards": [model_dir / name for name in preset.shards],
        "sidecar": sidecar,
        "index": sidecar.with_name(sidecar.name + ".idx"),
        "component_index": sidecar.with_name(sidecar.name + ".cidx"),
    }


def check_environment(preset: Preset, paths: dict) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.exit(f"error: needs an Apple Silicon Mac, found {platform.system()} {platform.machine()}")
    ram_gib = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout) / GIB
    if ram_gib < 47:
        print(f"warning: {ram_gib:.0f} GiB of RAM; the published runs used a 48 GB Mac and may not fit here")
    if not LLAMA_BIN.exists():
        sys.exit(f"error: {LLAMA_BIN} not found; run scripts/build.sh first (or set MOEVM_LLAMA_BIN)")
    missing = [p for p in paths["shards"] if not p.exists()]
    if missing:
        print("error: model not found:", *[f"  {p}" for p in missing], sep="\n")
        print("download it with the Hugging Face CLI (pip install -U huggingface_hub), then rerun:")
        print("  " + preset.download.format(dir=paths["dir"], parent=paths["dir"].parent))
        print(f"or point MOEVM_MODELS_DIR at the directory that holds {preset.model_dir}/ (now {MODELS_DIR})")
        sys.exit(2)


def run_checked(cmd: list, what: str) -> None:
    print(f"[bench] {what}", flush=True)
    result = subprocess.run([str(part) for part in cmd])
    if result.returncode != 0:
        sys.exit(f"error: {what} failed with exit code {result.returncode}")


def ensure_sidecar(setup: Setup) -> None:
    paths, work = setup.paths, setup.work
    if all(paths[key].exists() for key in ("sidecar", "index", "component_index")):
        return
    need = sum(p.stat().st_size for p in paths["shards"]) + 5 * GIB
    free = shutil.disk_usage(paths["dir"]).free
    if free < need:
        sys.exit(f"error: packing the sidecar needs about {need / GIB:.0f} GiB free next to the model, {free / GIB:.0f} GiB available")
    plan = work / "sidecar_plan.json"
    py = sys.executable
    run_checked([py, PHASE0 / "sidecar_plan.py", "--out", plan, *paths["shards"]], "planning sidecar")
    run_checked([py, SAFE_RUN, "--log", work / "sidecar_pack.log", "--summary", work / "sidecar_pack.safe_summary",
                 "--max-seconds", "3600", *setup.safe_args, "--", py, PHASE0 / "sidecar_pack.py", "--plan", plan,
                 "--out", paths["sidecar"], "--summary", work / "sidecar_pack.json", "--nocache"],
                f"packing sidecar into {paths['sidecar']} (about the size of the model)")
    for script, key in (("sidecar_index.py", "index"), ("sidecar_component_index.py", "component_index")):
        run_checked([py, PHASE0 / script, "--plan", plan, "--sidecar", paths["sidecar"], "--out", paths[key]],
                    f"building {paths[key].name}")


def run_decode(setup: Setup, extra: list, log: Path, logits: Path) -> dict:
    cmd = [sys.executable, SAFE_RUN, "--log", log, "--summary", log.with_suffix(".safe_summary"),
           "--max-seconds", "900", *setup.safe_args, "--", "/usr/bin/env", "GGML_OP_OFFLOAD_MIN_BATCH=1", LLAMA_BIN,
           "-m", setup.paths["shards"][0], "--logits-out", logits, *extra, *QWEN_COMMON]
    run_checked(cmd, f"decoding -> {log.name}")
    parsed = parse_log(log)
    decoded = parsed.get("decoded")
    if not decoded or decoded["tokens"] != decoded["target"]:
        sys.exit(f"error: {log} did not decode the full 1024 tokens")
    errors = [int(n) for n in RUNTIME_ERRORS_RE.findall(log.read_text(errors="replace"))]
    return {"log": str(log), "tok_s": decoded["tok_s"], "tokens": decoded["tokens"],
            "runtime_errors": max(errors, default=0),
            "demand_reads": parsed.get("runtime", {}).get("demand_reads")}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def logits_parity(base: Path, cand: Path) -> dict:
    with base.open("rb") as f:
        header = read_header(f)
    if file_digest(base) == file_digest(cand):
        return {"bit_identical": True, "rows": header["rows"], "n_vocab": header["n_vocab"], "max_abs": 0.0,
                "token_mismatch_count": 0, "argmax_mismatch_count": 0}
    print("[bench] logits differ; computing the full comparison (slow)", flush=True)
    result = compare(base, cand)
    return {"bit_identical": False, "rows": result["rows"], "n_vocab": result["n_vocab"],
            "max_abs": result["max_abs"], "token_mismatch_count": result["token_mismatch_count"],
            "argmax_mismatch_count": result["argmax_mismatch_count"]}


def stats(values: list[float]) -> dict:
    return {"runs": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def bench_qwen(preset: Preset, setup: Setup, runs: int) -> dict:
    paths, work = setup.paths, setup.work
    moevm = ["--moevm-index", paths["index"], "--moevm-component-index", paths["component_index"],
             "--moevm-sidecar", paths["sidecar"], *QWEN_MOEVM]
    results = {"off": [], "on": []}
    parity = []
    for i in range(runs):
        for mode, extra in (("off", []), ("on", moevm)):
            log = work / f"qwen_{mode}_{i:02d}.log"
            results[mode].append(run_decode(setup, extra, log, work / f"qwen_{mode}.logits"))
        parity.append(logits_parity(work / "qwen_off.logits", work / "qwen_on.logits"))
    summary = {mode: {**stats([r["tok_s"] for r in rows]), "published": preset.published[mode]}
               for mode, rows in results.items()}
    summary["speedup"] = summary["on"]["median"] / summary["off"]["median"]
    summary["published_speedup"] = preset.published["on"] / preset.published["off"]
    summary["runtime_errors"] = sum(r["runtime_errors"] for r in results["on"])
    return {"summary": summary, "parity": parity, "runs": results}


def qwen_table(report: dict) -> list[str]:
    s, parity = report["summary"], report["parity"]
    lines = ["| Mode | Runs | Median tok/s | Min | Max | Published |", "|---|---:|---:|---:|---:|---:|"]
    for mode, label in (("off", "llama.cpp --op-offload, MoEVM off"), ("on", "MoEVM on")):
        m = s[mode]
        lines.append(f"| {label} | {m['runs']} | {m['median']:.2f} | {m['min']:.2f} | {m['max']:.2f} | {m['published']:.2f} |")
    identical = sum(p["bit_identical"] for p in parity)
    worst = max(p["max_abs"] for p in parity)
    lines += ["", f"Speedup (median on / median off): {s['speedup']:.2f}x (published {s['published_speedup']:.2f}x)",
              f"Logits parity: {identical}/{len(parity)} pairs bit-identical over {parity[0]['rows']} rows x "
              f"{parity[0]['n_vocab']} vocab; worst max_abs={worst}",
              f"MoEVM runtime_errors: {s['runtime_errors']}"]
    return lines


def bench_gptoss(setup: Setup, runs: int) -> dict:
    paths, out = setup.paths, setup.work / "gptoss_soak"
    if setup.safe_args:
        sys.exit("error: --allow-docker is not supported for gpt-oss-120b; stop Docker/VMs first")
    run_checked([sys.executable, PHASE0 / "gptoss_soak.py", "--iterations", runs, "--tokens", "128", "--ctx", "4096",
                 "--arena-gib", "24", "--trim-min-gib", "12", "--pool-capacity", "16", "--prefetch-topk", "4",
                 "--moevm-threads", "4", "--copy-mode", "memcpy", "--logits-every", "0",
                 "--prompt-names", "observability", "--out-dir", out, "--model", paths["shards"][0],
                 "--index", paths["index"], "--component-index", paths["component_index"],
                 "--sidecar", paths["sidecar"]], f"running {runs} gpt-oss iterations")
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != runs:
        sys.exit(f"error: expected {runs} results in {out / 'results.jsonl'}, found {len(rows)}")
    summary = {**stats([r["tok_s"] for r in rows]), "published": PRESETS["gpt-oss-120b"].published["on"],
               "demand_reads": sum(r["demand_reads"] for r in rows),
               "runtime_errors": sum(r["runtime_errors"] for r in rows),
               "pressure_critical_events": sum(r["pressure_critical_events"] for r in rows)}
    return {"summary": summary, "runs": rows}


def gptoss_table(report: dict) -> list[str]:
    s = report["summary"]
    return ["| Mode | Runs | Median tok/s | Min | Max | Published |", "|---|---:|---:|---:|---:|---:|",
            f"| MoEVM graph pool, c=4096 | {s['runs']} | {s['median']:.2f} | {s['min']:.2f} | {s['max']:.2f} | {s['published']:.2f} |",
            "", f"demand_reads={s['demand_reads']} runtime_errors={s['runtime_errors']} "
            f"pressure_critical_events={s['pressure_critical_events']} (all should be 0)"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--runs", type=int, help="runs per mode (default: 3 for qwen, 5 for gpt-oss)")
    parser.add_argument("--out-dir", type=Path, help="default: runs/bench_<preset>_<timestamp>")
    parser.add_argument("--allow-docker", action="store_true",
                        help="let safe_run.py proceed while Docker or a VM is running (they compete for memory)")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    runs = args.runs or (3 if args.preset.startswith("qwen") else 5)
    if runs < 1:
        parser.error("--runs must be at least 1")
    paths = model_paths(preset)
    check_environment(preset, paths)
    work = args.out_dir or RUNS / f"bench_{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}"
    work.mkdir(parents=True, exist_ok=True)
    setup = Setup(paths, work, ("--allow-docker",) if args.allow_docker else ())
    ensure_sidecar(setup)
    if args.preset.startswith("qwen"):
        report = bench_qwen(preset, setup, runs)
        table = qwen_table(report)
    else:
        report = bench_gptoss(setup, runs)
        table = gptoss_table(report)
    report["preset"] = args.preset
    report["llama_bin"] = str(LLAMA_BIN)
    (work / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (work / "summary.md").write_text(f"# MoEVM benchmark: {args.preset}\n\n" + "\n".join(table) + "\n")
    print("\n" + "\n".join(table))
    print(f"\nwritten: {work / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
