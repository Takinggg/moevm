#!/usr/bin/env python3
"""Run risky MoEVM commands behind conservative macOS safety guards.

This wrapper cannot protect against every kernel/Metal driver failure. Its job
is to refuse known-dangerous command shapes, stop before severe memory pressure,
and preserve compact logs for post-mortem analysis.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable


METAL_OOM_PATTERNS = [
    "kIOGPUCommandBufferCallbackErrorOutOfMemory",
    "Insufficient Memory",
    "ggml_metal_graph_compute: backend is in error state",
    "llama_decode: failed to decode",
    "Compute error",
]

MODEL_IO_SCRIPTS = {
    "gguf_inspect.py",
    "sidecar_plan.py",
    "sidecar_index.py",
    "sidecar_component_index.py",
    "sidecar_pack.py",
}

MODEL_IO_EXECUTABLES = {
    "llama-moe-trace",
    "llama-bench",
    "pager_replay",
    "runtime_replay",
    "runtime_smoke",
}

MODEL_IO_LOCK = Path("/tmp/moevm_model_io.lock")


def run_text(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception:
        return ""


def parse_memory_pressure() -> dict[str, float]:
    out = run_text(["memory_pressure"], timeout=5.0)
    result: dict[str, float] = {}
    m = re.search(r"System-wide memory free percentage:\s+(\d+)%", out)
    if m:
        result["free_pct"] = float(m.group(1))
    total_pages = None
    free_pages = inactive_pages = speculative_pages = purgeable_pages = None
    m = re.search(r"\((\d+) pages with a page size of (\d+)\)", out)
    if m:
        total_pages = float(m.group(1))
        result["page_size"] = float(m.group(2))
    for key, field in [
        ("Pages free", "free_pages"),
        ("Pages inactive", "inactive_pages"),
        ("Pages speculative", "speculative_pages"),
        ("Pages purgeable", "purgeable_pages"),
    ]:
        m = re.search(rf"{re.escape(key)}:\s+(\d+)", out)
        if m:
            value = float(m.group(1))
            result[field] = value
            if field == "free_pages":
                free_pages = value
            elif field == "inactive_pages":
                inactive_pages = value
            elif field == "speculative_pages":
                speculative_pages = value
            elif field == "purgeable_pages":
                purgeable_pages = value
    if "free_pct" not in result and total_pages:
        available_pages = sum(x or 0.0 for x in [free_pages, inactive_pages, speculative_pages, purgeable_pages])
        result["free_pct"] = 100.0 * available_pages / total_pages
    m = re.search(r"Swapouts:\s+(\d+)", out)
    if m:
        result["swapouts"] = float(m.group(1))
    m = re.search(r"Pages wired down:\s+(\d+)", out)
    if m:
        result["wired_pages"] = float(m.group(1))
        result["wired_gib"] = float(m.group(1)) * 16384 / 1024**3
    return result


def process_group_rss_mib(pgid: int) -> float:
    out = run_text(["ps", "-axo", "pgid=,rss="], timeout=5.0)
    total_kib = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            row_pgid = int(parts[0])
            rss = int(parts[1])
        except ValueError:
            continue
        if row_pgid == pgid:
            total_kib += rss
    return total_kib / 1024


def docker_vm_running() -> bool:
    out = run_text(["ps", "-axo", "pid=,comm=,args="], timeout=5.0)
    for line in out.splitlines():
        if "com.apple.Virtualization.VirtualMachine" in line:
            return True
        if "/com.docker.virtualization" in line:
            return True
    return False


def command_tokens(command: str | list[str]) -> list[str]:
    if isinstance(command, list):
        return command
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def command_is_model_io(command: str | list[str], *, include_shell: bool = True) -> bool:
    tokens = command_tokens(command)
    if not tokens:
        return False
    exe = Path(tokens[0]).name
    if exe in MODEL_IO_EXECUTABLES or exe in MODEL_IO_SCRIPTS:
        return True
    if exe.startswith("python"):
        return any(Path(token).name in MODEL_IO_SCRIPTS for token in tokens[1:])
    if include_shell and exe in {"sh", "bash", "zsh"}:
        return any(
            Path(token).name in MODEL_IO_EXECUTABLES or Path(token).name in MODEL_IO_SCRIPTS
            for token in tokens[1:]
            if not token.endswith(".cpp")
        )
    return False


def active_model_io_jobs() -> list[str]:
    self_pid = os.getpid()
    self_pgid = os.getpgrp()
    out = run_text(["ps", "-axo", "pid=,pgid=,command="], timeout=5.0)
    jobs: list[str] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if pid == self_pid or pgid == self_pgid:
            continue
        if command_is_model_io(command, include_shell=False):
            jobs.append(f"{pid}: {command[:180]}")
    return jobs


def acquire_model_io_lock(cmd: list[str]) -> tuple[int | None, str | None]:
    if not command_is_model_io(cmd):
        return None, None
    fd = os.open(MODEL_IO_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            existing = os.read(fd, 4096).decode("utf-8", errors="replace").strip()
        except OSError:
            existing = ""
        os.close(fd)
        detail = f": {existing}" if existing else ""
        return None, f"another safe_run model I/O job holds {MODEL_IO_LOCK}{detail}"
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()} cmd={' '.join(cmd[:12])}\n".encode("utf-8", errors="replace"))
    return fd, None


def command_has_flag(argv: list[str], names: Iterable[str]) -> bool:
    flags = set(names)
    return any(arg in flags or any(arg.startswith(f"{name}=") for name in flags) for arg in argv)


def flag_int(argv: list[str], names: Iterable[str]) -> int | None:
    names = set(names)
    for i, arg in enumerate(argv):
        if arg in names and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return None
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                try:
                    return int(arg[len(prefix) :])
                except ValueError:
                    return None
    return None


def flag_float(argv: list[str], names: Iterable[str]) -> float | None:
    names = set(names)
    for i, arg in enumerate(argv):
        if arg in names and i + 1 < len(argv):
            try:
                return float(argv[i + 1])
            except ValueError:
                return None
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                try:
                    return float(arg[len(prefix) :])
                except ValueError:
                    return None
    return None


def flag_str(argv: list[str], names: Iterable[str]) -> str | None:
    names = set(names)
    for i, arg in enumerate(argv):
        if arg in names and i + 1 < len(argv):
            return argv[i + 1]
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                return arg[len(prefix) :]
    return None


def validate_gptoss_metal_gate(
    cmd: list[str],
    flash_on: bool,
    max_predict: int,
    max_arena_gib: float,
    require_pressure_trim_over_gib: float | None = None,
    allow_carved_pool: bool = False,
) -> list[str]:
    errors: list[str] = []
    n_predict = flag_int(cmd, ["-n", "--predict", "--n-predict"])
    arena_gib = flag_float(cmd, ["--moevm-arena-gib"])
    graph_mode = flag_str(cmd, ["--moevm-graph-mode"])
    pool_capacity = flag_int(cmd, ["--moevm-pool-capacity"])
    override = flag_str(cmd, ["-ot", "--override-tensor"]) or ""
    uses_moevm = any(arg == "--moevm-sidecar" or arg.startswith("--moevm-sidecar=") for arg in cmd)

    if flash_on:
        errors.append("gpt-oss Metal smoke requires FlashAttention off")
    if n_predict is None or n_predict < 0 or n_predict > max_predict:
        errors.append(f"gpt-oss Metal gate requires -n/--predict between 0 and {max_predict}")
    if uses_moevm:
        if arena_gib is None or arena_gib <= 0 or arena_gib > max_arena_gib:
            errors.append(f"gpt-oss Metal MoEVM gate requires --moevm-arena-gib <= {max_arena_gib:g}")
        if (
            arena_gib is not None
            and require_pressure_trim_over_gib is not None
            and arena_gib > require_pressure_trim_over_gib
            and not command_has_flag(cmd, ["--moevm-pressure-trim"])
        ):
            errors.append(
                "gpt-oss Metal scale gate requires --moevm-pressure-trim when --moevm-arena-gib exceeds "
                f"{require_pressure_trim_over_gib:g}"
            )
        if graph_mode != "pool":
            errors.append("gpt-oss Metal MoEVM gate requires --moevm-graph-mode pool")
        max_pool_capacity = 16
        if allow_carved_pool and arena_gib is not None and arena_gib <= 16:
            max_pool_capacity = 32
        elif allow_carved_pool and arena_gib is not None and arena_gib <= 20:
            max_pool_capacity = 24
        if pool_capacity is not None and pool_capacity > max_pool_capacity:
            errors.append(
                "gpt-oss Metal MoEVM gate requires --moevm-pool-capacity <= "
                f"{max_pool_capacity} for --moevm-arena-gib {arena_gib:g}"
            )
    else:
        if "exps.weight=CPU" not in override:
            errors.append("gpt-oss Metal baseline gate requires expert weight CPU override")
        if not command_has_flag(cmd, ["--op-offload", "--no-op-offload"]):
            errors.append("gpt-oss Metal baseline gate requires explicit --op-offload or --no-op-offload")
        if not command_has_flag(cmd, ["--no-repack", "-nr"]):
            errors.append("gpt-oss Metal baseline gate requires --no-repack")
    return errors


def preflight(args: argparse.Namespace, cmd: list[str]) -> list[str]:
    errors: list[str] = []
    joined = " ".join(cmd)
    stats = parse_memory_pressure()

    if stats.get("free_pct", 100.0) < args.preflight_min_free_pct:
        errors.append(f"free memory is {stats.get('free_pct')}%, below preflight minimum {args.preflight_min_free_pct}%")
    if "free_pct" not in stats:
        errors.append("memory_pressure did not return a usable free memory percentage")

    if not args.allow_docker and docker_vm_running():
        errors.append("Docker/Virtualization VM is running; stop Docker Desktop or pass --allow-docker")

    if not args.allow_concurrent_model_io:
        jobs = active_model_io_jobs()
        if jobs:
            errors.append("another model I/O job is already active; pass --allow-concurrent-model-io only for tiny tests: " + " | ".join(jobs[:3]))

    if "sidecar_pack.py" in joined and "--nocache" not in cmd:
        errors.append("sidecar_pack.py must run with --nocache to avoid filling the unified memory file cache")

    model_is_gptoss_120b = "gpt-oss-120b" in joined or "gptoss" in joined
    uses_metal_layers = (flag_int(cmd, ["-ngl", "--gpu-layers", "--n-gpu-layers"]) or 0) > 0
    flash_on = command_has_flag(cmd, ["-fa", "--flash-attn"]) and (
        " -fa on " in f" {joined} " or "--flash-attn on" in joined or "--flash-attn=on" in joined or " -fa 1 " in f" {joined} "
    )

    if model_is_gptoss_120b and uses_metal_layers and not args.allow_known_unsafe:
        if args.allow_gptoss_metal_scale_gate:
            errors.extend(validate_gptoss_metal_gate(cmd, flash_on, 128, 24, 16, allow_carved_pool=True))
        elif args.allow_gptoss_metal_parity_gate:
            errors.extend(validate_gptoss_metal_gate(cmd, flash_on, 128, 16))
        elif args.allow_gptoss_metal_smoke:
            errors.extend(validate_gptoss_metal_gate(cmd, flash_on, 8, 16))
        else:
            errors.append("blocked known-unsafe shape: gpt-oss-120b with Metal GPU layers")

    if model_is_gptoss_120b and flash_on and not args.allow_known_unsafe:
        errors.append("blocked known-unsafe shape: gpt-oss-120b with flash attention on Metal")

    if "llama-bench" in joined and "gpt-oss-120b" in joined and not args.allow_known_unsafe:
        errors.append("blocked: llama-bench cannot pass --no-repack for gpt-oss-120b CPU-MoE tests")

    return errors


def terminate_process_group(pgid: int, grace_s: float = 3.0) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    deadline = time.time() + grace_s
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="full command output log path")
    parser.add_argument("--summary", type=Path, help="compact guard summary path")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--preflight-min-free-pct", type=float, default=15.0)
    parser.add_argument("--min-free-pct", type=float, default=8.0)
    parser.add_argument("--max-wired-gib", type=float, default=34.0)
    parser.add_argument("--max-rss-mib", type=float, default=38000.0)
    parser.add_argument("--max-swapout-delta", type=float, default=50_000)
    parser.add_argument("--max-memory-pressure-failures", type=int, default=3)
    parser.add_argument("--max-log-mb", type=float, default=64.0)
    parser.add_argument("--allow-docker", action="store_true")
    parser.add_argument("--allow-concurrent-model-io", action="store_true")
    parser.add_argument("--allow-known-unsafe", action="store_true")
    parser.add_argument("--allow-gptoss-metal-smoke", action="store_true")
    parser.add_argument("--allow-gptoss-metal-parity-gate", action="store_true")
    parser.add_argument("--allow-gptoss-metal-scale-gate", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command after --")

    errors = preflight(args, cmd)
    if errors:
        for err in errors:
            print(f"safe-run preflight blocked: {err}", file=sys.stderr)
        return 2

    model_io_lock_fd: int | None = None
    if not args.allow_concurrent_model_io:
        model_io_lock_fd, lock_error = acquire_model_io_lock(cmd)
        if lock_error:
            print(f"safe-run preflight blocked: {lock_error}", file=sys.stderr)
            return 2

    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)

    start_stats = parse_memory_pressure()
    start_swapouts = start_stats.get("swapouts", 0.0)
    start = time.time()
    tail: deque[str] = deque(maxlen=80)
    stop_reason = "process exited"
    returncode: int | None = None

    with args.log.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)
        last_poll = 0.0
        memory_pressure_failures = 0
        pattern_window = ""

        assert proc.stdout is not None
        stdout_fd = proc.stdout.fileno()
        os.set_blocking(stdout_fd, False)

        def append_output(text: str) -> None:
            if not text:
                return
            log.write(text)
            for out_line in text.splitlines():
                tail.append(out_line)

        def drain_stdout() -> bool:
            nonlocal pattern_window, stop_reason
            saw_eof = False
            while True:
                try:
                    chunk = os.read(stdout_fd, 65536)
                except BlockingIOError:
                    break
                except InterruptedError:
                    continue
                if not chunk:
                    saw_eof = True
                    break
                text = chunk.decode("utf-8", errors="replace")
                append_output(text)
                pattern_window = (pattern_window + text)[-4096:]
                for pattern in METAL_OOM_PATTERNS:
                    if pattern in pattern_window:
                        stop_reason = f"log pattern triggered: {pattern}"
                        terminate_process_group(pgid)
                        return saw_eof
            return saw_eof

        while True:
            drain_stdout()
            if stop_reason.startswith("log pattern"):
                break

            now = time.time()
            if now - last_poll >= args.poll_seconds:
                last_poll = now
                stats = parse_memory_pressure()
                rss_mib = process_group_rss_mib(pgid)
                elapsed = now - start
                log_size_mb = args.log.stat().st_size / 1024**2 if args.log.exists() else 0.0
                if "free_pct" not in stats:
                    memory_pressure_failures += 1
                    if memory_pressure_failures >= args.max_memory_pressure_failures:
                        stop_reason = f"memory_pressure failed {memory_pressure_failures} consecutive times"
                        terminate_process_group(pgid)
                        break
                else:
                    memory_pressure_failures = 0

                if elapsed > args.max_seconds:
                    stop_reason = f"timeout {elapsed:.1f}s > {args.max_seconds:.1f}s"
                    terminate_process_group(pgid)
                    break
                if stats.get("free_pct", 100.0) < args.min_free_pct:
                    stop_reason = f"free memory {stats.get('free_pct')}% < {args.min_free_pct}%"
                    terminate_process_group(pgid)
                    break
                if stats.get("wired_gib", 0.0) > args.max_wired_gib:
                    stop_reason = f"wired memory {stats.get('wired_gib'):.2f}GiB > {args.max_wired_gib:.2f}GiB"
                    terminate_process_group(pgid)
                    break
                if rss_mib > args.max_rss_mib:
                    stop_reason = f"process-group RSS {rss_mib:.0f}MiB > {args.max_rss_mib:.0f}MiB"
                    terminate_process_group(pgid)
                    break
                if stats.get("swapouts", start_swapouts) - start_swapouts > args.max_swapout_delta:
                    stop_reason = f"swapout delta {stats.get('swapouts', 0.0) - start_swapouts:.0f} > {args.max_swapout_delta:.0f}"
                    terminate_process_group(pgid)
                    break
                if log_size_mb > args.max_log_mb:
                    stop_reason = f"log size {log_size_mb:.1f}MiB > {args.max_log_mb:.1f}MiB"
                    terminate_process_group(pgid)
                    break

            returncode = proc.poll()
            if returncode is not None:
                break

            time.sleep(0.05)

        if returncode is None:
            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop_reason = f"{stop_reason}; child still alive after SIGKILL"
                returncode = 124
        if proc.poll() is not None:
            drain_stdout()

    end_stats = parse_memory_pressure()
    summary_lines = [
        f"command: {' '.join(cmd)}",
        f"returncode: {returncode}",
        f"stop_reason: {stop_reason}",
        f"elapsed_s: {time.time() - start:.2f}",
        f"start_free_pct: {start_stats.get('free_pct', 'n/a')}",
        f"end_free_pct: {end_stats.get('free_pct', 'n/a')}",
        f"end_wired_gib: {end_stats.get('wired_gib', 'n/a')}",
        "",
        "tail:",
        *tail,
    ]
    summary = "\n".join(str(x) for x in summary_lines) + "\n"
    if args.summary:
        args.summary.write_text(summary, encoding="utf-8")
    print(summary)
    return returncode if returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
