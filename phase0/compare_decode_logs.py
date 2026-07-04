#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


TOKENS_RE = re.compile(r"generated_token_ids=\[([^\]]*)\]")
DECODE_RE = re.compile(r"decoded\s+(\d+)\s*/\s*(\d+)\s+tokens\s+in\s+([0-9.]+)s,\s+([0-9.]+)\s+tok/s")
GENERATED_TEXT_RE = re.compile(r"generated text written:\s+path=(\S+)\s+bytes=(\d+)")
MOEVM_LINE_RE = re.compile(r"MoEVM prefetch_records=.*")
RUNTIME_RE = re.compile(
    r"MoEVM runtime demand_reads=(\d+)\s+prefetch_reads=(\d+)\s+waits=(\d+)\s+"
    r"wait_s=([0-9.]+)\s+evictions=(\d+)\s+high_water=([0-9.]+)GiB"
)
COPY_STATS_RE = re.compile(
    r"MoEVM backend copy_stats .*?eligible=(\d+).*?attempts=(\d+)\s+handled=(\d+)\s+"
    r"callback_fallback=(\d+)(?:\s+graph_patches=(\d+)\s+graph_patch_restores=(\d+))?\s+normal_fallback=(\d+)"
)


def parse_tokens(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    out: dict = {"path": str(path)}

    m = TOKENS_RE.search(text)
    if not m:
        raise ValueError(f"{path}: missing generated_token_ids")
    out["generated_token_ids"] = parse_tokens(m.group(1))

    m = DECODE_RE.search(text)
    if m:
        out["decoded"] = {
            "tokens": int(m.group(1)),
            "target": int(m.group(2)),
            "seconds": float(m.group(3)),
            "tok_s": float(m.group(4)),
        }

    m = GENERATED_TEXT_RE.search(text)
    if m:
        out["generated_text"] = {
            "path": m.group(1),
            "bytes": int(m.group(2)),
        }

    m = MOEVM_LINE_RE.search(text)
    if m:
        fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", m.group(0)))

        def field_int(name: str, default: int = 0) -> int:
            return int(fields.get(name, default))

        out["moevm"] = {
            "prefetch_records": field_int("prefetch_records"),
            "copy_prefetches": field_int("copy_prefetches"),
            "copy_prefetch_skips": field_int("copy_prefetch_skips"),
            "copy_hook_calls": field_int("copy_hook_calls"),
            "handled": field_int("handled"),
            "copied_experts": field_int("copied_experts"),
            "copied_bytes": field_int("copied_bytes"),
        }
        if "copy_mode" in fields:
            out["moevm"]["copy_mode"] = fields["copy_mode"]
            out["moevm"]["async_ops"] = field_int("async_ops")
            out["moevm"]["memcpy_ops"] = field_int("memcpy_ops")
        for name in (
            "shadow_ids_attempts",
            "shadow_ids_applied",
            "shadow_ids_fallback",
            "compact_attempts",
            "compact_applied",
            "compact_fallback",
            "pool_attempts",
            "pool_applied",
            "pool_fallback",
            "pool_component_hits",
            "pool_component_misses",
            "pool_evictions",
            "resident_cache_hits",
            "resident_cache_misses",
            "resident_cache_resets",
        ):
            if name in fields:
                out["moevm"][name] = field_int(name)

    m = RUNTIME_RE.search(text)
    if m:
        out["runtime"] = {
            "demand_reads": int(m.group(1)),
            "prefetch_reads": int(m.group(2)),
            "waits": int(m.group(3)),
            "wait_s": float(m.group(4)),
            "evictions": int(m.group(5)),
            "high_water_gib": float(m.group(6)),
        }

    m = COPY_STATS_RE.search(text)
    if m:
        out["copy_stats"] = {
            "eligible": int(m.group(1)),
            "attempts": int(m.group(2)),
            "handled": int(m.group(3)),
            "callback_fallback": int(m.group(4)),
            "graph_patches": int(m.group(5) or 0),
            "graph_patch_restores": int(m.group(6) or 0),
            "normal_fallback": int(m.group(7)),
        }

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare generated token IDs and decode stats from llama-moe-trace logs.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path, help="write comparison summary as JSON")
    args = parser.parse_args()

    baseline = parse_log(args.baseline)
    candidate = parse_log(args.candidate)
    same = baseline["generated_token_ids"] == candidate["generated_token_ids"]
    summary = {
        "same_tokens": same,
        "baseline": baseline,
        "candidate": candidate,
    }

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0 if same else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
