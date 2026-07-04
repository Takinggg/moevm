#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from array import array
from pathlib import Path


HEADER = struct.Struct("<8sIIII")
ROW = struct.Struct("<Ii")
MAGIC = b"MOEVMLG1"


def read_header(f) -> dict:
    raw = f.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError("short logits header")
    magic, version, n_vocab, rows, reserved = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"bad logits magic: {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported logits version: {version}")
    if reserved != 0:
        raise ValueError(f"unexpected reserved header value: {reserved}")
    return {"version": version, "n_vocab": n_vocab, "rows": rows}


def read_row(f, n_vocab: int) -> tuple[int, int, array]:
    raw = f.read(ROW.size)
    if len(raw) != ROW.size:
        raise EOFError("short row header")
    step, token = ROW.unpack(raw)
    values = array("f")
    values.fromfile(f, n_vocab)
    if len(values) != n_vocab:
        raise EOFError("short logits row")
    return step, token, values


def argmax(values: array) -> tuple[int, float]:
    best_i = 0
    best_v = values[0]
    for i, v in enumerate(values):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i, float(best_v)


def compare(base_path: Path, cand_path: Path) -> dict:
    with base_path.open("rb") as fb, cand_path.open("rb") as fc:
        hb = read_header(fb)
        hc = read_header(fc)
        if hb["n_vocab"] != hc["n_vocab"]:
            raise ValueError(f"n_vocab mismatch: {hb['n_vocab']} != {hc['n_vocab']}")
        if hb["rows"] != hc["rows"]:
            raise ValueError(f"row mismatch: {hb['rows']} != {hc['rows']}")

        n_vocab = hb["n_vocab"]
        rows = hb["rows"]
        total = rows * n_vocab
        sum_abs = 0.0
        sum_sq = 0.0
        max_abs = -1.0
        max_at = {"row": -1, "token": -1}
        sampled_max_abs = -1.0
        sampled_max_at = {"row": -1, "token": -1}
        token_mismatches = []
        argmax_matches = 0
        argmax_mismatches = []

        for row in range(rows):
            step_b, token_b, logits_b = read_row(fb, n_vocab)
            step_c, token_c, logits_c = read_row(fc, n_vocab)
            if step_b != step_c or token_b != token_c:
                token_mismatches.append({
                    "row": row,
                    "baseline_step": step_b,
                    "candidate_step": step_c,
                    "baseline_token": token_b,
                    "candidate_token": token_c,
                })

            argmax_b, argmax_v_b = argmax(logits_b)
            argmax_c, argmax_v_c = argmax(logits_c)
            if argmax_b == argmax_c:
                argmax_matches += 1
            elif len(argmax_mismatches) < 10:
                argmax_mismatches.append({
                    "row": row,
                    "baseline_argmax": argmax_b,
                    "candidate_argmax": argmax_c,
                    "baseline_argmax_logit": argmax_v_b,
                    "candidate_argmax_logit": argmax_v_c,
                })

            if 0 <= token_b < n_vocab:
                sampled_delta = abs(float(logits_b[token_b]) - float(logits_c[token_b]))
                if sampled_delta > sampled_max_abs:
                    sampled_max_abs = sampled_delta
                    sampled_max_at = {"row": row, "token": token_b}

            for i, (a, b) in enumerate(zip(logits_b, logits_c)):
                d = abs(float(a) - float(b))
                sum_abs += d
                sum_sq += d * d
                if d > max_abs:
                    max_abs = d
                    max_at = {"row": row, "token": i}

    return {
        "same_shape": True,
        "n_vocab": n_vocab,
        "rows": rows,
        "token_mismatch_count": len(token_mismatches),
        "token_mismatches_sample": token_mismatches[:10],
        "argmax_match_count": argmax_matches,
        "argmax_mismatch_count": rows - argmax_matches,
        "argmax_mismatches_sample": argmax_mismatches,
        "max_abs": max_abs,
        "max_abs_at": max_at,
        "mean_abs": sum_abs / total if total else 0.0,
        "rms_abs": math.sqrt(sum_sq / total) if total else 0.0,
        "sampled_token_max_abs": sampled_max_abs,
        "sampled_token_max_abs_at": sampled_max_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare binary logits dumps written by llama-moe-trace --logits-out.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path, help="write comparison summary as JSON")
    parser.add_argument("--max-abs-threshold", type=float, help="exit nonzero if max_abs exceeds this threshold")
    args = parser.parse_args()

    summary = compare(args.baseline, args.candidate)
    text = json.dumps(summary, indent=2)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)

    if args.max_abs_threshold is not None and summary["max_abs"] > args.max_abs_threshold:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
