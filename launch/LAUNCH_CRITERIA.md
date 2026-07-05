# MoEVM Launch Criteria

Declared before reading the final launch-soak summary.

The launch candidate is allowed to ship only if the final soak directory passes
the checks below. Do not relax these after seeing the result.

## Soak Under Test

Default launch soak:

```bash
python3 moevm/phase0/gptoss_soak.py \
  --hours 6 \
  --tokens 64 \
  --logits-every 20 \
  --out-dir moevm/runs/gpt_oss_120b_launch_soak_20260705_6h_detached
```

Current supervised run:

- `screen` session: `moevm_gptoss_soak_6h`
- driver PID file: `moevm/runs/gpt_oss_120b_launch_soak_20260705_6h_detached/soak.pid`

## Hard PASS Gates

Run:

```bash
python3 moevm/phase0/validate_soak.py \
  moevm/runs/gpt_oss_120b_launch_soak_20260705_6h_detached \
  --final \
  --json moevm/runs/gpt_oss_120b_launch_soak_20260705_6h_detached/validation.json
```

The result must be `status: pass`.

Hard failures:

- `summary.json` missing or `failures != 0`.
- Any row has `returncode != 0`.
- Any row has `runtime_errors != 0`.
- Any row has `pressure_critical_events != 0`.
- Any row decodes fewer than 64 tokens.
- Any same-prompt logits snapshot differs from the first snapshot:
  - token mismatches: 0 required.
  - argmax mismatches: 0 required.
  - `max_abs`: 0.0 required.
- First-hour vs last-hour median tok/s drift is above 10% for any prompt with
  enough samples in both windows.
- End wired-memory proxy crosses 38 GiB, or its first-quarter vs last-quarter
  median grows by more than 4 GiB.
- `jetsam_after.log` contains a real kill/jettison/highwater line, or a new
  `JetsamEvent-*.ips` diagnostic report appears after the soak start time. Benign
  RunningBoard "not jetsam managed" chatter is ignored by the validator.

WARN events are allowed only if the process exits normally, `CRITICAL` stays at
zero, and the trim counters explain the recovery. WARN without a clean exit is a
launch blocker.

## Manual Review

After the validator passes, inspect:

```bash
cat moevm/runs/gpt_oss_120b_launch_soak_20260705_6h_detached/summary.json
tail -n 20 moevm/runs/gpt_oss_120b_launch_soak_20260705_6h_detached/results.jsonl
```

The manual review is not a chance to waive failures. It is only to understand
the launch notes and publish the right numbers.

## Tag Rule

Only tag `moevm-launch-v0` after:

1. `validate_soak.py --final` passes.
2. `gpt_oss_120b_q4_benchmark_summary.md` includes the final soak result.
3. The README includes the measured launch table and limits.
4. The one-command repro path has been checked on the launch commit.
