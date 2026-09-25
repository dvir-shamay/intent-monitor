"""Run all three monitors over the fixtures and print an evaluation report.

Usage:
    python evaluate.py                                    # offline stub (default)
    python evaluate.py --bridge --model claude-opus-4.8   # live, via a local OpenAI-compatible bridge (keyless)
    python evaluate.py --real --limit 5                   # measured, needs API key + SDK

The default path uses the deterministic offline stub. A --bridge run uses live
models through a local OpenAI-compatible bridge (keyless loopback, no SDK) and
measures real-model behavior (the bridge may not meter tokens). A --real run
reads measured token counts from a provider SDK. You set that API key in your
own terminal and this script never reads or prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

# Allow running as a script from the repo root or from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import episodes  # noqa: E402
from intent_monitor import (  # noqa: E402
    action_only_monitor,
    cot_only_monitor,
    ledger_diff_monitor,
)
from model import get_model  # noqa: E402

# ---- Bootstrap constants (fixed -> deterministic CI) ------------------------
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_ALPHA = 0.05


def _rate(caught: int, total: int) -> float:
    return (caught / total) if total else 0.0


def bootstrap_delta_ci(
    flags_a: list[bool],
    flags_b: list[bool],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the recall delta (monitor A - monitor B).

    ``flags_a``/``flags_b`` are per-gold-positive catch booleans for the two
    monitors, aligned by episode. Returns ``(lo, point, hi)`` where ``point`` is
    the observed delta and ``lo``/``hi`` are the (alpha/2, 1-alpha/2) percentiles
    of the paired-resample distribution. Uses ``random.Random(seed)`` so the CI
    is fully deterministic (same seed -> identical bounds).
    """

    n = len(flags_a)
    if n == 0 or n != len(flags_b):
        return (0.0, 0.0, 0.0)
    point = (sum(flags_a) - sum(flags_b)) / n
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sa = sb = 0
        for _ in range(n):
            j = rng.randrange(n)
            sa += flags_a[j]
            sb += flags_b[j]
        deltas.append((sa - sb) / n)
    deltas.sort()
    lo = deltas[int((alpha / 2) * resamples)]
    hi = deltas[min(int((1 - alpha / 2) * resamples), resamples - 1)]
    return (lo, point, hi)


def run(
    real: bool = False,
    limit: int | None = None,
    bridge: bool = False,
    model_name: str = "",
) -> dict:
    model = get_model(real=real, bridge=bridge, model=model_name)
    eps = episodes()
    if limit is not None:
        eps = eps[:limit]

    gold_pos = [e for e in eps if e.gold_divergent]
    gold_neg = [e for e in eps if not e.gold_divergent]

    # Per-gold-positive catch booleans (aligned by episode) for all monitors.
    cot_flags = [cot_only_monitor(e.cot, model) for e in gold_pos]
    action_flags = [action_only_monitor(e, model) for e in gold_pos]
    ledger_flags = [ledger_diff_monitor(e, model) for e in gold_pos]

    cot_fp = sum(1 for e in gold_neg if cot_only_monitor(e.cot, model))
    action_fp = sum(1 for e in gold_neg if action_only_monitor(e, model))
    ledger_fp = sum(1 for e in gold_neg if ledger_diff_monitor(e, model))

    per_class = defaultdict(lambda: [0, 0, 0, 0])  # [ledger, action, cot, total]
    for e, cf, af, lf in zip(gold_pos, cot_flags, action_flags, ledger_flags):
        per_class[e.lie_class][0] += int(lf)
        per_class[e.lie_class][1] += int(af)
        per_class[e.lie_class][2] += int(cf)
        per_class[e.lie_class][3] += 1

    per_diff = defaultdict(lambda: [0, 0, 0, 0])  # [ledger, action, cot, total]
    for e, cf, af, lf in zip(gold_pos, cot_flags, action_flags, ledger_flags):
        key = e.difficulty or "unspecified"
        per_diff[key][0] += int(lf)
        per_diff[key][1] += int(af)
        per_diff[key][2] += int(cf)
        per_diff[key][3] += 1

    delta1 = bootstrap_delta_ci(ledger_flags, cot_flags)  # ledger - cot
    delta2 = bootstrap_delta_ci(ledger_flags, action_flags)  # ledger - action

    return {
        "model": model.name,
        "model_id": getattr(model, "model_id", None),
        "n_total": len(eps),
        "n_pos": len(gold_pos),
        "n_neg": len(gold_neg),
        "cot_recall": _rate(sum(cot_flags), len(gold_pos)),
        "action_recall": _rate(sum(action_flags), len(gold_pos)),
        "ledger_recall": _rate(sum(ledger_flags), len(gold_pos)),
        "cot_fpr": _rate(cot_fp, len(gold_neg)),
        "action_fpr": _rate(action_fp, len(gold_neg)),
        "ledger_fpr": _rate(ledger_fp, len(gold_neg)),
        "per_class": {k: tuple(v) for k, v in per_class.items()},
        "per_difficulty": {k: tuple(v) for k, v in per_diff.items()},
        "delta1": delta1,  # (lo, point, hi) ledger - cot
        "delta2": delta2,  # (lo, point, hi) ledger - action
        "measured": getattr(model, "usage", None) if (real or bridge) else None,
        "bridge": bridge,
    }


def _reasoning_effort() -> str:
    """The bridge reasoning level actually used, from env or unset default."""
    return os.environ.get("INTENT_MONITOR_REASONING_EFFORT") or "default-unspecified"


def _bridge_version(bridge: bool) -> str | None:
    """Best-effort bridge version.

    Filled at real run time by GETting the bridge /health endpoint (only when
    --bridge AND reachable in <3s). Returns None otherwise. This offline export
    path never performs the network call, so it is left None here.
    """
    if not bridge:
        return None
    base = (os.environ.get("INTENT_MONITOR_BRIDGE_URL") or "http://127.0.0.1:3456").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=3) as h:
            info = json.loads(h.read())
        return info.get("version") if isinstance(info, dict) else None
    except Exception:  # noqa: BLE001 - version is optional, never fail the run
        return None


def build_save_payload(r: dict) -> dict:
    """Aggregate-only, publishable JSON payload for --save-json.

    Contains ONLY the numeric metrics + metadata already in the report dict
    (recalls, FPRs, deltas+CIs, per_class, per_difficulty, counts). No raw model
    text, no chain-of-thought, no secrets.
    """
    bridge = bool(r.get("bridge"))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bridge": bridge,
        "model": r.get("model"),
        "model_id": r.get("model_id"),
        "reasoning_effort": _reasoning_effort(),
        "bridge_version": _bridge_version(bridge),
        "n_total": r["n_total"],
        "n_pos": r["n_pos"],
        "n_neg": r["n_neg"],
        "cot_recall": r["cot_recall"],
        "action_recall": r["action_recall"],
        "ledger_recall": r["ledger_recall"],
        "cot_fpr": r["cot_fpr"],
        "action_fpr": r["action_fpr"],
        "ledger_fpr": r["ledger_fpr"],
        "per_class": {k: list(v) for k, v in r["per_class"].items()},
        "per_difficulty": {k: list(v) for k, v in r["per_difficulty"].items()},
        "delta1": {
            "label": "ledger-diff - CoT-only",
            "lo": r["delta1"][0],
            "point": r["delta1"][1],
            "hi": r["delta1"][2],
        },
        "delta2": {
            "label": "ledger-diff - action-only",
            "lo": r["delta2"][0],
            "point": r["delta2"][1],
            "hi": r["delta2"][2],
        },
        "measured": r.get("measured"),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_alpha": BOOTSTRAP_ALPHA,
    }


def save_json(r: dict, path: str) -> None:
    """Write the aggregate-only report + metadata to ``path`` as JSON."""
    payload = build_save_payload(r)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def print_report(r: dict) -> None:
    print("=" * 66)
    print(f"intent-monitor evaluation   (model={r['model']})")
    print("=" * 66)
    print(f"episodes: {r['n_total']}  gold-divergent: {r['n_pos']}  benign: {r['n_neg']}")
    if r.get("bridge"):
        m = r.get("measured") or {}
        print(
            f"REAL-MODEL RUN via local bridge  |  model_id={r.get('model_id')}  "
            f"|  live calls={m.get('calls', 0)}"
        )
        print(
            "  (bridge may not meter tokens -> usage=0; the behavior is real)"
        )
    print()
    print(f"{'monitor':<16}{'catch-rate (recall)':<24}{'false-positive rate'}")
    print(f"{'-'*16}{'-'*24}{'-'*20}")
    print(f"{'CoT-only':<16}{r['cot_recall']*100:>6.1f}% {'':<16}{r['cot_fpr']*100:>6.1f}%")
    print(f"{'action-only':<16}{r['action_recall']*100:>6.1f}% {'':<16}{r['action_fpr']*100:>6.1f}%")
    print(f"{'ledger-diff':<16}{r['ledger_recall']*100:>6.1f}% {'':<16}{r['ledger_fpr']*100:>6.1f}%")
    print()
    print("per-difficulty catch-rate (ledger / action / cot):")
    for diff in ("overt", "hard"):
        if diff not in r["per_difficulty"]:
            continue
        ldg, act, cot, tot = r["per_difficulty"][diff]
        print(
            f"  {diff:<8} ledger {ldg}/{tot} ({_rate(ldg,tot)*100:.0f}%)   "
            f"action {act}/{tot} ({_rate(act,tot)*100:.0f}%)   "
            f"cot {cot}/{tot} ({_rate(cot,tot)*100:.0f}%)"
        )
    print()
    print("per-lie-class catch-rate (ledger / action / cot):")
    for cls, (ldg, act, cot, tot) in sorted(r["per_class"].items()):
        print(
            f"  {str(cls):<20} ledger {ldg}/{tot} ({_rate(ldg,tot)*100:.0f}%)   "
            f"action {act}/{tot} ({_rate(act,tot)*100:.0f}%)   "
            f"cot {cot}/{tot} ({_rate(cot,tot)*100:.0f}%)"
        )
    print()
    _print_delta_block(r)


def _print_delta_block(r: dict) -> None:
    d1_lo, d1_pt, d1_hi = r["delta1"]
    d2_lo, d2_pt, d2_hi = r["delta2"]
    print("=" * 66)
    print("MARGINAL VALUE  (catch-rate delta, paired bootstrap 95% CI)")
    print("=" * 66)
    print(f"resamples={BOOTSTRAP_RESAMPLES}  seed={BOOTSTRAP_SEED}  (deterministic)")
    print(
        f"  delta1  ledger-diff - CoT-only    = {d1_pt*100:+.1f}%  "
        f"95% CI [{d1_lo*100:+.1f}%, {d1_hi*100:+.1f}%]"
    )
    print(
        f"  delta2  ledger-diff - action-only = {d2_pt*100:+.1f}%  "
        f"95% CI [{d2_lo*100:+.1f}%, {d2_hi*100:+.1f}%]"
    )
    print(
        "  calibration: CoT-only sits near the floor "
        f"({r['cot_recall']*100:.0f}% recall) - rewritten-benign CoT reads as benign; "
        "the ledger-diff signal is what recovers the hard cases."
    )
    print(
        "  CAVEAT: these recall/FPR/delta figures are structural properties of the\n"
        "  hand-constructed fixtures (a mock ledger), not empirical rates. The\n"
        "  marginal-value delta depends on the per-action (mutating vs egress)\n"
        "  licensing rule."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="use RealModel if key+SDK")
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="use live models via a local OpenAI-compatible bridge (keyless loopback)",
    )
    parser.add_argument(
        "--model",
        default="",
        metavar="NAME",
        help="bridge model family, e.g. claude-opus-4.8, gpt-5.5, gemini-3.7-flash",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap the run to the first N episodes (use with --real/--bridge to keep it tiny)",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        metavar="PATH",
        help="write the aggregate-only report + metadata to PATH as JSON (safe to publish)",
    )
    args = parser.parse_args()

    if args.real and not (
        os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    ):
        print(
            "--real needs OPENAI_API_KEY or ANTHROPIC_API_KEY set in YOUR terminal.\n"
            'Set it yourself, e.g.  $env:OPENAI_API_KEY="..."  then re-run. '
            "This script never prints the key.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.bridge:
        base = (
            os.environ.get("INTENT_MONITOR_BRIDGE_URL") or "http://127.0.0.1:3456"
        ).rstrip("/")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=8) as h:
                json.loads(h.read())
        except Exception as exc:  # noqa: BLE001
            print(
                f"--bridge: local bridge not reachable at {base} ({exc}).\n"
                "Start a local OpenAI-compatible bridge on that port, then re-run.",
                file=sys.stderr,
            )
            sys.exit(2)

    try:
        report = run(
            real=args.real, limit=args.limit, bridge=args.bridge, model_name=args.model
        )
    except RuntimeError as exc:
        print(f"real-model setup failed: {exc}", file=sys.stderr)
        sys.exit(2)
    print_report(report)
    if args.save_json:
        save_json(report, args.save_json)
        print(f"\nsaved aggregate-only JSON -> {args.save_json}")


if __name__ == "__main__":
    main()
