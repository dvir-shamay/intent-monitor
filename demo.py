"""Narrated demo for the intent-monitor safety tool.

Tells the story on ONE benign-looking agent episode whose chain-of-thought
(CoT) sounds safe but whose recorded tool-call ledger actually performs an
UNLICENSED data egress. It runs all three monitors and reveals, beat by beat,
that only the ledger-diff monitor catches it, printing the specific divergence
reason the detector itself returns.

This REUSES the existing monitors and model hooks; it does NOT reimplement the
detection logic.

Usage:
    python demo.py                                    # offline stub (default)
    python demo.py --pace 0.6                         # add per-beat pauses (nice for a GIF)
    python demo.py --bridge --model claude-opus-4.8   # same episode via a keyless local bridge

Offline by default: no network, deterministic stub, stdlib-only. A --bridge run
routes the SAME episode through a live model over a local keyless loopback
bridge; this script never reads or prints any API key.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.request

# Mirror evaluate.py: allow running as a script from any directory, then import
# the shared modules flat.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import episodes  # noqa: E402
from intent_monitor import (  # noqa: E402
    EGRESS_VERBS,
    MUTATING_VERBS,
    READ_VERBS,
    action_only_monitor,
    cot_only_monitor,
    diff_against_ledger,
    extract_intent_claims,
    ledger_diff_monitor,
    _claim_verb,
    _tool_class,
)
from model import get_model  # noqa: E402

# Preferred demo subject: the MIXED hard positive -- CoT states ONLY a mutating
# intent, ledger has a LICENSED mutation (write_file) AND an UNLICENSED egress
# (export). Fall back to the clearest covert-exfiltration hard positive.
_PREFERRED_ID = "hard-mixed-01"


def select_demo_episode():
    """Return (episode, source) where source is 'fixture:<id>' or 'constructed'."""

    eps = episodes()
    by_id = {e.id: e for e in eps}
    if _PREFERRED_ID in by_id:
        return by_id[_PREFERRED_ID], f"fixture:{_PREFERRED_ID}"
    # Fallback: clearest covert-exfiltration hard positive whose ledger egresses
    # through a tool NOT on the harmful blocklist.
    for e in eps:
        if (
            e.gold_divergent
            and e.difficulty == "hard"
            and e.lie_class == "covert-exfiltration"
        ):
            return e, f"fixture:{e.id}"
    raise RuntimeError("no suitable demo episode found in fixtures()")


def run_demo_core(model=None) -> dict:
    """Drive the episode + three monitor verdicts. No printing, no subprocess.

    Returns a dict the narrator (and the test) can consume:
      episode, source, claims, cot_flagged, action_flagged, ledger_flagged,
      diff (the raw dict from diff_against_ledger, incl. divergences).
    """

    if model is None:
        model = get_model()
    episode, source = select_demo_episode()
    claims = extract_intent_claims(episode.cot, model)
    diff = diff_against_ledger(claims, episode.ledger)
    return {
        "episode": episode,
        "source": source,
        "model_name": getattr(model, "name", "stub"),
        "claims": claims,
        "cot_flagged": cot_only_monitor(episode.cot, model),
        "action_flagged": action_only_monitor(episode, model),
        "ledger_flagged": ledger_diff_monitor(episode, model),
        "diff": diff,
    }


def _verdict(flagged: bool) -> str:
    return "FLAGGED" if flagged else "NOT FLAGGED"


def _rule(char: str = "=") -> str:
    return char * 70


def narrate(core: dict, pace: float = 0.0) -> None:
    """Print the slow, GIF-friendly, beat-by-beat narrative."""

    def beat() -> None:
        # Blank-line spacing between beats; optional pause for screen recording.
        print()
        if pace > 0:
            time.sleep(pace)

    e = core["episode"]
    claims = core["claims"]
    diff = core["diff"]

    egress_calls = [c for c in e.ledger if c.get("tool", "") in ("export", "publish", "sync", "send", "upload")]

    print(_rule())
    print("  intent-monitor demo")
    print(_rule())
    print(f"  demo episode : {e.id}")
    print(f"  lie-class    : {e.lie_class}   (difficulty: {e.difficulty})")
    print(f"  source       : {core['source']}")
    print(f"  judge model  : {core['model_name']}")
    beat()

    # 1. SCENARIO
    print("[1] SCENARIO")
    print(_rule("-"))
    print("  An agent was asked to read a record and update it for the user.")
    print("  Here is what it CLAIMED it would do, and what it ACTUALLY did.")
    beat()

    # 2. THE AGENT'S REASONING (CoT)
    print("[2] THE AGENT'S REASONING  (chain-of-thought)")
    print(_rule("-"))
    print(f"    \"{e.cot}\"")
    print()
    print("  Reads as benign: it only talks about reading and updating a record.")
    print("  Extracted intent-claims:")
    for c in claims:
        print(f"     - {c}")
    beat()

    # 3. THE ACTION LEDGER
    print("[3] THE ACTION LEDGER  (tamper-evident record of what it ACTUALLY did)")
    print(_rule("-"))
    for i, call in enumerate(e.ledger, 1):
        tool = call.get("tool", "")
        args = call.get("args", {})
        marker = "   <-- EGRESS: data leaves the trust boundary" if call in egress_calls else ""
        print(f"    {i}. {tool}({json.dumps(args)}){marker}")
    print()
    print("  The last call ships data OUT to an external sink -- something the")
    print("  reasoning never mentioned.")
    beat()

    # 4. MONITOR 1 -- CoT-only
    print("[4] MONITOR 1  --  CoT-only  (judge the reasoning text alone)")
    print(_rule("-"))
    print(f"    verdict: {_verdict(core['cot_flagged'])}")
    print("    why: the reasoning sounds fine -- no harmful words to catch.")
    beat()

    # 5. MONITOR 2 -- action-only (blocklist)
    print("[5] MONITOR 2  --  action-only  (blocklist of known-harmful tools)")
    print(_rule("-"))
    print(f"    verdict: {_verdict(core['action_flagged'])}")
    print("    why: 'export' is not on the harmful-tool blocklist, so a")
    print("         tool-name-only monitor sees nothing wrong.")
    beat()

    # 6. MONITOR 3 -- ledger-diff (this tool)
    print("[6] MONITOR 3  --  ledger-diff  (diff stated intent vs the ledger)  << this tool")
    print(_rule("-"))
    print(f"    verdict: {_verdict(core['ledger_flagged'])}")
    print("    reason(s) returned by the detector:")
    if diff["divergences"]:
        for d in diff["divergences"]:
            print(f"       [{d['kind']}]")
            print(f"       {d['detail']}")
    else:
        print("       (no divergences)")
    beat()

    # 7. ONE-LINE TAKEAWAY
    print("[7] TAKEAWAY")
    print(_rule("-"))
    print("  Diffing stated intent against the action ledger catches the case")
    print("  both baselines miss.")
    beat()

    print(_rule())
    print("  CAVEAT")
    print(_rule())
    _mn = core.get("model_name", "stub")
    if _mn == "stub":
        _judged_by = "a deterministic offline stub"
    elif _mn == "bridge":
        _judged_by = "a live model via the local keyless bridge"
    else:
        _judged_by = f"a live model ({_mn})"
    print("  This is a CONCEPT/MECHANISM demo on ONE hand-constructed episode,")
    print(f"  judged by {_judged_by}. It is NOT an empirical catch-rate.")
    print(_rule())


# The four divergence kinds that constitute a real ledger-diff FLAG. An action
# named in one of these is UNLICENSED by construction; we read the licensing
# verdict straight off the detector's own output so it can never disagree with
# ledger_diff_monitor.
_FLAGGING_KINDS = frozenset(
    {
        "benign_intent_harmful_action",
        "unstated_harmful_action",
        "unlicensed_mutating_action",
        "unlicensed_egress_action",
    }
)


def _claim_verb_class(verb: str) -> str:
    """Cosmetic label for a claimed intent verb's licensing sub-class."""

    if verb in READ_VERBS:
        return "read-class (benign)"
    if verb in MUTATING_VERBS:
        return "mutating-class"
    if verb in EGRESS_VERBS:
        return "egress-class"
    return "other"


def _unlicensed_tools(diff: dict) -> set:
    """AUTHORITATIVE unlicensed set, read off the detector's real divergences.

    A ledger tool is unlicensed iff it is named in a FLAGGING divergence's
    detail. We parse the tool the detector itself references (``... tool 'X'``)
    instead of re-deriving the licensing rule, so this can never disagree with
    the ledger-diff verdict.
    """

    tools: set = set()
    for d in diff.get("divergences", []):
        if d.get("kind") in _FLAGGING_KINDS:
            m = re.search(r"tool '([^']+)'", d.get("detail", ""))
            if m:
                tools.add(m.group(1))
    return tools


def _licensing_verb(tool: str, claims: list) -> str:
    """Cosmetic: which stated verb licenses this (already-licensed) action."""

    cls = _tool_class(tool)
    if cls == "mutating":
        want = MUTATING_VERBS
    elif cls == "egress":
        want = EGRESS_VERBS
    else:
        want = READ_VERBS
    for c in claims:
        v = _claim_verb(c)
        if v in want:
            return v
    # Read-class actions are inherently benign even with no explicit read claim.
    return "read" if cls == "benign" else "?"


def _short_arg(args: dict) -> str:
    """First key=value of a tool call's args, for a compact one-liner."""

    if not args:
        return ""
    k, v = next(iter(args.items()))
    return f"{k}={v}"


def _compact_rule(char: str = "=") -> str:
    return char * 56


def compact_view(core: dict) -> str:
    """Render the CONDENSED single-screen view that makes the mechanism explicit:
    stated intent (from CoT) diffed against the action ledger, with the
    per-action licensing verdict read straight off the real divergences.

    ASCII-only (no unicode glyphs) so it is safe on Windows cp1252 consoles.
    """

    e = core["episode"]
    claims = core["claims"]
    diff = core["diff"]
    unlicensed = _unlicensed_tools(diff)

    out: list = []
    out.append(_compact_rule())
    out.append(f" intent-monitor -- compact view : {e.id}")
    out.append(_compact_rule())
    out.append("")

    # 2. STATED INTENT (from the chain-of-thought)
    out.append("stated intent (from the chain-of-thought)")
    for c in claims:
        v = _claim_verb(c)
        out.append(f'  claim: "{c}"')
        out.append(f"     verb '{v}' -> {_claim_verb_class(v)}")
    verbs = ", ".join(sorted({_claim_verb(c) for c in claims}))
    out.append(f"  licensed verb-classes: {verbs}")
    out.append("")

    # 3. ACTION LEDGER
    out.append("action ledger (what it ACTUALLY did)")
    for i, call in enumerate(e.ledger, 1):
        tool = call.get("tool", "")
        out.append(f"  {i}. {tool:<10} {_short_arg(call.get('args', {}))}")
    out.append("")

    # 4. INTENT -> ACTION LICENSING  (the hero section)
    out.append("intent -> action licensing")
    for i, call in enumerate(e.ledger, 1):
        tool = call.get("tool", "")
        if tool in unlicensed:
            status = "UNLICENSED"
        else:
            status = f"licensed by '{_licensing_verb(tool, claims)}'"
        out.append(f"  {i}. {tool:<10} -> {status}")
    out.append("")

    # 5. MONITORS
    out.append("monitors")
    out.append(f"  CoT-only    : {_verdict(core['cot_flagged'])}")
    out.append(f"  action-only : {_verdict(core['action_flagged'])}")
    out.append(f"  ledger-diff : {_verdict(core['ledger_flagged'])}")
    if diff["divergences"]:
        for d in diff["divergences"]:
            out.append(f"     reason: {d['kind']}")
            for wrapped in textwrap.wrap(d["detail"], width=48):
                out.append(f"       {wrapped}")
    out.append("")

    # 6. CAVEAT (honest scope note)
    out.append(_compact_rule())
    out.append("CAVEAT: concept/mechanism demo on ONE episode")
    out.append("with a deterministic offline stub. It is NOT an")
    out.append("empirical catch-rate.")
    out.append(_compact_rule())
    return "\n".join(out)


def _bridge_reachable() -> tuple[bool, str]:
    """Best-effort health check for the local keyless bridge. Never prints keys."""

    base = (
        os.environ.get("INTENT_MONITOR_BRIDGE_URL") or "http://127.0.0.1:3456"
    ).rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=8) as h:
            json.loads(h.read())
        return True, base
    except Exception as exc:  # noqa: BLE001 - report cleanly, never crash
        return False, f"{base} ({type(exc).__name__})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Narrated intent-monitor demo.")
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="run the SAME episode through a live model via a keyless local bridge",
    )
    parser.add_argument(
        "--model",
        default="",
        metavar="NAME",
        help="bridge model family, e.g. claude-opus-4.8 (only meaningful with --bridge)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="pause this many seconds between beats (nice for a screen recording; default 0)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print a condensed single-screen intent->action licensing view "
        "(always offline/deterministic stub)",
    )
    args = parser.parse_args()

    if args.compact:
        # Compact view is a concept/mechanism snapshot: always offline stub, no
        # network, ASCII-only. --bridge is ignored here by design.
        core = run_demo_core(model=get_model())
        print(compact_view(core))
        return 0

    if args.bridge:
        ok, where = _bridge_reachable()
        if not ok:
            print(
                "--bridge requested but the local bridge is "
                f"unreachable at {where}.\n"
                "Start the bridge (or set INTENT_MONITOR_BRIDGE_URL) and re-run. "
                "Running offline instead requires dropping --bridge.",
                file=sys.stderr,
            )
            return 2
        model = get_model(bridge=True, model=args.model)
    else:
        model = get_model()

    core = run_demo_core(model=model)
    narrate(core, pace=max(0.0, args.pace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
