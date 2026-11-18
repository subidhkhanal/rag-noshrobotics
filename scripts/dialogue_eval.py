"""Dialogue-understanding eval: right kind of answer, about the right dish (data/eval/dialogue.yaml).

    python scripts/dialogue_eval.py              # as configured (Claude command step on if ANTHROPIC_API_KEY is set)
    python scripts/dialogue_eval.py --rules-only # rules only, no LLM
    python scripts/dialogue_eval.py --fails      # list failures
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def judge(expect: dict, out: dict, screen: dict[str, int], names: dict[int, str]) -> str | None:
    """None if the reply meets the expectation, else a short reason."""
    mode = out.get("mode")
    results = out.get("results") or []
    if mode != expect["mode"]:
        return f"mode {mode} != {expect['mode']}"
    if expect.get("recall") and "latest suggestions" not in out.get("reply", ""):
        return "not a recall answer"
    if mode == "detail":
        item = expect.get("item")
        if item == "all":
            if len(results) > 1 or (results and len(screen) > 1):
                return "answered about one dish, not all"
        elif item is not None:
            want_id = screen.get(item)
            if want_id is None:
                return f"setup has no card {item}"
            got = results[0]["id"] if results else None
            if got != want_id:
                return f"picked {names.get(got, got)!r}, wanted {item}: {names.get(want_id)!r}"
        if "name" in expect:
            got = results[0]["name"].lower() if results else ""
            if expect["name"] not in got:
                return f"picked {got!r}, wanted a '{expect['name']}' dish"
        want = expect.get("want")
        detail = out.get("detail") or {}
        if want == "steps" and not detail.get("steps"):
            return "no steps"
        if want == "ingredients" and (not detail.get("ingredients") or detail.get("steps")):
            return "not an ingredients answer"
    return None


def run(rules_only: bool = False, verbose: bool = False) -> dict:
    if rules_only:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    from forge.chat import Assistant
    from forge.search import Engine

    assistant = Assistant(Engine())
    cases = yaml.safe_load((ROOT / "data" / "eval" / "dialogue.yaml").read_text(encoding="utf-8"))
    by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    fails, latencies = [], []
    for case in cases:
        state = None
        for msg in case["setup"]:
            state = assistant.chat(state, msg)["state"]
        ids, labels = (state or {}).get("last_ids", []), (state or {}).get("last_labels", [])
        screen = dict(zip(labels or [str(i + 1) for i in range(len(ids))], ids))
        names = {r["id"]: r["name"] for r in assistant.engine.recipes if r["id"] in ids}
        for turn in case["turns"]:
            exp = turn["expect"]
            if "name" in exp and not any(exp["name"] in n.lower() for n in names.values()):
                continue  # the setup's results don't include such a dish this time: not a fair test
            t = time.perf_counter()
            try:
                out = assistant.chat(state, turn["say"])
                why = judge(exp, out, screen, names)
            except Exception as e:  # noqa: BLE001 - a crash is a failure, not a stop
                why = f"crash: {e!r}"[:120]
            latencies.append(time.perf_counter() - t)
            ok = why is None
            for tag in (turn["tag"], "all"):
                by_tag[tag][0] += ok
                by_tag[tag][1] += 1
            if not ok:
                fails.append((turn["tag"], turn["say"], why))
    latencies.sort()
    return {"by_tag": dict(by_tag), "fails": fails,
            "p50_ms": 1000 * latencies[len(latencies) // 2], "p95_ms": 1000 * latencies[int(len(latencies) * .95) - 1]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--fails", action="store_true")
    args = ap.parse_args()
    res = run(args.rules_only)
    print(f"{'slice':14}{'pass':>6}{'n':>5}{'rate':>8}")
    for tag, (ok, n) in sorted(res["by_tag"].items(), key=lambda kv: (kv[0] == "all", kv[0])):
        print(f"{tag:14}{ok:>6}{n:>5}{ok / n:>8.0%}")
    print(f"latency p50 {res['p50_ms']:.0f} ms, p95 {res['p95_ms']:.0f} ms")
    if args.fails:
        for tag, say, why in res["fails"]:
            print(f"  FAIL [{tag}] {say!r}: {why}")
