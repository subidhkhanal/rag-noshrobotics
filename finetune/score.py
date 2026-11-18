"""Score parsers on the hand-written test set.

Usage: python -m finetune.score [predictions.jsonl ...]
Each predictions file has one {"query", "pred"} per line; the rule parser is always scored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from forge.query import Constraints, parse

from forge.parser_schema import FIELDS, normalize, parse_json

TEST = Path(__file__).resolve().parent / "data" / "test.jsonl"
HARD = ("diets", "exclude", "max_time")


def rule_pred(onto, q: str) -> dict:
    c = parse(onto, q)
    intent = next((i for i in ("plan", "history", "different", "similar", "more") if i in c.intents), "search")
    if "usual" in c.intents:
        intent = "different"
    return normalize({
        "diets": sorted(c.diets), "exclude": sorted(c.exclude),
        "include": [i for i in c.include if i not in c.swaps.values()],
        "max_time": c.max_time, "course": c.course, "cuisine": c.cuisine, "attributes": sorted(c.attributes),
        "servings": c.servings, "non_veg": c.prefer_non_veg, "intent": intent, "days": c.days,
    })


def set_f1(p: list, g: list) -> tuple[float, float, float]:
    tp = len(set(p) & set(g))
    return tp, len(p), len(g)


def allowed(engine, d: dict) -> frozenset:
    """Recipes a parsed request lets through the hard filter (unknown ids are ignored, as the app would)."""
    key = (tuple(d["diets"]), tuple(d["exclude"]), d["max_time"])
    if key not in _ALLOWED:
        c = Constraints(diets={x for x in d["diets"] if x in engine.onto.diets},
                        exclude={x for x in d["exclude"] if x in engine.onto.ingredients}, max_time=d["max_time"])
        _ALLOWED[key] = frozenset(engine.hard_filter(c)[0])
    return _ALLOWED[key]


_ALLOWED: dict = {}


def union(a: dict, b: dict) -> dict:
    """Hybrid parser: combine hard constraints so either parser can only add restrictions."""
    out = dict(b)
    out["diets"] = sorted(set(a["diets"]) | set(b["diets"]))
    out["exclude"] = sorted(set(a["exclude"]) | set(b["exclude"]))
    times = [t for t in (a["max_time"], b["max_time"]) if t]
    out["max_time"] = min(times) if times else None
    return out


def safety(engine, preds: dict[str, dict], gold: dict[str, dict]) -> dict:
    equiv = unsafe = 0
    over = []
    examples = []
    for q, g in gold.items():
        p = preds.get(q, normalize(None))
        ga, pa = allowed(engine, g), allowed(engine, p)
        leak = pa - ga
        equiv += pa == ga
        unsafe += bool(leak)
        over.append(len(ga - pa) / max(1, len(ga)))
        if leak:
            examples.append((q, len(leak)))
    n = len(gold)
    return {"equivalent": equiv / n, "unsafe": unsafe / n, "over_block": sum(over) / n, "unsafe_examples": examples}


def score(preds: dict[str, dict], gold: dict[str, dict]) -> dict:
    field_acc = {f: 0 for f in FIELDS}
    hard_exact = full_exact = 0
    tp = np_ = ng = 0
    errors = []
    for q, g in gold.items():
        p = preds.get(q, normalize(None))
        for f in FIELDS:
            field_acc[f] += p[f] == g[f]
        hard_exact += all(p[f] == g[f] for f in HARD)
        full_exact += all(p[f] == g[f] for f in FIELDS)
        for f in ("diets", "exclude"):
            a, b, c = set_f1(p[f], g[f])
            tp, np_, ng = tp + a, np_ + b, ng + c
        if not all(p[f] == g[f] for f in HARD):
            errors.append((q, {f: p[f] for f in HARD}, {f: g[f] for f in HARD}))
    n = len(gold)
    prec, rec = (tp / np_ if np_ else 1.0), (tp / ng if ng else 1.0)
    return {
        "n": n, "hard_exact": hard_exact / n, "full_exact": full_exact / n,
        "hard_precision": prec, "hard_recall": rec,
        "fields": {f: v / n for f, v in field_acc.items()}, "hard_errors": errors,
    }


def load_preds(path: str) -> dict[str, dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["query"]: (normalize(r["pred"]) if "pred" in r else parse_json(r["raw"])) for r in rows}


def main(paths: list[str], names: list[str] | None = None, out: str | None = None) -> str:
    from forge.search import Engine
    engine = Engine(profile="fast")
    onto = engine.onto
    gold = {}
    for line in TEST.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        gold[r["query"]] = normalize(r["gold"])
    rules = {q: rule_pred(onto, q) for q in gold}
    systems = {"Rule parser": rules}
    for i, p in enumerate(paths):
        name = names[i] if names else Path(p).stem
        systems[name] = load_preds(p)
        systems[f"Rules ∪ {name}"] = {q: union(rules[q], systems[name].get(q, normalize(None))) for q in gold}
    results = {name: score(preds, gold) | safety(engine, preds, gold) for name, preds in systems.items()}
    lines = [
        "| System | Same recipes allowed as gold | Unsafe (lets a ruled-out recipe through) | Over-blocking | "
        "Hard-constraint exact | Full exact | include | course | cuisine | attributes | intent |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        f = r["fields"]
        lines.append(f"| {name} | {r['equivalent']:.0%} | **{r['unsafe']:.0%}** | {r['over_block']:.1%} | "
                     f"{r['hard_exact']:.0%} | {r['full_exact']:.0%} | {f['include']:.0%} | {f['course']:.0%} | "
                     f"{f['cuisine']:.0%} | {f['attributes']:.0%} | {f['intent']:.0%} |")
    for name, r in results.items():
        if r["unsafe_examples"]:
            lines += ["", f"{name}: unsafe queries — " + "; ".join(f"`{q}` ({n} recipes)" for q, n in r["unsafe_examples"])]
    text = "\n".join(lines)
    print(text)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    return text


if __name__ == "__main__":
    main(sys.argv[1:])
