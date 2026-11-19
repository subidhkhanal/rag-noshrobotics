"""Context precision on vague, preference-style requests ("something light", "for my kids", "party starters").

Runs each query through the real search (Nosh One filter on, as in the app) and asks LLM judges whether each of the
top 5 dishes fits the request. Reports precision@5 per judge and how often the judges agree (Cohen's kappa).

To limit self-written-test bias, --heldout first has a model that never saw the system write a fresh set of vague
requests (data/eval/vague_heldout.json, reused if present). The tuned set above is kept for comparison.

    ANTHROPIC_API_KEY=... python scripts/vague_eval.py [--heldout] [--judges opus,sonnet] [--save out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERIES = [
    "something light for dinner", "a healthy lunch", "food for my kids, nothing spicy", "kid friendly snack",
    "party starters for guests", "something to cook when guests come over", "comfort food for a rainy day",
    "something soothing when I'm sick", "high protein vegetarian dinner", "protein rich breakfast",
    "a rich indulgent weekend dinner", "festive dish for diwali", "simple everyday dal sabzi type food",
    "quick weeknight dinner", "light and easy breakfast", "something not too oily", "healthy snack for evening tea",
    "halka khana chahiye", "bachon ke liye kuch", "mehmaan aa rahe hain kuch special",
    "something filling after the gym", "a light summer lunch", "warm soup for the cold",
    "something sweet but not too heavy",
]
SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
        "n": {"type": "integer"}, "fits": {"type": "boolean"}, "why": {"type": "string"}},
        "required": ["n", "fits", "why"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False,
}
SYSTEM = """You judge a recipe search. For the user's request and each numbered dish (name, course, time,
description, ingredients), decide if the dish genuinely FITS the request as a sensible home cook would read it
(e.g. "light" = not heavy/fried/creamy; "for kids" = mild and kid-appealing; "party starters" = snack/starter
suited to guests; "high protein" = a real protein source; "comfort food" = warm, homely). A dish that only
shares a word but doesn't fit, or is the wrong kind of dish (a condiment, drink or dessert for a dinner request),
does not fit. Be strict and consistent."""


JUDGES = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}
HELDOUT = ROOT / "data" / "eval" / "vague_heldout.json"
GEN_SYSTEM = """You write realistic, VAGUE requests that Indian home cooks type into a recipe app for a cooking robot:
moods, occasions, audiences, health goals, weather, festivals, textures, effort level. No dish names, no exact
ingredient lists. Mix English, Hinglish (Roman script) and a few in Hindi Devanagari. Return {"queries": [...]}."""


def heldout_queries(client, n: int = 40) -> list[str]:
    if HELDOUT.exists():
        return json.loads(HELDOUT.read_text(encoding="utf-8"))
    resp = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=4000, system=GEN_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
            "required": ["queries"], "additionalProperties": False}}},
        messages=[{"role": "user", "content": f"Write {n} distinct requests."}])
    qs = json.loads(next(b.text for b in resp.content if b.type == "text"))["queries"][:n]
    HELDOUT.write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
    return qs


def judge(client, model: str, q: str, listing: str, n: int) -> list[dict]:
    for _ in range(3):  # the judge must rate every dish; retry if it skips some
        resp = client.messages.create(
            model=model, max_tokens=3000, system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": f"Request: {q}\n\nDishes:\n{listing}\n\n"
                                                  f"Return exactly {n} verdicts, one per dish."}])
        v = json.loads(next(b.text for b in resp.content if b.type == "text"))["verdicts"]
        if len(v) == n:
            return v
    return v


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def describe(r: dict) -> str:
    ings = ", ".join(dict.fromkeys(i["name"] for i in r["ingredients"]))[:240]
    return f"{r['name']} | {r['course']} | {r['total_min']} min | {r.get('description', '')} | ingredients: {ings}"


def run(save: str | None = None, heldout: bool = False, judges: tuple = ("opus",)) -> dict:
    import anthropic

    from forge.query import parse
    from forge.search import Engine

    e = Engine()
    client = anthropic.Anthropic(max_retries=3)
    queries = heldout_queries(client) if heldout else QUERIES
    rows = []
    for q in queries:
        c = parse(e.onto, q)
        c.robot_only = True
        dishes = [r.recipe for r in e.search(c, k=5).results]
        if not dishes:
            continue
        listing = "\n".join(f"{n}. {describe(r)}" for n, r in enumerate(dishes, 1))
        row = {"query": q, "dishes": [r["name"] for r in dishes]}
        for j in judges:
            v = judge(client, JUDGES[j], q, listing, len(dishes))
            row[j] = {"precision@5": sum(x["fits"] for x in v) / len(dishes), "verdicts": v}
        rows.append(row)
        print("  " + "  ".join(f"{j} {row[j]['precision@5']:.1f}" for j in judges) + f"  {q}", flush=True)
    summary = {j: sum(r[j]["precision@5"] for r in rows) / len(rows) for j in judges}
    print(f"\n{'held-out' if heldout else 'tuned'} set, {len(rows)} vague queries, context precision@5:")
    for j, p in summary.items():
        print(f"  {j:7} {p:.3f}")
    if len(judges) == 2:
        a, b = judges
        fa, fb = [], []
        for r in rows:
            if len(r[a]["verdicts"]) == len(r[b]["verdicts"]) == len(r["dishes"]):
                fa += [x["fits"] for x in r[a]["verdicts"]]
                fb += [x["fits"] for x in r[b]["verdicts"]]
        agree = sum(x == y for x, y in zip(fa, fb)) / max(1, len(fa))
        summary["agreement"], summary["kappa"] = agree, kappa(fa, fb)
        print(f"  judges agree on {agree:.0%} of {len(fa)} dish verdicts, Cohen's kappa {summary['kappa']:.2f}")
    if save:
        Path(save).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--judges", default="opus")
    a = ap.parse_args()
    run(a.save, a.heldout, tuple(a.judges.split(",")))
