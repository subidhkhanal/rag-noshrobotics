"""Context precision on vague, preference-style requests ("something light", "for my kids", "party starters").

Runs each query through the real search (Nosh One filter on, as in the app) and asks an LLM judge (Claude Opus)
whether each of the top 5 dishes fits the request. Reports precision@5 overall and per query.

    ANTHROPIC_API_KEY=... python scripts/vague_eval.py [--save reports/vague_eval.json]
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


def describe(r: dict) -> str:
    ings = ", ".join(dict.fromkeys(i["name"] for i in r["ingredients"]))[:240]
    return f"{r['name']} | {r['course']} | {r['total_min']} min | {r.get('description', '')} | ingredients: {ings}"


def run(save: str | None = None) -> dict:
    import anthropic

    from forge.query import parse
    from forge.search import Engine

    e = Engine()
    client = anthropic.Anthropic(max_retries=3)
    rows = []
    for q in QUERIES:
        c = parse(e.onto, q)
        c.robot_only = True
        dishes = [r.recipe for r in e.search(c, k=5).results]
        listing = "\n".join(f"{n}. {describe(r)}" for n, r in enumerate(dishes, 1))
        for _ in range(3):  # the judge must rate every dish; retry if it skips some
            resp = client.messages.create(
                model="claude-opus-5-5", max_tokens=3000, system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": f"Request: {q}\n\nDishes:\n{listing}\n\n"
                                                      f"Return exactly {len(dishes)} verdicts, one per dish."}])
            v = json.loads(next(b.text for b in resp.content if b.type == "text"))["verdicts"]
            if len(v) == len(dishes):
                break
        fits = sum(x["fits"] for x in v)
        rows.append({"query": q, "precision@5": fits / max(1, len(dishes)), "dishes": [r["name"] for r in dishes],
                     "verdicts": v})
        print(f"  {fits}/{len(dishes)}  {q}", flush=True)
    p = sum(r["precision@5"] for r in rows) / len(rows)
    print(f"\ncontext precision@5 on {len(rows)} vague queries: {p:.3f}")
    if save:
        Path(save).write_text(json.dumps({"precision@5": p, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"precision@5": p, "rows": rows}


if __name__ == "__main__":
    run(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--save" else None)
