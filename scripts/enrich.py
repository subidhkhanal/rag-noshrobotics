"""One-off: write a short search description and alternate names for every recipe (Claude Haiku).

Recipes list ingredients and steps but rarely say what the dish *is* ("deep-fried puffed bread" never appears in
a puri recipe), so descriptive searches miss. This adds that text to the search index only. Diet and allergen
checks never read it: they run on the real ingredient lines.

    ANTHROPIC_API_KEY=... python scripts/enrich.py          # resumable; appends to data/processed/descriptions.jsonl
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "recipes.jsonl"
OUT = ROOT / "data" / "processed" / "descriptions.jsonl"
MODEL = "claude-haiku-5-5"
SYSTEM = """You write search metadata for one recipe in an Indian recipe app.
Return:
- aka: up to 5 other names people use for this exact dish (English, Hindi/regional names in Roman script,
  common spellings). Empty if none. Never include the recipe's own name.
- description: one plain sentence, at most 25 words, saying what the dish is: its type (bread, curry, dessert,
  drink...), texture, how it is cooked, and when or how it is usually eaten.
Base everything on the recipe given. Do not mention ingredients that are not listed. Do not make diet, health,
allergy or nutrition claims (no "vegan", "healthy", "gluten-free" and the like)."""
SCHEMA = {
    "type": "object",
    "properties": {"aka": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"}},
    "required": ["aka", "description"],
    "additionalProperties": False,
}


def prompt(r: dict) -> str:
    names = ", ".join(dict.fromkeys(i["name"] for i in r["ingredients"]))[:600]
    return (f"Name: {r['name']}\nCuisine: {r['cuisine']}\nCourse: {r['course']}\nIngredients: {names}\n"
            f"Method (start): {r['instructions'][:400]}")


def describe(client: anthropic.Anthropic, r: dict) -> dict:
    for attempt in range(6):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=1024, system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": prompt(r)}])
            data = json.loads(next(b.text for b in resp.content if b.type == "text"))
            aka = [a.strip() for a in data["aka"] if a.strip() and a.strip().lower() != r["name"].lower()][:5]
            return {"id": r["id"], "aka": aka, "description": data["description"].strip()[:300],
                    "tokens": [resp.usage.input_tokens, resp.usage.output_tokens]}
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on recipe {r['id']}")


def main(workers: int = 12) -> None:
    recipes = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines()]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in recipes if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    client = anthropic.Anthropic(timeout=60, max_retries=2)
    lock = threading.Lock()
    tin = tout = n = 0
    with OUT.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(describe, client, r) for r in todo]
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001 - skip and report; a rerun picks it up
                print("skip:", e, flush=True)
                continue
            with lock:
                tin += row["tokens"][0]; tout += row["tokens"][1]; n += 1
                f.write(json.dumps({k: row[k] for k in ("id", "aka", "description")}, ensure_ascii=False) + "\n")
                if n % 250 == 0:
                    f.flush()
                    print(f"{n}/{len(todo)}  tokens in {tin:,} out {tout:,}  "
                          f"(~${tin / 1e6 * 0.10 + tout / 1e6 * 0.50:.2f})", flush=True)
    print(f"finished {n}; tokens in {tin:,} out {tout:,}; cost ~${tin / 1e6 * 0.10 + tout / 1e6 * 0.50:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
