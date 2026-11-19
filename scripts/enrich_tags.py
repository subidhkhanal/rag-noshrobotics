"""One-off: tag every recipe with the qualities people ask for in vague requests (Claude Haiku).

"Something light", "for my kids", "party starters", "when I'm sick" can't be answered well from ingredient rules
alone, and source course labels are unreliable. This writes data/processed/tags.jsonl with, per recipe:
  qualities  from a fixed vocabulary (light, heavy, high_protein, kid_friendly, party, comfort, soothing,
             filling, cooling, warming, festive, everyday)
  spice      mild | medium | hot
  dish_type  main | side | snack | breakfast | dessert | drink | condiment | bread | soup | salad
Search uses them to filter and rank vague requests. Diet and allergen checks never use them.

    ANTHROPIC_API_KEY=... python scripts/enrich_tags.py     # resumable
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
DESC = ROOT / "data" / "processed" / "descriptions.jsonl"
OUT = ROOT / "data" / "processed" / "tags.jsonl"
QUALITIES = ["light", "heavy", "high_protein", "kid_friendly", "party", "comfort", "soothing", "filling",
             "cooling", "warming", "festive", "everyday"]
DISH_TYPES = ["main", "side", "snack", "breakfast", "dessert", "drink", "condiment", "bread", "soup", "salad"]
SCHEMA = {
    "type": "object",
    "properties": {
        "qualities": {"type": "array", "items": {"type": "string", "enum": QUALITIES}},
        "spice": {"type": "string", "enum": ["mild", "medium", "hot"]},
        "dish_type": {"type": "string", "enum": DISH_TYPES},
    },
    "required": ["qualities", "spice", "dish_type"],
    "additionalProperties": False,
}
SYSTEM = """You tag one Indian-home-cooking recipe with the qualities people ask for. Judge from the ingredients
and method, as an experienced home cook would. Use only tags that clearly apply (usually 1-4):
- light: not fried, not creamy/buttery, modest oil; a light meal.  heavy: rich, creamy, buttery, fried or very filling.
- high_protein: built around a real protein (dal/lentils, beans, chickpeas, paneer, tofu, soy, eggs, meat, fish).
- kid_friendly: mild and familiar, appealing to children.  party: good to serve guests (starters, festive mains,
  crowd-pleasers).  comfort: warm, homely, nostalgic.  soothing: gentle and easy to digest (for someone unwell:
  khichdi, dal-rice, soups, curd rice, porridge).  filling: hearty and satiating.  cooling: good in hot weather.
  warming: good in cold or rainy weather.  festive: made for festivals or special occasions.
  everyday: an ordinary weekday dish.
spice: how hot it is to eat. dish_type: what it actually is (a chutney is a condiment, lassi is a drink)."""


def prompt(r: dict, desc: dict) -> str:
    ings = ", ".join(dict.fromkeys(i["name"] for i in r["ingredients"]))[:500]
    d = desc.get(r["id"], {}).get("description", "")
    return (f"Name: {r['name']}\nCourse label (unreliable): {r['course']}\nDescription: {d}\nIngredients: {ings}\n"
            f"Method (start): {r['instructions'][:350]}")


def tag(client, r: dict, desc: dict) -> dict:
    for attempt in range(6):
        try:
            resp = client.messages.create(
                model="claude-haiku-5-5", max_tokens=1024, system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": prompt(r, desc)}])
            d = json.loads(next(b.text for b in resp.content if b.type == "text"))
            return {"id": r["id"], "qualities": sorted(set(d["qualities"]) & set(QUALITIES)), "spice": d["spice"],
                    "dish_type": d["dish_type"], "tokens": [resp.usage.input_tokens, resp.usage.output_tokens]}
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError, json.JSONDecodeError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {r['id']}")


def main(workers: int = 12) -> None:
    recipes = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines()]
    desc = {d["id"]: d for d in map(json.loads, DESC.read_text(encoding="utf-8").splitlines())} if DESC.exists() else {}
    done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()} if OUT.exists() else set()
    todo = [r for r in recipes if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    client = anthropic.Anthropic(timeout=60, max_retries=2)
    lock, tin, tout, n = threading.Lock(), 0, 0, 0
    with OUT.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(tag, client, r, desc) for r in todo]):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001 - a rerun picks it up
                print("skip:", e, flush=True)
                continue
            with lock:
                tin += row["tokens"][0]; tout += row["tokens"][1]; n += 1
                f.write(json.dumps({k: row[k] for k in ("id", "qualities", "spice", "dish_type")}) + "\n")
                if n % 500 == 0:
                    f.flush()
                    print(f"{n}/{len(todo)}  ~${tin / 1e6 * 0.10 + tout / 1e6 * 0.50:.2f}", flush=True)
    print(f"finished {n}; cost ~${tin / 1e6 * 0.10 + tout / 1e6 * 0.50:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
