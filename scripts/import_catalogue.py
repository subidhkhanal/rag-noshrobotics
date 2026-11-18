"""Import a robot-recipe catalogue (e.g. Nosh's own recipes) so the assistant runs on it instead of the demo data.

    python scripts/import_catalogue.py nosh_recipes.json data/catalogues/nosh
    FORGE_DATA_DIR=data/catalogues/nosh uvicorn forge.api:app      # serve it (or set the env var on Vercel)

Input: a JSON list of recipes (format in docs/catalogue_format.md). Every recipe gets the same treatment as the
demo corpus: ingredient lines are mapped to the food ontology, diet and allergen verdicts are computed from them,
and the search index and embeddings are built. A recipe's own `robot` block (Nosh's ground truth: can the robot
cook it, what the user preps) replaces the rule-based guess in forge/robot.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.ontology import load  # noqa: E402
from forge.recipes import course_tags, parse_ingredients  # noqa: E402

REQUIRED = ("name", "ingredients", "steps")


def convert(item: dict, idx: int, onto) -> dict:
    missing = [k for k in REQUIRED if not item.get(k)]
    if missing:
        raise ValueError(f"recipe #{idx} ({item.get('name', '?')}): missing {', '.join(missing)}")
    lines = item["ingredients"] if isinstance(item["ingredients"], list) else str(item["ingredients"]).split("\n")
    ings = parse_ingredients(onto, ",".join(str(x).replace(",", ";") for x in lines))
    steps = item["steps"] if isinstance(item["steps"], str) else " ".join(
        s.strip().rstrip(".") + "." for s in item["steps"] if str(s).strip())
    verdicts = {}
    for diet in onto.diets:
        v = onto.diet_verdict(diet, ings)
        verdicts[diet] = {"status": v.status, "reasons": list(v.reasons)[:6]}
    prep, cook = int(item.get("prep_min") or 0), int(item.get("cook_min") or 0)
    rec = {
        "id": int(item.get("id") or idx),
        "name": str(item["name"]).strip(),
        "cuisine": str(item.get("cuisine") or "Indian"),
        "course": str(item.get("course") or "Main Course"),
        "course_tags": course_tags(str(item.get("course") or "Main Course")),
        "source_diet_label": str(item.get("diet_label") or ""),
        "prep_min": prep, "cook_min": cook, "total_min": int(item.get("total_min") or prep + cook),
        "servings": int(item.get("servings") or 0),
        "url": str(item.get("url") or ""),
        "ingredients": ings,
        "ingredient_ids": sorted({i for it in ings for i in it["ids"]}),
        "verdicts": verdicts,
        "instructions": steps[:3000],
        "method_only": [],
        "is_component": False,
    }
    if isinstance(item.get("robot"), dict):
        r = item["robot"]
        rec["robot"] = {"fit": "robot" if r.get("cookable", True) else "manual", "reason": str(r.get("reason") or ""),
                        "prep": [str(x) for x in r.get("prep_steps", [])][:12], "prep_min": int(r.get("prep_min") or prep),
                        "spices": [], "extra_spices": 0, "batches_for_recipe": 1}
    unknown = [it["line"] for it in ings if not it["ids"]]
    return rec, unknown


def main(src: str, out_dir: str) -> None:
    onto = load()
    items = json.loads(Path(src).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recipes, unknown = [], {}
    for i, item in enumerate(items, 1):
        rec, miss = convert(item, i, onto)
        recipes.append(rec)
        for line in miss:
            unknown[line] = unknown.get(line, 0) + 1
    ids = [r["id"] for r in recipes]
    if len(set(ids)) != len(ids):
        raise ValueError("recipe ids must be unique")
    (out / "recipes.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recipes) + "\n",
                                       encoding="utf-8")
    from forge.search import build_embeddings, build_index
    build_index(onto, out, write=True)
    build_embeddings("fast", out)
    fit = sum(1 for r in recipes if (r.get("robot") or {}).get("fit", "robot") == "robot")
    print(f"imported {len(recipes)} recipes into {out} ({fit} marked robot-cookable by the catalogue or rules)")
    if unknown:
        print(f"{len(unknown)} ingredient lines matched nothing in the ontology (add aliases so diet checks see them):")
        for line, n in sorted(unknown.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {n}× {line}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
