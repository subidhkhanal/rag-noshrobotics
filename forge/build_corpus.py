"""Build the normalized recipe corpus and the source-label audit.

Usage: python -m forge.build_corpus data/raw/IndianFoodDatasetCSV.csv
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .ontology import load
from .recipes import course_tags, is_english, parse_ingredients

OUT = Path(__file__).resolve().parent.parent / "data" / "processed"

SOURCE_LABEL_TO_DIET = {
    "Vegetarian": "vegetarian",
    "High Protein Vegetarian": "vegetarian",
    "Vegan": "vegan",
    "No Onion No Garlic (Sattvic)": "sattvic",
    "Gluten Free": "gluten_free",
    "Eggetarian": "eggetarian",
}


def clean_text(s) -> str:
    s = str(s or "").replace("�", " ").replace("\xa0", " ").replace("﻿", "")
    return re.sub(r"\s+", " ", s).strip()


COMPONENT = re.compile(
    r"\b(masala powder|spice (mix|powder|blend)|podi|powder|premix|homemade (ghee|butter|paneer|curd|yogurt|stock|"
    r"cream|khoya|mawa|tomato puree|.*masala|.*paste|.*sauce|.*spread|.*seasoning|.*dressing|.*syrup)|stock|"
    r"how to (make|cook|boil|steam|roast)|basic\b)",
    re.I,
)


def clean_name(s: str) -> str:
    s = clean_text(s)
    s = re.sub(r"\s*\((recipe )?in hindi\)|\s*\(recipe\)", "", s, flags=re.I)
    s = re.sub(r"\s+Recipe\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" -")
    left, sep, right = s.partition(" - ")
    if sep and right.lower().startswith(left.lower()):
        s = right
    return s


METHOD_NOISE = re.compile(r"\b(no|without|skip|optional(ly)?|instead of|or|eggless|like)\s+\w+", re.I)
# Only high-signal ingredients that are commonly left off ingredient lists; generic words ("meat shredder",
# "sausage-shaped dough") made broader checks too noisy.
METHOD_IDS = {"onion", "shallot", "spring_onion", "leek", "garlic", "ginger_garlic_paste", "egg", "chicken", "fish",
              "shellfish", "mushroom"}
SERVING = re.compile(r"\b(serve|served|serving|along with|pair|accompani|goes well|tastes great|enjoy)\w*\b", re.I)
OTHER_RECIPE = re.compile(r"(\b[\w&'-]+\s+){1,8}recipes?\b", re.I)
DISH_IDS = {"crepes", "momos", "prepared_snack", "indian_sweet", "confectionery", "dairy_sweet", "khichdi", "dhokla",
            "batter", "cake", "biscuit", "roti", "bread"}


def method_only_ingredients(onto, name: str, instructions: str, ings: list[dict]) -> list[str]:
    """Diet-relevant ingredients the cooking steps use that the ingredient list omits (source data errors)."""
    steps = [s for s in re.split(r"(?<=[.!?।])\s*", instructions)
             if not SERVING.search(s) and not re.search(r"परोस|के साथ", s)]
    text = " ".join(steps)
    text = OTHER_RECIPE.sub(" ", text)
    text = re.sub(re.escape(re.sub(r"\s*\(.*", "", name)), " ", text, flags=re.I)
    text = METHOD_NOISE.sub(" ", text)
    listed = set().union(*[onto.lineage(i) for it in ings for i in it["ids"]]) if ings else set()
    out = []
    for m in onto.match(text):
        iid = m.ingredient
        if iid in listed or iid in out or iid not in METHOD_IDS:
            continue
        if iid == "egg" and re.search(r"eggless|egg[- ]free|vegan", name, re.I):
            continue
        if any(iid in onto.lineage(x) or x in onto.lineage(iid) for x in listed):
            continue
        out.append(iid)
    return out


def build(csv_path: str) -> None:
    onto = load()
    df = pd.read_csv(csv_path)
    df = df[df.TranslatedIngredients.map(is_english) & df.TranslatedRecipeName.map(is_english)]
    recipes, seen = [], set()
    for _, r in df.iterrows():
        name = clean_name(r.TranslatedRecipeName)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        ings = parse_ingredients(onto, r.TranslatedIngredients)
        instructions = clean_text(r.TranslatedInstructions)
        hidden = method_only_ingredients(onto, name, instructions, ings)
        verdicts = {}
        for diet in onto.diets:
            v = onto.diet_verdict(diet, ings)
            reasons = list(v.reasons)
            status = v.status
            block = set(onto.diets[diet].get("block", []))
            for iid in hidden:
                hit = onto.effective_tags(iid) & block
                if hit and status != "block":
                    status = "flag"
                    reasons.insert(0, f"method mentions {onto.label(iid)} ({', '.join(sorted(hit))}), "
                                      f"but the ingredient list doesn't")
            verdicts[diet] = {"status": status, "reasons": reasons[:6]}
        ids = sorted({i for it in ings for i in it["ids"]})
        recipes.append(
            {
                "id": int(r.Srno),
                "name": name,
                "cuisine": re.sub(r"\s+Recipes$", "", clean_text(r.Cuisine)),
                "course": clean_text(r.Course),
                "course_tags": course_tags(r.Course),
                "source_diet_label": clean_text(r.Diet),
                "prep_min": int(r.PrepTimeInMins or 0),
                "cook_min": int(r.CookTimeInMins or 0),
                "total_min": int(r.TotalTimeInMins or 0),
                "servings": int(r.Servings or 0),
                "url": clean_text(r.URL),
                "ingredients": ings,
                "ingredient_ids": ids,
                "verdicts": verdicts,
                "instructions": instructions[:1500],
                "method_only": hidden,
                "is_component": bool(COMPONENT.search(name)),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "recipes.jsonl", "w", encoding="utf-8") as f:
        for rec in recipes:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    audit = {"total_recipes": len(recipes), "labels": {}}
    for label, diet in SOURCE_LABEL_TO_DIET.items():
        labelled = [x for x in recipes if x["source_diet_label"] == label]
        bad = [x for x in labelled if x["verdicts"][diet]["status"] == "block"]
        audit["labels"][label] = {
            "checked_as": diet,
            "labelled": len(labelled),
            "violations": len(bad),
            "examples": [
                {"id": x["id"], "name": x["name"], "reasons": x["verdicts"][diet]["reasons"][:3]}
                for x in bad[:8]
            ],
        }
    status_counts = {
        d: dict(Counter(x["verdicts"][d]["status"] for x in recipes)) for d in onto.diets
    }
    audit["verdict_counts"] = status_counts
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(recipes)} recipes")
    for label, a in audit["labels"].items():
        print(f"  source label {label!r}: {a['violations']}/{a['labelled']} violate {a['checked_as']}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/raw/IndianFoodDatasetCSV.csv")
