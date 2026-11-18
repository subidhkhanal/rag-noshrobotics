"""Evaluate constraint compliance and ranking proxies across pipeline configurations.

Usage: python scripts/evaluate.py [--out reports/evaluation.md]
Compliance is judged by an independent regex checker over raw ingredient lines, not by the ontology.
Relevance has no human labels yet, so facet-match proxies are reported instead of nDCG.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from forge.query import parse  # noqa: E402
from forge.search import Engine  # noqa: E402

QUERIES = [
    "no onion no garlic vegetarian", "Jain food options", "jain breakfast", "jain dinner under 40 minutes",
    "sattvic lunch", "bina pyaz lahsun ka khana", "no onion no garlic paneer curry", "vegan recipes",
    "vegan high protein lunch", "vegan dessert", "vegan south indian breakfast", "I'm lactose intolerant what can I make",
    "lactose free dinner under 30 minutes", "dairy free snack", "gluten free breakfast", "gluten free dessert in half an hour",
    "gluten free north indian dinner", "recipes without nuts", "nut free dessert", "peanut allergy, quick snack",
    "egg free cake", "eggless dessert", "no mushrooms vegetarian dinner", "without coconut south indian breakfast",
    "no potato jain snack", "vegetarian breakfast under 20 mins", "high protein non veg under 30 mins",
    "I have chicken tomatoes and rice", "I have paneer and capsicum", "something with paneer and tomatoes",
    "I have potatoes and peas", "dal recipes under 30 minutes", "spicy chettinad dinner", "creamy vegetarian curry",
    "kuch spicy banana hai", "healthy millet breakfast", "quick kids lunchbox", "I want butter chicken but I'm vegetarian",
    "vegan butter chicken", "jain pav bhaji", "no onion no garlic biryani", "gujarati snack", "bengali fish curry",
    "kerala vegetarian lunch", "italian pasta without cheese", "mexican vegan dinner", "thai curry no onion no garlic",
    "something for a dinner party for 6 people", "sesame free asian dinner", "shellfish allergy seafood alternative",
]

INDEPENDENT = {
    "no_onion_garlic": r"\b(onions?|garlic|shallots?|leeks?|chives|lahsun|lehsun|pyaz|scallions?)\b",
    "jain": r"\b(onions?|garlic|shallots?|potato(es)?|aloo|carrots?|beetroots?|radish|mooli|mushrooms?|ginger|honey|eggs?|chicken|mutton|fish|prawns?)\b",
    "sattvic": r"\b(onions?|garlic|shallots?|mushrooms?|eggs?|chicken|mutton|fish|prawns?|wine|rum)\b",
    "vegan": r"\b(milk|ghee|butter|curd|dahi|paneer|cream|cheese|honey|eggs?|yogurt|khoya|mawa)\b",
    "vegetarian": r"\b(chicken|mutton|lamb|fish|prawns?|shrimps?|eggs?|gelatin|bacon|ham|beef|pork|tuna)\b",
    "lactose_free": r"\b(milk|curd|dahi|paneer|cream|cheese|yogurt|khoya|mawa|condensed)\b",
    "gluten_free": r"\b(wheat|maida|atta|semolina|sooji|suji|rava|bread|pasta|noodles|barley|soy sauce|all purpose flour)\b",
    "nut_free": r"\b(almonds?|cashews?|walnuts?|pistachios?|peanuts?|groundnuts?|badam|kaju|pecans?|hazelnuts?)\b",
    "peanut_free": r"\b(peanuts?|groundnuts?|moongphali)\b",
    "egg_free": r"\b(eggs?|mayonnaise|mayo)\b",
    "sesame_free": r"\b(sesame|til|tahini|gingelly)\b",
    "shellfish_free": r"\b(prawns?|shrimps?|crabs?|lobsters?|oysters?|squid)\b",
}
SAFE = re.compile(
    r"onion (nigella )?seeds|nigella|kalonji|coconut (milk|cream|yogurt|curd)|almond milk|badam milk|soy milk|oat milk|"
    r"peanut butter|cocoa butter|almond butter|eggless|eggplant|egg replacer|butter ?scotch|dry ginger|ginger powder|sonth|"
    r"saunth|cream of tartar|ginger ale|honey ?dew|eggless mayo\w*|veg mayo\w*|(rice|corn|millet|ragi|jowar|bajra) ?(flour|noodles|rava)|"
    r"rice rava|idli rava|buckwheat|coconut cream|ham ?burger|gluten free|vegan (butter|cheese|mayo)",
    re.I,
)
EXCLUDE_WORDS = {"mushroom": r"mushrooms?", "coconut": r"coconut", "potato": r"potato(es)?|aloo", "cheese": r"(?<!cottage )cheese",
                 "onion": r"onions?", "garlic": r"garlic"}


def violations(engine, c, rec) -> list[str]:
    bad = []
    lines = [SAFE.sub(" ", it["line"]) for it in rec["ingredients"]]
    for d in c.diets:
        pat = INDEPENDENT.get(d)
        if pat:
            bad += [f"{d}: {l}" for l in lines if re.search(pat, l, re.I)]
    for e in c.exclude:
        pat = EXCLUDE_WORDS.get(e)
        if pat:
            bad += [f"no {e}: {l}" for l in lines if re.search(rf"\b({pat})\b", l, re.I)]
    if c.max_time and not (0 < rec["total_min"] <= c.max_time):
        bad.append(f"time {rec['total_min']} > {c.max_time}")
    return bad


def run(engine: Engine, name: str):
    total, viol, flagged, cover, course_ok, cuisine_ok, sims, lat, empty = 0, [], 0, [], [], [], [], [], 0
    for q in QUERIES:
        c = parse(engine.onto, q)
        t = time.perf_counter()
        out = engine.search(c, k=5)
        lat.append(time.perf_counter() - t)
        if not out.results:
            empty += 1
        idxs = []
        for r in out.results:
            rec, i = r.recipe, engine.by_id[r.recipe["id"]]
            idxs.append(i)
            total += 1
            v = violations(engine, c, rec)
            if v:
                viol.append((q, rec["name"], v[0]))
            flagged += r.flagged
            if c.include:
                cover.append(len(set(c.include) & engine.lineages[i]) / len(c.include))
            if c.course:
                course_ok.append(c.course in rec["course_tags"])
            if c.cuisine:
                cuisine_ok.append(engine._cuisine_ok(i, c.cuisine))
        if engine.emb is not None and len(idxs) > 1:
            m = engine.emb[idxs] @ engine.emb[idxs].T
            sims.append(float((m.sum() - np.trace(m)) / (len(idxs) * (len(idxs) - 1))))
    avg = lambda xs: statistics.mean(xs) if xs else float("nan")
    return {
        "config": name, "queries": len(QUERIES), "results": total, "violations": len(viol),
        "violation_examples": viol[:5], "flagged": flagged, "empty": empty,
        "include_coverage": avg(cover), "course_match": avg(course_ok), "cuisine_match": avg(cuisine_ok),
        "intra_list_similarity": avg(sims), "p50_ms": 1000 * statistics.median(lat), "p95_ms": 1000 * sorted(lat)[int(0.95 * len(lat)) - 1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "evaluation.md"))
    args = ap.parse_args()
    dense = Engine(use_dense=True, use_rerank=True)
    configs = [("BM25 only", False, False), ("BM25 + dense (RRF)", True, False), ("BM25 + dense + cross-encoder", True, True)]
    rows = []
    for name, use_dense, use_rerank in configs:
        dense.use_rerank = use_rerank
        emb = dense.emb
        if not use_dense:
            dense.emb = None
        rows.append(run(dense, name))
        dense.emb = emb
    md = ["# RecipeForge evaluation", "",
          f"{len(QUERIES)} constrained queries, top-5 results each. Compliance is checked by an independent regex "
          "checker over raw ingredient lines (not the ontology). Relevance proxies: share of requested ingredients "
          "present, course/cuisine match rate. Lower intra-list similarity = more varied results.", "",
          "| Config | Results | Violations | Flagged ⚠ | Empty | Ingredient coverage | Course match | Cuisine match | Intra-list sim | p50 ms | p95 ms |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['results']} | **{r['violations']}** | {r['flagged']} | {r['empty']} | "
                  f"{r['include_coverage']:.2f} | {r['course_match']:.2f} | {r['cuisine_match']:.2f} | "
                  f"{r['intra_list_similarity']:.3f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |")
    for r in rows:
        if r["violation_examples"]:
            md += ["", f"Violations in {r['config']}:"] + [f"- `{q}` → {n}: {v}" for q, n, v in r["violation_examples"]]
    out = Path(args.out)
    notes = ""
    if out.exists() and "## Reading the table" in (old := out.read_text(encoding="utf-8")):
        notes = "\n" + old[old.index("## Reading the table"):]
    out.write_text("\n".join(md) + "\n" + notes, encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
