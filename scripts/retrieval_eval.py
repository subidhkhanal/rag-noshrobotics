"""Retrieval quality on the hand-written golden set (data/eval/golden.yaml), per language.

Runs the real pipeline (rule parser -> hard filter -> hybrid search -> rerank -> MMR, top 10) and scores
each result against the recipe-fact relevance rules in the golden file.

    python scripts/retrieval_eval.py                 # print the table
    python scripts/retrieval_eval.py --json out.json # also save per-query results
    python scripts/retrieval_eval.py --min-ndcg 0.6  # exit 1 if overall nDCG@10 is below the floor (CI)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.query import parse  # noqa: E402
from forge.search import Engine  # noqa: E402

LANGS = ("en", "hinglish", "devanagari")
K = 10


def load_golden(path: Path = ROOT / "data" / "eval" / "golden.yaml") -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def relevant(engine: Engine, i: int, item: dict) -> bool:
    r = engine.recipes[i]
    if "name" in item and not re.search(item["name"], r["name"], re.I):
        return False
    return all(x in engine.lineages[i] for x in item.get("ings", []))


def n_relevant(engine: Engine, item: dict) -> int:
    return sum(relevant(engine, i, item) for i in range(len(engine.recipes)))


def score(rels: list[bool], total: int) -> dict:
    hit5 = float(any(rels[:5]))
    mrr = next((1 / (j + 1) for j, x in enumerate(rels) if x), 0.0)
    dcg = sum(1 / math.log2(j + 2) for j, x in enumerate(rels) if x)
    ideal = sum(1 / math.log2(j + 2) for j in range(min(K, total)))
    return {"hit@5": hit5, "mrr@10": mrr, "ndcg@10": dcg / ideal if ideal else 0.0,
            "p@5": sum(rels[:5]) / 5}


def run(engine: Engine | None = None, golden: list[dict] | None = None) -> list[dict]:
    engine = engine or Engine()
    golden = golden or load_golden()
    rows = []
    for item in golden:
        for ing in item.get("ings", []):
            assert ing in engine.onto.ingredients, f"{item['id']}: unknown ingredient {ing}"
        total = n_relevant(engine, item)
        assert total, f"{item['id']}: no recipe in the corpus satisfies its relevance rule"
        for lang in LANGS:
            q = item[lang]
            out = engine.search(parse(engine.onto, q), k=K)
            ids = [engine.by_id[r.recipe["id"]] for r in out.results]
            rels = [relevant(engine, i, item) for i in ids]
            rows.append({"id": item["id"], "lang": lang, "query": q, **score(rels, total),
                         "top3": [engine.recipes[i]["name"] for i in ids[:3]]})
    return rows


def ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choice(values) for _ in values) / len(values) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for lang in LANGS + ("all",):
        sub = [r for r in rows if lang == "all" or r["lang"] == lang]
        out[lang] = {m: sum(r[m] for r in sub) / len(sub) for m in ("hit@5", "mrr@10", "ndcg@10", "p@5")}
        out[lang]["n"] = len(sub)
        out[lang]["ndcg_ci"] = ci([r["ndcg@10"] for r in sub])
    return out


def print_table(summary: dict) -> None:
    print(f"{'':12}{'n':>4}{'Hit@5':>8}{'MRR@10':>8}{'nDCG@10':>9}{'P@5':>7}   nDCG 95% CI")
    for lang, s in summary.items():
        lo, hi = s["ndcg_ci"]
        print(f"{lang:12}{s['n']:>4}{s['hit@5']:>8.3f}{s['mrr@10']:>8.3f}{s['ndcg@10']:>9.3f}{s['p@5']:>7.3f}"
              f"   [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--min-ndcg", type=float)
    ap.add_argument("--misses", action="store_true", help="list queries with no relevant result in the top 5")
    args = ap.parse_args()
    rows = run()
    summary = summarize(rows)
    print_table(summary)
    if args.misses:
        for r in rows:
            if not r["hit@5"]:
                print(f"  MISS {r['lang']:10} {r['query']!r:45} -> {r['top3']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    if args.min_ndcg is not None and summary["all"]["ndcg@10"] < args.min_ndcg:
        print(f"FAIL: nDCG@10 {summary['all']['ndcg@10']:.3f} < {args.min_ndcg}")
        sys.exit(1)
