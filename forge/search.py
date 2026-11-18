"""Constraint-first hybrid retrieval.

Pipeline: hard filter (diet/exclusion/time, computed from ingredients) -> course/cuisine filter with
fallback -> BM25 + dense retrieval over survivors -> reciprocal rank fusion -> cross-encoder rerank ->
feature-based adjustments -> MMR diversity -> top k with per-result evidence.
"""
from __future__ import annotations

import gc
import json
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .ontology import Ontology, load, tokenize
from .query import CUISINE_GROUPS, Constraints

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
M2V_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "potion-base-8M"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
TOKEN = re.compile(r"[a-z0-9]+")

SPICE_IDS = {"green_chilli", "red_chilli", "black_pepper", "chilli_sauce", "schezwan_sauce", "curry_paste", "long_pepper"}
CREAMY_IDS = {"cream", "milk", "curd", "cashew", "coconut_milk", "butter", "khoya", "condensed_milk", "cheese", "paneer"}
PROTEIN_IDS = {"paneer", "dal", "chickpeas", "kidney_beans", "tofu", "soy_chunks", "egg", "chicken", "mutton", "fish",
               "shellfish", "peanut", "besan", "cheese", "curd"}
TANGY_IDS = {"tamarind", "lemon", "amchur", "kokum", "anardana", "tomato", "curd", "vinegar"}
HEAVY_IDS = {"cream", "butter", "maida", "condensed_milk", "khoya", "sugar", "cheese", "milk_chocolate", "mayonnaise"}
LIGHT_IDS = {"dal", "millet", "greens", "spinach", "methi_leaves", "gourd", "cabbage", "beans", "broccoli", "cucumber",
             "tomato", "capsicum", "chickpeas", "kidney_beans", "pseudo_grain", "zucchini", "pumpkin", "oats"}
MEAT_TAGS = {"meat", "poultry", "fish", "shellfish", "egg"}
COMPONENT_QUERY = re.compile(r"powder|masala|homemade|paste|sauce|podi|spread|stock|seasoning|dressing|syrup|ghee|how to", re.I)


@dataclass
class Result:
    recipe: dict
    score: float
    checks: list[dict] = field(default_factory=list)
    why: list[str] = field(default_factory=list)
    flagged: bool = False


@dataclass
class SearchOutput:
    results: list[Result]
    trace: dict
    blocked_examples: list[dict]
    closest: Result | None = None
    relaxed: list[str] = field(default_factory=list)


def _norm(v: np.ndarray) -> np.ndarray:
    if len(v) == 0:
        return v
    lo, hi = float(v.min()), float(v.max())
    return np.zeros_like(v) if hi - lo < 1e-9 else (v - lo) / (hi - lo)


class BM25:
    """Okapi BM25 with precomputed per-term postings, so scoring is a few numpy adds."""

    def __init__(self, postings: dict[str, tuple[np.ndarray, np.ndarray]], n_docs: int):
        self.postings, self.n = postings, n_docs

    @classmethod
    def build(cls, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> "BM25":
        lens = np.array([len(d) for d in docs], dtype=np.float32)
        avg = float(lens.mean())
        n = len(docs)
        cols: dict[str, tuple[list[int], list[float]]] = {}
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                ids, tfs = cols.setdefault(t, ([], []))
                ids.append(i)
                tfs.append(f)
        postings = {}
        for t, (ids, tfs) in cols.items():
            idf = math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            ids_a, f = np.array(ids, dtype=np.int32), np.array(tfs, dtype=np.float32)
            w = idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[ids_a] / avg))
            postings[t] = (ids_a, w.astype(np.float32))
        return cls(postings, n)

    def scores(self, query: list[str], idx: np.ndarray) -> np.ndarray:
        full = np.zeros(self.n, dtype=np.float32)
        for t in set(query):
            if t in self.postings:
                ids, w = self.postings[t]
                full[ids] += w
        return full[idx]


def doc_text(r: dict) -> str:
    names = ", ".join(dict.fromkeys(i["name"] for i in r["ingredients"]))
    return f"{r['name']}. {r['cuisine']} {r['course']}. Ingredients: {names}. {r['instructions'][:300]}"


class StaticEmbedder:
    """Model2Vec static embeddings: token-vector mean + L2 norm. Pure numpy, ~80 ms load, <1 ms per query."""

    def __init__(self, model_dir: Path = M2V_DIR):
        from safetensors.numpy import load_file
        from tokenizers import Tokenizer

        self.vectors = load_file(str(model_dir / "model.safetensors"))["embeddings"]
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.vectors.shape[1]), dtype=np.float32)
        for i, enc in enumerate(self.tokenizer.encode_batch(texts, add_special_tokens=False)):
            if enc.ids:
                v = self.vectors[enc.ids[:512]].mean(axis=0)
                out[i] = v / (np.linalg.norm(v) or 1.0)
        return out


class Engine:
    """profile "fast": BM25 + Model2Vec, no reranker (serverless-friendly).
    profile "full": BM25 + bge-small + cross-encoder rerank (needs fastembed/onnxruntime)."""

    def __init__(self, data_dir: Path = DATA, profile: str | None = None, use_dense: bool = True,
                 use_rerank: bool | None = None):
        self.profile = profile or os.getenv("FORGE_PROFILE", "fast")
        self.onto: Ontology = load()
        idx_path = data_dir / "index.pkl"
        src = data_dir / "recipes.jsonl"
        index = None
        if idx_path.exists():
            gc.disable()
            try:
                # Our own build artifact (python -m forge.search index), never user input.
                index = pickle.loads(idx_path.read_bytes())
            finally:
                gc.enable()
            if src.exists() and index.get("source_lines") != count_lines(src):
                index = None
        if index is None:
            index = build_index(self.onto, data_dir)
        self.recipes: list[dict] = index["recipes"]
        self.bm25 = BM25(index["postings"], len(self.recipes))
        self.lineages = index["lineages"]
        self.tags = index["tags"]
        self.name_tokens = index["name_tokens"]
        self.name_ids = index["name_ids"]
        self.by_id = {r["id"]: i for i, r in enumerate(self.recipes)}
        if self.profile == "full":
            try:
                import fastembed  # noqa: F401
            except ImportError:
                self.profile = "fast"
        self.use_dense = use_dense
        self.use_rerank = (self.profile == "full") if use_rerank is None else (use_rerank and self.profile == "full")
        emb_path = data_dir / ("embeddings_bge.npy" if self.profile == "full" else "embeddings_m2v.npy")
        self.emb = np.load(emb_path) if use_dense and emb_path.exists() else None
        self._exclusion_cache: dict[str, list] = {}

    @cached_property
    def embedder(self):
        if self.profile == "full":
            from fastembed import TextEmbedding
            return TextEmbedding(EMBED_MODEL)
        return StaticEmbedder()

    def embed_query(self, text: str) -> np.ndarray:
        if self.profile == "full":
            return np.array(next(iter(self.embedder.query_embed(text))), dtype=np.float32)
        return self.embedder.encode([text])[0]

    @cached_property
    def reranker(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        return TextCrossEncoder(RERANK_MODEL)

    # ---------- hard constraints ----------
    def exclusion_status(self, excluded: str) -> list:
        if excluded not in self._exclusion_cache:
            self._exclusion_cache[excluded] = [
                self.onto.exclusion_verdict(excluded, r["ingredients"]) for r in self.recipes
            ]
        return self._exclusion_cache[excluded]

    def hard_filter(self, c: Constraints, skip_time: bool = False):
        keep, removed = [], Counter()
        blocked_reason: dict[int, str] = {}
        excl = {e: self.exclusion_status(e) for e in c.exclude}
        for i, r in enumerate(self.recipes):
            reason = None
            for d in sorted(c.diets):
                v = r["verdicts"][d]
                if v["status"] == "block":
                    reason = (self.onto.diets[d]["label"], v["reasons"][0] if v["reasons"] else "")
                    break
            if not reason:
                for e, statuses in excl.items():
                    if statuses[i].status == "block":
                        reason = (f"no {self.onto.label(e)}", statuses[i].reasons[0])
                        break
            if not reason and c.max_time and not skip_time and not (0 < r["total_min"] <= c.max_time):
                reason = (f"under {c.max_time} min", f"takes {r['total_min']} min")
            if reason:
                removed[reason[0]] += 1
                blocked_reason[i] = f"{reason[0]}: {reason[1]}"
            else:
                keep.append(i)
        return keep, removed, blocked_reason

    def is_flagged(self, i: int, c: Constraints) -> list[str]:
        r = self.recipes[i]
        notes = []
        for d in sorted(c.diets):
            v = r["verdicts"][d]
            if v["status"] == "flag":
                notes += [f"{self.onto.diets[d]['label']}: {x}" for x in v["reasons"][:2]]
        for e in c.exclude:
            v = self.exclusion_status(e)[i]
            if v.status == "flag":
                notes += v.reasons[:1]
        return notes

    # ---------- soft features ----------
    def features(self, i: int) -> dict[str, float]:
        r, ids = self.recipes[i], set(self.recipes[i]["ingredient_ids"])
        name = r["name"].lower()
        instr = r["instructions"].lower()
        spice = len(ids & SPICE_IDS) / 3 + (0.5 if re.search(r"spicy|masala|chilli|mirchi|pepper|schezwan|chettinad|kolhapuri", name) else 0)
        fried = bool(re.search(r"deep fry|deep-fry|deep fried", instr))
        heavy = len(ids & HEAVY_IDS)
        light = len(ids & LIGHT_IDS)
        return {
            "spicy": min(1.0, spice),
            "mild": 1.0 - min(1.0, spice),
            "creamy": min(1.0, len(ids & CREAMY_IDS) / 2 + (0.5 if re.search(r"creamy|makhani|malai|korma|butter", name) else 0)),
            "high_protein": min(1.0, len(ids & PROTEIN_IDS) / 2 + (0.4 if "High Protein" in r["source_diet_label"] else 0)),
            "healthy": max(0.0, min(1.0, 0.5 + 0.15 * light - 0.2 * heavy - (0.5 if fried else 0))),
            "tangy": min(1.0, len(ids & TANGY_IDS) / 2),
            "crispy": 1.0 if re.search(r"crisp|fry|fried|pakora|bhaji|vada|tikki|cutlet|chips", name) else 0.0,
            "comfort": 1.0 if re.search(r"khichdi|dal|soup|rasam|curd rice|porridge|pongal|kadhi|stew", name) else 0.0,
            "festive": 1.0 if r["course"] in ("Dessert",) or re.search(r"biryani|pulao|kofta|paneer|kheer|halwa|korma|tikka", name) else 0.3,
            "quick": max(0.0, 1 - min(r["total_min"] or 60, 60) / 60),
        }

    def _course_ok(self, i: int, course: str) -> bool:
        return course in self.recipes[i]["course_tags"]

    def _cuisine_ok(self, i: int, cuisine: str) -> bool:
        labels = CUISINE_GROUPS.get(cuisine, [])
        return self.recipes[i]["cuisine"] in labels

    def query_text(self, c: Constraints) -> str:
        parts = [c.text]
        parts += [self.onto.label(i) for i in c.include if self.onto.label(i) not in c.text]
        parts += sorted(a.replace("_", " ") for a in c.attributes)
        if c.course:
            parts.append(c.course)
        if c.cuisine:
            parts.append(c.cuisine)
        if not any(p.strip() for p in parts):
            parts += [self.onto.diets[d]["label"] for d in sorted(c.diets)] + ["home style indian recipe"]
        return " ".join(p for p in parts if p).strip()

    # ---------- main entry ----------
    def search(self, c: Constraints, k: int = 5, exclude_ids: set[int] | None = None,
               anchor_id: int | None = None) -> SearchOutput:
        exclude_ids = exclude_ids or set()
        keep, removed, blocked_reason = self.hard_filter(c)
        trace = {"corpus": len(self.recipes), "removed_by_constraint": dict(removed), "after_hard_filter": len(keep)}
        relaxed: list[str] = []

        cand = [i for i in keep if self.recipes[i]["id"] not in exclude_ids]
        for attr, ok in (("course", self._course_ok), ("cuisine", self._cuisine_ok)):
            val = getattr(c, attr)
            if val:
                narrowed = [i for i in cand if ok(i, val)]
                if len(narrowed) >= 3:
                    cand = narrowed
                else:
                    relaxed.append(f"few {val} matches, widened {attr}")
        if c.pantry_mode and c.include:
            have = set(c.include)
            narrowed = [i for i in cand if have & self.lineages[i]]
            if narrowed:
                cand = narrowed
        trace["after_soft_filters"] = len(cand)

        qtext = self.query_text(c)
        trace["query_text"] = qtext
        qvec = None
        if self.emb is not None:
            qvec = self.embed_query(qtext)

        blocked_examples = self._blocked_examples(qtext, qvec, blocked_reason, c)

        if not cand:
            closest = None
            if c.max_time:
                keep2, _, _ = self.hard_filter(c, skip_time=True)
                pool = [i for i in keep2 if self.recipes[i]["id"] not in exclude_ids]
                if pool:
                    ranked = self._rank(c, pool, qtext, qvec, anchor_id, k=1, trace={})
                    closest = ranked[0] if ranked else None
            return SearchOutput([], trace, blocked_examples, closest, relaxed)

        results = self._rank(c, cand, qtext, qvec, anchor_id, k, trace)
        return SearchOutput(results, trace, blocked_examples, None, relaxed)

    def _rank(self, c, cand, qtext, qvec, anchor_id, k, trace) -> list[Result]:
        idx = np.array(cand)
        qtok = tokenize(qtext)
        lex = self.bm25.scores(qtok, idx)
        ranks = {}
        lex_order = idx[np.argsort(-lex)]
        rrf = Counter()
        for r, i in enumerate(lex_order[:200]):
            rrf[int(i)] += 1 / (60 + r)
        if qvec is not None:
            dense = self.emb[idx] @ qvec
            for r, i in enumerate(idx[np.argsort(-dense)][:200]):
                rrf[int(i)] += 1 / (60 + r)
        pool = [i for i, _ in rrf.most_common(30)] or list(idx[:30])
        trace["fused_pool"] = len(pool)

        rel = np.array([rrf[i] for i in pool], dtype=float)
        if self.use_rerank and len(pool) > 1:
            docs = [f"{self.recipes[i]['name']}. {self.recipes[i]['cuisine']} {self.recipes[i]['course']}. "
                    + ", ".join(dict.fromkeys(x["name"] for x in self.recipes[i]["ingredients"]))[:300] for i in pool]
            rel = np.array(list(self.reranker.rerank(qtext, docs)), dtype=float)
            trace["reranked"] = len(pool)
        rel = _norm(rel)

        anchor_vec = None
        if anchor_id is not None and self.emb is not None and anchor_id in self.by_id:
            anchor_vec = self.emb[self.by_id[anchor_id]]

        scored = []
        include = set(c.include)
        content = {t for t in tokenize(c.text) if len(t) > 2}
        for j, i in enumerate(pool):
            feats = self.features(i)
            s = float(rel[j])
            cover = len(include & self.lineages[i]) / len(include) if include else 0.0
            s += (0.9 if c.pantry_mode else 0.5) * cover
            if content:
                s += 0.5 * len(content & self.name_tokens[i]) / len(content)
            if include:
                s += 0.4 * len(include & self.name_ids[i]) / len(include)
            for a in c.attributes:
                s += 0.3 * feats.get(a, 0.0)
            if c.prefer_non_veg and self.tags[i] & MEAT_TAGS:
                s += 0.4
            if c.cuisine and self._cuisine_ok(i, c.cuisine):
                s += 0.15
            if anchor_vec is not None:
                s += 0.5 * float(self.emb[i] @ anchor_vec)
                if self.recipes[i]["id"] == anchor_id:
                    s -= 0.4
            if self.recipes[i].get("is_component") and not COMPONENT_QUERY.search(qtext):
                s -= 0.8
            flags = self.is_flagged(i, c)
            if flags:
                s -= 0.35
            scored.append((s, i, flags, cover))

        scored.sort(reverse=True)
        chosen = self._mmr(scored, k)
        out = []
        for s, i, flags, cover in chosen:
            out.append(self._explain(i, s, flags, c))
        return out

    def _base_name(self, i: int) -> str:
        return re.sub(r"[^a-z ]", "", self.recipes[i]["name"].lower().split(" - ")[0].split("(")[0]).strip()

    def _mmr(self, scored, k, lam: float = 0.75):
        seen, deduped = set(), []
        for item in scored:
            base = self._base_name(item[1])
            if base not in seen:
                seen.add(base)
                deduped.append(item)
        scored = deduped
        if self.emb is None:
            return scored[:k]
        chosen, rest = [], list(scored)
        while rest and len(chosen) < k:
            best, best_val = None, -1e9
            for item in rest:
                sim = max((float(self.emb[item[1]] @ self.emb[x[1]]) for x in chosen), default=0.0)
                val = lam * item[0] - (1 - lam) * sim
                if val > best_val:
                    best, best_val = item, val
            chosen.append(best)
            rest.remove(best)
        return chosen

    def _explain(self, i: int, score: float, flags: list[str], c: Constraints) -> Result:
        r = self.recipes[i]
        n = len(r["ingredients"])
        checks = []
        for d in sorted(c.diets):
            v = r["verdicts"][d]
            checks.append({"label": self.onto.diets[d]["label"], "status": v["status"],
                           "detail": f"all {n} ingredients checked" if v["status"] == "pass" else "; ".join(v["reasons"][:2])})
        for e in sorted(c.exclude):
            v = self.exclusion_status(e)[i]
            checks.append({"label": f"No {self.onto.label(e)}", "status": v.status,
                           "detail": f"all {n} ingredients checked" if v.status == "pass" else "; ".join(v.reasons[:2])})
        if c.max_time:
            checks.append({"label": f"Under {c.max_time} min", "status": "pass", "detail": f"{r['total_min']} min total"})
        why = []
        if c.include:
            have = [self.onto.label(x) for x in c.include if x in self.lineages[i]]
            miss = [self.onto.label(x) for x in c.include if x not in self.lineages[i]]
            if have:
                why.append("uses " + ", ".join(have) + (f" (no {', '.join(miss)})" if miss else ""))
        ids = set(r["ingredient_ids"])
        if "spicy" in c.attributes and ids & SPICE_IDS:
            why.append("heat from " + ", ".join(sorted(self.onto.label(x) for x in ids & SPICE_IDS)))
        if "creamy" in c.attributes and ids & CREAMY_IDS:
            why.append("creamy from " + ", ".join(sorted(self.onto.label(x) for x in ids & CREAMY_IDS))[:80])
        if "high_protein" in c.attributes and ids & PROTEIN_IDS:
            why.append("protein from " + ", ".join(sorted(self.onto.label(x) for x in ids & PROTEIN_IDS)))
        if "healthy" in c.attributes:
            light = sorted(self.onto.label(x) for x in ids & LIGHT_IDS)
            fried = re.search(r"deep fry|deep-fry|deep fried", r["instructions"].lower())
            why.append(("no deep frying" if not fried else "has a fried step")
                       + (f"; built on {', '.join(light[:3])}" if light else ""))
        if "quick" in c.attributes or c.max_time:
            why.append(f"ready in {r['total_min']} min")
        if c.servings and r["servings"]:
            factor = c.servings / r["servings"]
            if abs(factor - 1) > 0.01:
                why.append(f"serves {r['servings']} — scale ×{factor:.1f} for {c.servings}")
            else:
                why.append(f"serves {r['servings']}")
        return Result(recipe=r, score=round(score, 3), checks=checks, why=why, flagged=bool(flags))

    def _blocked_examples(self, qtext, qvec, blocked_reason, c, n: int = 3) -> list[dict]:
        if not blocked_reason or not (c.diets or c.exclude):
            return []
        idx = np.array([i for i, why in blocked_reason.items() if not why.startswith("under ")])
        if len(idx) == 0:
            return []
        lex = self.bm25.scores(tokenize(qtext), idx)
        score = _norm(lex)
        if qvec is not None:
            score = score + _norm(self.emb[idx] @ qvec)
        top = idx[np.argsort(-score)][:n]
        return [{"id": self.recipes[i]["id"], "name": self.recipes[i]["name"], "reason": blocked_reason[i]} for i in top]


def count_lines(path: Path) -> int:
    """Line-ending-agnostic freshness signature (git may convert LF/CRLF between platforms)."""
    return path.read_bytes().count(b"\n")


def build_index(onto: Ontology, data_dir: Path = DATA, write: bool = False) -> dict:
    recipes = [json.loads(l) for l in (data_dir / "recipes.jsonl").read_text(encoding="utf-8").splitlines()]
    union = lambda sets: frozenset().union(*sets) if sets else frozenset()
    index = {
        "source_lines": count_lines(data_dir / "recipes.jsonl"),
        "recipes": recipes,
        "postings": BM25.build([tokenize((r["name"] + " ") * 3 + doc_text(r)) for r in recipes]).postings,
        "lineages": [union([onto.lineage(i) for i in r["ingredient_ids"]]) for r in recipes],
        "tags": [union([onto.effective_tags(i) for i in r["ingredient_ids"]]) for r in recipes],
        "name_tokens": [frozenset(tokenize(r["name"])) for r in recipes],
        "name_ids": [frozenset(m.ingredient for m in onto.match(r["name"])) for r in recipes],
    }
    if write:
        (data_dir / "index.pkl").write_bytes(pickle.dumps(index, protocol=5))
        print("index.pkl", len(recipes), "recipes")
    return index


def build_embeddings(profile: str, data_dir: Path = DATA) -> None:
    recipes = [json.loads(l) for l in (data_dir / "recipes.jsonl").read_text(encoding="utf-8").splitlines()]
    docs = [doc_text(r) for r in recipes]
    if profile == "full":
        from fastembed import TextEmbedding
        vecs = np.array(list(TextEmbedding(EMBED_MODEL).embed(docs, batch_size=64)), dtype=np.float32)
        out = data_dir / "embeddings_bge.npy"
    else:
        vecs = StaticEmbedder().encode(docs)
        out = data_dir / "embeddings_m2v.npy"
    np.save(out, vecs)
    print(out.name, vecs.shape)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if cmd == "index":
        build_index(load(), write=True)
    else:
        build_embeddings(cmd)
