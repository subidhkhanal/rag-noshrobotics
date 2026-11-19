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
from .profile import STAPLES
from .query import COMMON_WORDS, CUISINE_GROUPS, HINGLISH_WORDS, Constraints
from .robot import assess as robot_fit, batches as robot_batches
from .translit import Speller

KEEP_WORDS = COMMON_WORDS | HINGLISH_WORDS

# Vague request words -> recipe qualities tagged offline (scripts/enrich_tags.py).
QUALITY_WORDS = [(re.compile(rx, re.I), q) for rx, q in [
    (r"\b(kids?|children|child|bach+[oe]n?|bachch?[oe]n|toddlers?|little ones|school)\b", "kid_friendly"),
    (r"\b(party|parties|guests?|mehm[ae]an|get ?together|potluck|starters?|appetizers?|special occasion)\b", "party"),
    (r"\b(sick|unwell|fever|soothing|upset (stomach|tummy)|recover\w*|easy to digest|cold and cough|bimaar)\b", "soothing"),
    (r"\b(filling|hearty|gym|workout|hungry|satiating|fuller)\b", "filling"),
    (r"\b(summer|hot day|cooling|garmi|refreshing)\b", "cooling"),
    (r"\b(winter|cold (day|weather|night)|for the cold|rainy|monsoon|baarish|warming|warm (me|up))\b", "warming"),
    (r"\b(everyday|daily|weeknight|regular|ghar ka|homestyle|home style)\b", "everyday"),
    (r"\b(indulgent|indulge|heavy|treat|lavish)\b", "heavy"),
    (r"\b(comfort|comforting|cozy|cosy|homely|soul ?food)\b", "comfort"),
    (r"\b(festive|festival|diwali|holi|eid|navratri|pongal|onam|celebration)\b", "festive"),
]]
ATTRIBUTE_QUALITIES = {"healthy": "light", "comfort": "comfort", "festive": "festive", "high_protein": "high_protein"}
# Requested course -> dish types that genuinely fit it (source course labels are unreliable).
MEAL_TYPES = {"dinner": {"main", "side", "soup", "salad", "bread"}, "lunch": {"main", "side", "soup", "salad", "bread"},
              "snack": {"snack"}, "breakfast": {"breakfast", "main", "snack", "bread"}}


def wanted_qualities(c) -> tuple[set, set, set]:
    """(qualities wanted, qualities to avoid, acceptable spice levels) for a request."""
    text = c.text or ""
    want = {ATTRIBUTE_QUALITIES[a] for a in c.attributes if a in ATTRIBUTE_QUALITIES}
    want |= {q for rx, q in QUALITY_WORDS if rx.search(text)}
    if re.search(r"\bnot (too |so )?heavy\b|\bnot (too |so )?(rich|oily)\b", text, re.I):
        want.add("light")
    if "festive" in c.attributes and re.search(r"\b(party|guests?|starters?)\b", text, re.I):
        want.add("party")
    avoid = {"heavy"} if "light" in want else ({"light"} if "heavy" in want else set())
    if "light" in want and "heavy" in want:
        want.discard("heavy")
        avoid = {"heavy"}
    spice_ok = set()
    if "kid_friendly" in want or "soothing" in want or "mild" in c.attributes:
        spice_ok = {"mild"}
    elif "spicy" in c.attributes:
        spice_ok = {"medium", "hot"}
    return want, avoid, spice_ok

# Domain thesaurus: general English food descriptions -> Indian dish names that fit them. Query expansion only
# (adds search words); it never filters, and diet checks are unaffected.
DISH_SYNONYMS = [(re.compile(rx, re.I), names) for rx, names in [
    (r"\bflat ?breads?\b|\bgriddle bread", "roti chapati paratha phulka thepla bhakri"),
    (r"\b(puff(ed|y)?|balloon\w*)\b.*\bbread|\bfried (wheat )?bread", "puri poori bhatura"),
    (r"\bstuffed (flat)?bread", "paratha kulcha"),
    (r"\bfritters?\b|\bbatter(ed)? (fried )?(veg|vegetable|onion)", "pakora pakoda bhaji bajji bonda"),
    (r"\bdumplings?\b|\b(veg|vegetable|paneer) balls?\b", "kofta momo modak kozhukattai"),
    (r"\b(rice|milk) pudding\b|\bpudding\b", "kheer payasam phirni"),
    (r"\bcrepes?\b|\bfermented batter\b", "dosa"),
    (r"\b(savou?ry|lentil|gram flour|besan) pancakes?\b|\bpancakes?\b", "cheela chilla uttapam"),
    (r"\bdoughnuts?\b|\bdonuts?\b", "vada"),
    (r"\bsemolina\b", "upma rava sooji"),
    (r"\bporridge\b", "upma pongal daliya khichdi"),
    (r"\blentil (stew|soup|curry)\b|\blentils?\b", "dal"),
    (r"\b(steamed )?rice (and lentil )?cakes?\b", "idli"),
    (r"\bsteamed (savou?ry )?(gram flour )?cake\b|\bspongy\b.*\bcake", "dhokla khaman"),
    (r"\byog(h)?urt drink\b|\bchurned\b|\bbuttermilk\b", "lassi chaas"),
    (r"\b(strained|hung) (yog(h)?urt|curd)\b", "shrikhand"),
    (r"\b(flattened|beaten) rice\b", "poha"),
    (r"\bspiced rice\b|\bfragrant rice\b|\bpilaf\b", "pulao biryani"),
    (r"\bcottage cheese\b", "paneer"),
    (r"\bchickpeas?\b", "chole chana"),
    (r"\bkidney beans?\b", "rajma"),
    (r"\bsweet balls?\b|\bround sweets?\b", "ladoo laddu"),
    (r"\b(fried|crispy) (savou?ry )?pastr(y|ies)\b", "samosa kachori"),
    (r"\b(tamarind|pepper) soup\b", "rasam"),
]]

# Which recipe catalogue to serve: the demo corpus by default, or an imported one (scripts/import_catalogue.py),
# e.g. FORGE_DATA_DIR=data/catalogues/nosh.
DATA = Path(os.getenv("FORGE_DATA_DIR") or Path(__file__).resolve().parent.parent / "data" / "processed")
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
BABY = re.compile(r"\b(bab(y|ies)|toddlers?|infants?|first food|weaning|\d+\s*months?)\b", re.I)
SWEET_OR_DRINK = re.compile(
    r"\b(trifle|kheer|halwa|ladoo|laddu|cake|cupcake|pudding|payasam|custard|ice ?cream|kulfi|mousse|barfi|burfi|"
    r"brownies?|cookies?|tarts?|fudge|truffles?|cheesecake|muffins?|parfait|souffle|panna cotta|shrikhand|rasmalai|"
    r"jamun|jalebi|smoothie|milkshake|shake|lassi|juice|mocktail|cocktail|sherbet|sharbat|tea|coffee|cooler|punch|"
    r"lemonade|sorbet|granita|popsicle|candy|toffee|chikki|meetha|sweet)\b", re.I)
SWEET_WANTED = re.compile(r"\b(sweet|dessert|mithai|cake|kheer|halwa|ladoo|laddu|barfi|cookies?|ice ?cream|"
                          r"pudding|chocolate|smoothie|shake|lassi|juice|drink|beverage|tea|coffee|festive)\b", re.I)
SWEET_ONLY = re.compile(r"\b(sweet|dessert|mithai|meetha)\b", re.I)
DRINK_WANTED = re.compile(r"\b(drinks?|beverages?|smoothie|shake|lassi|juice|mocktail|sharbat|sherbet|cooler|tea|coffee)\b", re.I)
# Hindu festivals where the festive table is usually vegetarian. A soft preference, never a locked diet: naming meat
# ("diwali mutton biryani") turns it off.
VEG_FESTIVAL = re.compile(r"\b(diwali|deepavali|navratri|navaratri|janmashtami|ganesh chaturthi|ganpati|ganeshotsav|holi|"
                          r"raksha ?bandhan|rakhi|karva ?chauth|dussehra|dasara|pongal|sankranti|ugadi|gudi padwa|teej|"
                          r"chhath|onam|vishu|dhanteras|bhai dooj|diwali)\b|दिवाली|दीवाली|नवरात्रि|होली|जन्माष्टमी", re.I)
MEAT_WORDS = re.compile(r"\b(chicken|mutton|lamb|goat|meat|keema|fish|prawns?|shrimp|egg|eggs|non[- ]?veg|beef|pork|"
                        r"crab|seafood|gosht|murg|machli|macchi|anda)\b", re.I)
OILY = re.compile(r"\b(fr(y|ied|ies|itters?)|pakor[ae]s?|pakod[ae]s?|bhajj?(i|ia|iya)s?|vad[ae]s?|crispy|tempura|"
                  r"chilli oil|red oil|deep)\b", re.I)
NOT_OILY = re.compile(r"\b(not|less|without|no|kam|na|nahi)\s+(too\s+|so\s+|very\s+)?(oily|greasy|fried|tel\w*)\b|"
                      r"\b(low|less|no|kam)[- ]oil\b", re.I)
# Main ingredients people name on their own to mean "a dish with this" (proteins and everyday vegetables).
MAIN_INGREDIENTS = {"paneer", "chickpeas", "kidney_beans", "tofu", "soy_chunks", "egg", "chicken", "mutton", "fish",
                    "shellfish", "potato", "cauliflower", "cabbage", "spinach", "okra", "brinjal", "mushroom", "peas",
                    "capsicum", "carrot", "bottle_gourd", "pumpkin", "beans", "raw_banana", "corn", "methi", "beetroot",
                    "sweet_potato", "drumstick", "bitter_gourd", "ridge_gourd", "colocasia", "yam"}
KID_HOT_IDS = {"green_chilli", "curry_paste", "chilli_sauce", "chilli"}
KID_HOT_NAME = re.compile(r"\b(chill?i|chilly|mirchi|schezwan|szechuan|sichuan|thai|peri[- ]?peri|vindaloo|kolhapuri|"
                          r"chettinad|teekh[ae]|tikha|spicy|hot|jalape[nñ]o|laal maas|pepper|nasi lemak|sambal)\b", re.I)
CHILLED = re.compile(r"\b(chilled|cold|iced|gazpacho)\b", re.I)
WARM_WANTED = re.compile(r"\b(warm|hot soup|warming|for the cold|winter|rainy|monsoon|baarish)\b", re.I)
CONDIMENT = re.compile(r"\b(salsa|chutney|dip|raita|pickle|achar|sauce|dressing|thokku|podi|relish|jam|spread|pachadi)\b", re.I)


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
    about = f" Also called {', '.join(r['aka'])}." if r.get("aka") else ""
    about += f" {r['description']}" if r.get("description") else ""
    return f"{r['name']}.{about} {r['cuisine']} {r['course']}. Ingredients: {names}. {r['instructions'][:300]}"


def load_recipes(data_dir: Path = DATA) -> list[dict]:
    """Recipes plus their search descriptions (scripts/enrich.py). Descriptions only feed ranking: diet checks
    use the ingredient lines."""
    recipes = [json.loads(l) for l in (data_dir / "recipes.jsonl").read_text(encoding="utf-8").splitlines()]
    desc_path = data_dir / "descriptions.jsonl"
    if desc_path.exists():
        desc = {d["id"]: d for d in map(json.loads, desc_path.read_text(encoding="utf-8").splitlines()) if d}
        for r in recipes:
            if r["id"] in desc:
                r["aka"], r["description"] = desc[r["id"]]["aka"], desc[r["id"]]["description"]
    tags_path = data_dir / "tags.jsonl"
    if tags_path.exists():
        tags = {d["id"]: d for d in map(json.loads, tags_path.read_text(encoding="utf-8").splitlines()) if d}
        for r in recipes:
            if r["id"] in tags:
                d = tags[r["id"]]
                r["qualities"], r["spice_level"], r["dish_type"] = d["qualities"], d["spice"], d["dish_type"]
    return recipes


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
            if src.exists() and index.get("source_lines") != source_signature(data_dir):
                index = None
        if index is None:
            index = build_index(self.onto, data_dir)
        self.recipes: list[dict] = index["recipes"]
        self.bm25 = BM25(index["postings"], len(self.recipes))
        self.lineages = index["lineages"]
        self.tags = index["tags"]
        self.name_tokens = index["name_tokens"]
        self.robot = index.get("robot") or [robot_fit(r) for r in self.recipes]
        for r, fit in zip(self.recipes, self.robot):
            r["robot_fit"] = fit
        self.aka_tokens = index.get("aka_tokens") or [frozenset(tokenize(" ".join(r.get("aka", [])))) for r in self.recipes]
        self.desc_tokens = index.get("desc_tokens") or [frozenset(tokenize(r.get("description", ""))) for r in self.recipes]
        self.name_ids = index["name_ids"]
        self._vocab = index.get("vocab")
        self.by_id = {r["id"]: i for i, r in enumerate(self.recipes)}
        self.is_baby = [bool(BABY.search(r["name"])) for r in self.recipes]
        self.is_condiment = [bool(CONDIMENT.search(r["name"])) for r in self.recipes]
        self.is_sweet = [r["course"] == "Dessert" or bool(SWEET_OR_DRINK.search(r["name"])) for r in self.recipes]
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
        # Convex 0.5 vs RRF on data/eval/golden.yaml: nDCG@10 0.690 vs 0.668; no worse on either half of the set.
        self.fusion = os.getenv("FORGE_FUSION", "convex")
        self.alpha = float(os.getenv("FORGE_ALPHA", "0.5"))
        # Re-ranking weights (tuned on data/eval/golden.yaml; see scripts/retrieval_eval.py).
        # Alternate names count as names, description words earn a smaller bonus, traits read the description:
        # nDCG@10 0.739 -> ~0.746 and MRR 0.868 -> ~0.886, better on both halves of the set.
        self.w_name, self.w_desc = 0.5, 0.3
        self.aka_as_name, self.desc_features = True, True

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
    def speller(self) -> Speller:
        vocab = self._vocab or build_vocab(self.onto, self.recipes)
        return Speller(vocab, keep=KEEP_WORDS)

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
            if not reason and c.robot_only and self.robot[i]["fit"] != "robot":
                reason = ("Nosh One cookable", self.robot[i]["reason"])
            if reason:
                removed[reason[0]] += 1
                blocked_reason[i] = f"{reason[0]}: {reason[1]}"
            else:
                keep.append(i)
        return keep, removed, blocked_reason

    def _only_ingredients(self, c: Constraints) -> bool:
        """True when the request names ingredients and nothing else (no dish, course or style words)."""
        names = set()
        for i in [*c.include, *c.conflicts, *c.swaps, *c.exclude]:
            names |= set(tokenize(self.onto.label(i))) | set(i.split("_"))
        rest = {w for w in tokenize(c.text or "")
                if not ({w, w[:-1], w[:-2]} & names)}  # tomatoes -> tomato
        return not rest

    def _occasion_filters(self, c: Constraints, cand: list[int], want: set, trace: dict) -> list[int]:
        """Soft, common-sense narrowing for occasion and audience words. Each step applies only if 5+ dishes
        survive it, and none of them is a locked restriction."""
        text = c.text or ""
        said = " ".join([text, *c.understood])

        def narrow(keep, label):
            nonlocal cand
            kept = [i for i in cand if keep(i)]
            if len(kept) >= 5:
                cand = kept
                trace.setdefault("occasion", []).append(label)

        if VEG_FESTIVAL.search(said) and not MEAT_WORDS.search(said) and not c.prefer_non_veg:
            narrow(lambda i: self.recipes[i]["verdicts"]["vegetarian"]["status"] != "block", "festival: vegetarian")
        if "kid_friendly" in want and not BABY.search(said):
            # Kids: no baby purees, nothing with green chillies, curry pastes or chilli sauces, and no dish whose name
            # says it's hot (chaat-style or Thai-curry "mild" still bites).
            narrow(lambda i: not self.is_baby[i] and not KID_HOT_IDS & set(self.recipes[i]["ingredient_ids"])
                   and not KID_HOT_NAME.search(self.recipes[i]["name"]), "kids: no baby food or chilli heat")
        if NOT_OILY.search(said):
            narrow(lambda i: not OILY.search(self.recipes[i]["name"]) and not OILY.search(self.recipes[i].get("description", "")),
                   "not oily: no fried dishes")
        if WARM_WANTED.search(said):
            narrow(lambda i: not CHILLED.search(self.recipes[i]["name"]), "warm: no chilled dishes")
        return cand

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
        if self.desc_features and r.get("description"):
            # Traits stated in the dish description count too ("deep-fried, crisp" -> crispy).
            name += " " + r["description"].lower()
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
        if course in ("lunch", "dinner") and self.is_sweet[i]:
            return False
        if course == "dessert":
            kind = self.recipes[i].get("dish_type")
            if kind:  # tagged: a real dessert (or sweet drink), not "Sweet Corn Sundal"
                return kind in ("dessert", "drink")
            return self.is_sweet[i] or course in self.recipes[i]["course_tags"]
        return course in self.recipes[i]["course_tags"]

    def _cuisine_ok(self, i: int, cuisine: str) -> bool:
        labels = CUISINE_GROUPS.get(cuisine, [])
        return self.recipes[i]["cuisine"] in labels

    def query_text(self, c: Constraints) -> str:
        parts = [c.text]
        parts += [self.onto.label(i) for i in c.include if self.onto.label(i) not in c.text]
        # Preference words the quality tags already handle ("healthy", "comfort") would only match dish *names*
        # ("Healthy Bharwa Karela") here, so they stay out of the search text once tags exist.
        tagged = bool(self.recipes and self.recipes[0].get("qualities") is not None)
        parts += sorted(a.replace("_", " ") for a in c.attributes if not (tagged and a in ATTRIBUTE_QUALITIES))
        if c.cuisine:
            parts.append(c.cuisine)
        if c.prefer_non_veg:
            parts.append("chicken mutton fish egg")
        if parts[0]:
            # Spelling variants the corpus never uses ("biriyani", "sambhar", "chatni") -> the corpus spelling.
            parts[0] = " ".join(w if w in KEEP_WORDS else self.speller.correct(w) for w in parts[0].split())
            # English descriptions -> the Indian dish names they describe ("puffed fried bread" -> puri).
            extra = [names for rx, names in DISH_SYNONYMS if rx.search(parts[0])]
            if extra:
                parts.append(" ".join(dict.fromkeys(" ".join(extra).split())))
        if not any(p.strip() for p in parts):
            # Course is a filter, not a search word ("dinner" would match "Dinner Rolls").
            parts += [self.onto.diets[d]["label"] for d in sorted(c.diets)] + [
                {"breakfast": "breakfast", "dessert": "sweet dessert", "snack": "snack"}.get(c.course or "", "")
                + " home style indian recipe"]
        return " ".join(p for p in parts if p).strip()

    # ---------- main entry ----------
    def search(self, c: Constraints, k: int = 5, exclude_ids: set[int] | None = None,
               anchor_id: int | None = None, profile=None) -> SearchOutput:
        exclude_ids = set(exclude_ids or ())
        if profile is not None:
            exclude_ids |= profile.disliked_ids
            if c.intents & {"different", "usual"}:
                exclude_ids |= profile.cooked_ids
        keep, removed, blocked_reason = self.hard_filter(c)
        trace = {"corpus": len(self.recipes), "removed_by_constraint": dict(removed), "after_hard_filter": len(keep)}
        relaxed: list[str] = []

        cand = [i for i in keep if self.recipes[i]["id"] not in exclude_ids]
        if not COMPONENT_QUERY.search(c.text):
            cand = [i for i in cand if not self.recipes[i].get("is_component")]
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
        if c.prefer_non_veg:
            narrowed = [i for i in cand if self.tags[i] & MEAT_TAGS]
            if len(narrowed) >= 3:
                cand = narrowed
        if profile is not None and not profile.empty and c.intents & {"different", "usual"} and not c.course:
            usual = {t for e in profile.events if e.action != "dislike"
                     for t in self.recipes[self.by_id[e.id]]["course_tags"]}
            narrowed = [i for i in cand if usual & set(self.recipes[i]["course_tags"])]
            if len(narrowed) >= 5:
                cand = narrowed
        if not (c.text.strip() or c.include or c.course or c.cuisine or c.prefer_non_veg):
            # Nothing to rank by ("no mushrooms", "within an hour"): offer varied main dishes, not snacks or sweets.
            mains = [i for i in cand if {"lunch", "dinner"} & set(self.recipes[i]["course_tags"]) and not self.is_sweet[i]
                     and not self.is_condiment[i] and not self.is_baby[i] and not self.recipes[i].get("is_component")]
            if len(mains) >= 10:
                cand = mains
        want, avoid, spice_ok = wanted_qualities(c)
        if want or avoid or spice_ok:
            # Vague requests ("something light", "for my kids", "party starters"): keep dishes tagged with the wanted
            # qualities (scripts/enrich_tags.py) when enough exist, so results never thin out to nothing.
            def fits(i):
                r = self.recipes[i]
                q = set(r.get("qualities", []))
                return ((not want or q & want) and not (q & avoid)
                        and (not spice_ok or r.get("spice_level", "medium") in spice_ok))
            narrowed = [i for i in cand if fits(i)]
            if len(narrowed) >= 5:
                cand = narrowed
            trace["qualities"] = sorted(want)
        cand = self._occasion_filters(c, cand, want, trace)
        meal = MEAL_TYPES.get(c.course or "")
        if not meal and SWEET_ONLY.search(c.text or "") and not DRINK_WANTED.search(c.text or ""):
            meal = {"dessert", "snack"}  # "something sweet" is a sweet dish, not a mocktail
        if not meal and want and not SWEET_WANTED.search(c.text or "") and not CONDIMENT.search(c.text or ""):
            # A vague request with no course ("comfort food for a rainy day", "halka khana") means a dish to eat,
            # not a drink, dessert or chutney.
            meal = {"main", "side", "soup", "salad", "breakfast", "snack", "bread"}
        # One main ingredient named on its own ("chicken", "paneer"; under Jain, chicken -> paneer): dishes built on
        # it. Dish-like ingredients (poha, roti, dhokla) are searched as the dish itself.
        asked = (set(c.include) - set(c.swaps.values())) | set(c.conflicts)
        bare = len(asked) == 1 and asked <= MAIN_INGREDIENTS and len(tokenize(c.text or "")) <= 2
        if not meal and (c.pantry_mode or bare) and c.include and self._only_ingredients(c):
            # "I have paneer, what can I cook?": a proper dish built on it, not a fruit platter or a drink.
            meal = {"main", "side", "soup", "breakfast"}
        if meal:
            narrowed = [i for i in cand if self.recipes[i].get("dish_type", "main") in meal]
            if len(narrowed) >= 5:
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
                    ranked = self._rank(c, pool, qtext, qvec, anchor_id, k=1, trace={}, profile=profile)
                    closest = ranked[0] if ranked else None
            return SearchOutput([], trace, blocked_examples, closest, relaxed)

        results = self._rank(c, cand, qtext, qvec, anchor_id, k, trace, profile=profile)
        # "5 minute biryani": if the time limit leaves no dish of the named kind, say so and offer the quickest one.
        content = {t for t in tokenize(c.text) if len(t) > 3}
        include = set(c.include)
        if c.max_time and content and results and not any(
                content & self.name_tokens[self.by_id[r.recipe["id"]]] or include & self.lineages[self.by_id[r.recipe["id"]]]
                for r in results):
            keep2, _, _ = self.hard_filter(c, skip_time=True)
            named = [i for i in keep2 if content & self.name_tokens[i] and self.recipes[i]["id"] not in exclude_ids
                     and self.recipes[i]["total_min"] > 0 and not self.recipes[i].get("is_component")
                     and not self.is_condiment[i] and not self.is_baby[i]
                     and (not c.course or self._course_ok(i, c.course))
                     and (not self.is_sweet[i] or c.course == "dessert" or SWEET_WANTED.search(qtext))]
            if named:
                best = min(named, key=lambda i: self.recipes[i]["total_min"])
                closest = self._explain(best, 0.0, self.is_flagged(best, c), c, profile)
                return SearchOutput([], trace, blocked_examples, closest, relaxed)
        return SearchOutput(results, trace, blocked_examples, None, relaxed)

    def _rank(self, c, cand, qtext, qvec, anchor_id, k, trace, profile=None) -> list[Result]:
        scored = self.score_pool(c, cand, qtext, qvec, anchor_id, trace, profile=profile)
        chosen = self._mmr(scored, k)
        return [self._explain(i, s, flags, c, profile) for s, i, flags, cover in chosen]

    def score_pool(self, c, cand, qtext, qvec, anchor_id, trace, pool_size: int = 30, profile=None):
        idx = np.array(cand)
        qtok = tokenize(qtext)
        lex = self.bm25.scores(qtok, idx)
        dense = self.emb[idx] @ qvec if qvec is not None else None
        novelty = profile is not None and not profile.empty and bool(c.intents & {"different", "usual"})
        taste = None
        if profile is not None and not profile.empty and profile.liked_vec is not None and len(idx) and not novelty:
            taste = self.emb[idx] @ profile.liked_vec
        rrf = Counter()
        if self.fusion == "convex" and dense is not None:
            # Weighted sum of min-max normalised scores (tuned on the golden set; beats RRF once labels exist).
            fused = self.alpha * _norm(lex) + (1 - self.alpha) * _norm(dense)
            if taste is not None:
                fused = fused + 0.15 * _norm(taste)
            for i, v in zip(idx, fused):
                rrf[int(i)] = float(v)
        else:
            for r, i in enumerate(idx[np.argsort(-lex)][:200]):
                rrf[int(i)] += 1 / (60 + r)
            if dense is not None:
                for r, i in enumerate(idx[np.argsort(-dense)][:200]):
                    rrf[int(i)] += 1 / (60 + r)
            if taste is not None:
                for r, i in enumerate(idx[np.argsort(-taste)][:100]):
                    rrf[int(i)] += 0.5 / (60 + r)
        pool = [i for i, _ in rrf.most_common(pool_size)] or list(idx[:pool_size])
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
        want_q = wanted_qualities(c)[0]
        include = set(c.include)
        content = {t for t in tokenize(c.text) if len(t) > 2}
        for j, i in enumerate(pool):
            feats = self.features(i)
            s = float(rel[j])
            cover = len(include & self.lineages[i]) / len(include) if include else 0.0
            s += (0.9 if c.pantry_mode else 0.5) * cover
            if content:
                names = self.name_tokens[i] | self.aka_tokens[i] if self.aka_as_name else self.name_tokens[i]
                s += self.w_name * len(content & names) / len(content)
                if self.w_desc:
                    s += self.w_desc * len(content & self.desc_tokens[i]) / len(content)
            if include:
                s += 0.4 * len(include & self.name_ids[i]) / len(include)
            for a in c.attributes:
                s += 0.3 * feats.get(a, 0.0)
            if c.prefer_non_veg and self.tags[i] & MEAT_TAGS:
                s += 0.4
            if want_q:
                s += 0.4 * len(want_q & set(self.recipes[i].get("qualities", []))) / len(want_q)
            if c.cuisine and self._cuisine_ok(i, c.cuisine):
                s += 0.15
            if anchor_vec is not None:
                s += 0.5 * float(self.emb[i] @ anchor_vec)
                if self.recipes[i]["id"] == anchor_id:
                    s -= 0.4
            if self.recipes[i].get("is_component") and not COMPONENT_QUERY.search(qtext):
                s -= 0.8
            if self.is_baby[i] and not BABY.search(qtext):
                s -= 0.8
            if self.is_condiment[i] and not CONDIMENT.search(qtext):
                s -= 0.6
            if self.is_sweet[i] and c.course != "dessert" and not SWEET_WANTED.search(qtext):
                s -= 0.5
            flags = self.is_flagged(i, c)
            if flags:
                s -= 0.35
            if profile is not None and not profile.empty and self.emb is not None:
                rid = self.recipes[i]["id"]
                sim = float(self.emb[i] @ profile.liked_vec) if profile.liked_vec is not None else 0.0
                if novelty:
                    s += 0.6 * (1 - sim)
                    if self.recipes[i]["cuisine"] not in profile.cuisines:
                        s += 0.25
                else:
                    s += 0.35 * sim
                    if rid in profile.recent_ids:
                        s -= 0.5
                if profile.disliked_vec is not None:
                    s -= 0.4 * max(0.0, float(self.emb[i] @ profile.disliked_vec))
            scored.append((s, i, flags, cover))
        scored.sort(reverse=True)
        return scored

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

    def _explain(self, i: int, score: float, flags: list[str], c: Constraints, profile=None) -> Result:
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
                why.append(f"serves {r['servings']} · make {factor:g}× for {c.servings} people"
                           if factor == round(factor, 1) else
                           f"serves {r['servings']} · make about {factor:.1f}× for {c.servings} people")
            else:
                why.append(f"serves {r['servings']}")
        if c.robot_only and c.servings and robot_batches(c.servings) > 1:
            why.append(f"Nosh One: cook in {robot_batches(c.servings)} batches for {c.servings} people")
        if profile is not None and not profile.empty:
            if c.intents & {"different", "usual"}:
                if r["cuisine"] not in profile.cuisines:
                    why.append(f"new cuisine for you ({r['cuisine']})")
                new = [self.onto.label(x) for x in r["ingredient_ids"] if x not in profile.ingredients
                       and x not in STAPLES][:3]
                if new:
                    why.append("ingredients you haven't cooked with: " + ", ".join(new))
            elif profile.liked_vec is not None and self.emb is not None and float(self.emb[i] @ profile.liked_vec) > 0.8:
                why.append("close to dishes you've liked")
        return Result(recipe=r, score=round(score, 3), checks=checks, why=why, flagged=bool(flags))

    def _blocked_examples(self, qtext, qvec, blocked_reason, c, n: int = 3) -> list[dict]:
        if not blocked_reason or not (c.diets or c.exclude or c.robot_only):
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


def build_vocab(onto: Ontology, recipes: list[dict]) -> Counter:
    """Words the corpus actually uses (names, ingredient lines, ontology aliases): the speller's targets."""
    vocab = Counter()
    for r in recipes:
        vocab.update(TOKEN.findall(r["name"].lower()) * 3)
        for x in r["ingredients"]:
            vocab.update(TOKEN.findall(x["name"].lower()))
    for ing in onto.ingredients.values():
        for a in ing.aliases:
            vocab.update(TOKEN.findall(a.lower()))
    return Counter({w: n for w, n in vocab.items() if w.isalpha()})


def count_lines(path: Path) -> int:
    """Line-ending-agnostic freshness signature (git may convert LF/CRLF between platforms)."""
    return path.read_bytes().count(b"\n")


def source_signature(data_dir: Path = DATA) -> tuple[int, ...]:
    """Recipes and descriptions line counts: the index is rebuilt when either changes."""
    desc, tags = data_dir / "descriptions.jsonl", data_dir / "tags.jsonl"
    return (count_lines(data_dir / "recipes.jsonl"), count_lines(desc) if desc.exists() else 0,
            count_lines(tags) if tags.exists() else 0)


def build_index(onto: Ontology, data_dir: Path = DATA, write: bool = False) -> dict:
    recipes = load_recipes(data_dir)
    union = lambda sets: frozenset().union(*sets) if sets else frozenset()
    index = {
        "source_lines": source_signature(data_dir),
        "recipes": recipes,
        "postings": BM25.build([tokenize((r["name"] + " ") * 3 + (" ".join(r.get("aka", [])) + " ") * 2 + doc_text(r))
                                for r in recipes]).postings,
        "lineages": [union([onto.lineage(i) for i in r["ingredient_ids"]]) for r in recipes],
        "tags": [union([onto.effective_tags(i) for i in r["ingredient_ids"]]) for r in recipes],
        "name_tokens": [frozenset(tokenize(r["name"])) for r in recipes],
        "aka_tokens": [frozenset(tokenize(" ".join(r.get("aka", [])))) for r in recipes],
        "robot": [robot_fit(r) for r in recipes],
        "desc_tokens": [frozenset(tokenize(r.get("description", ""))) for r in recipes],
        "name_ids": [frozenset(m.ingredient for m in onto.match(r["name"])) for r in recipes],
        "vocab": build_vocab(onto, recipes),
    }
    if write:
        (data_dir / "index.pkl").write_bytes(pickle.dumps(index, protocol=5))
        print("index.pkl", len(recipes), "recipes")
    return index


def build_embeddings(profile: str, data_dir: Path = DATA) -> None:
    recipes = load_recipes(data_dir)
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
