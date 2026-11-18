"""Ingredient ontology: alias matching, tag inheritance and diet verdicts."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "data" / "ontology"
# Tags that define an ingredient class: excluding egg/fish/peanut/... excludes everything carrying that tag.
CLASS_DEFINING_TAGS = {"egg", "fish", "shellfish", "peanut", "sesame", "soy", "fungi", "honey", "gelatin", "alcohol"}

_TOKEN_SPLIT = re.compile(r"[^0-9a-zऀ-ॿ]+")
_CONSONANT_Y = re.compile(r"(?<=[^aeiou])y$")


def _stem(tok: str) -> str:
    if len(tok) <= 3 or not tok.isascii():
        return tok
    if tok.endswith("ies"):
        return tok[:-3] + "i"
    if tok.endswith(("ches", "shes", "sses", "xes", "zes", "oes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return _CONSONANT_Y.sub("i", tok)


_SPELLING = {
    "chili": "chilli", "chile": "chilli", "asafetida": "asafoetida", "channa": "chana",
    "daal": "dal", "dhal": "dal", "parseley": "parsley", "anis": "anise", "anee": "anise",
    "kenua": "quinoa", "kinua": "quinoa", "kisami": "kishmish", "kisamis": "kishmish", "kisamey": "kishmish",
    "jalepino": "jalapeno", "gleppino": "jalapeno", "kasoori": "kasuri", "habenaro": "habanero",
}


def tokenize(text: str) -> list[str]:
    toks = [_stem(t) for t in _TOKEN_SPLIT.split(text.lower()) if t]
    return [_SPELLING.get(t, t) for t in toks]


@dataclass
class Ingredient:
    id: str
    aliases: list[str]
    tags: set[str] = field(default_factory=set)
    may: set[str] = field(default_factory=set)
    family: str | None = None
    contains: list[str] = field(default_factory=list)
    note: str | None = None


@dataclass
class Verdict:
    status: str  # "pass" | "flag" | "block"
    reasons: list[str] = field(default_factory=list)


@dataclass
class Match:
    ingredient: str
    alias: str


class Ontology:
    def __init__(self, ingredients: dict, diets: dict):
        self.ingredients: dict[str, Ingredient] = {}
        for iid, spec in ingredients.items():
            self.ingredients[iid] = Ingredient(
                id=iid,
                aliases=[str(a) for a in spec.get("aliases", [])],
                tags=set(spec.get("tags", [])),
                may=set(spec.get("may", [])),
                family=spec.get("family"),
                contains=list(spec.get("contains", [])),
                note=spec.get("note"),
            )
        self.diets = diets
        self._validate()
        self._alias_index: dict[str, dict[tuple, tuple[str, str]]] = {}
        for ing in self.ingredients.values():
            for alias in ing.aliases + [ing.id.replace("_", " ")]:
                toks = tuple(tokenize(alias))
                if toks:
                    self._alias_index.setdefault(toks[0], {}).setdefault(toks, (ing.id, alias))
        self._max_alias_len = max(len(k) for d in self._alias_index.values() for k in d)

    def _validate(self) -> None:
        for ing in self.ingredients.values():
            for ref in ing.contains + ([ing.family] if ing.family else []):
                if ref not in self.ingredients:
                    raise ValueError(f"{ing.id} references unknown ingredient {ref!r}")
        known_tags = {t for i in self.ingredients.values() for t in i.tags | i.may}
        for name, rule in self.diets.items():
            for t in rule.get("block", []) + rule.get("flag", []):
                if t not in known_tags:
                    raise ValueError(f"diet {name} uses tag {t!r} that no ingredient carries")

    # ---------- matching ----------
    def match(self, text: str) -> list[Match]:
        """Longest-phrase-first token matching; returns every ingredient mentioned."""
        toks = tokenize(text)
        out: list[Match] = []
        i = 0
        while i < len(toks):
            cands = self._alias_index.get(toks[i])
            hit = None
            if cands:
                for n in range(min(self._max_alias_len, len(toks) - i), 0, -1):
                    key = tuple(toks[i : i + n])
                    if key in cands:
                        hit = (n, cands[key])
                        break
            if hit:
                n, (iid, alias) = hit
                if not any(m.ingredient == iid for m in out):
                    out.append(Match(iid, alias))
                i += n
            else:
                i += 1
        return out

    # ---------- inheritance ----------
    @lru_cache(maxsize=None)
    def effective_tags(self, iid: str) -> frozenset[str]:
        ing = self.ingredients[iid]
        tags = set(ing.tags)
        for c in ing.contains:
            tags |= self.effective_tags(c)
        return frozenset(tags)

    @lru_cache(maxsize=None)
    def effective_may(self, iid: str) -> frozenset[str]:
        ing = self.ingredients[iid]
        may = set(ing.may)
        for c in ing.contains:
            may |= self.effective_may(c)
        return frozenset(may - self.effective_tags(iid))

    @lru_cache(maxsize=None)
    def lineage(self, iid: str) -> frozenset[str]:
        """The ingredient itself plus every family ancestor and component."""
        ing = self.ingredients[iid]
        out = {iid}
        if ing.family:
            out |= self.lineage(ing.family)
        for c in ing.contains:
            out |= self.lineage(c)
        return frozenset(out)

    def label(self, iid: str) -> str:
        return iid.replace("_", " ")

    # ---------- verdicts ----------
    def diet_verdict(self, diet: str, ingredients: list[dict]) -> Verdict:
        """ingredients: [{"line": str, "ids": [str]}] as produced by parse_recipe."""
        rule = self.diets[diet]
        block, flag = set(rule.get("block", [])), set(rule.get("flag", []))
        blocks, flags = [], []
        for item in ingredients:
            if not item["ids"]:
                if rule.get("unknown_flag"):
                    flags.append(f"unrecognised ingredient: {item['line']}")
                continue
            for iid in item["ids"]:
                hit = self.effective_tags(iid) & block
                if hit:
                    blocks.append(f"{item['line']} → {self.label(iid)} ({', '.join(sorted(hit))})")
                    continue
                may_hit = self.effective_may(iid) & block
                if may_hit:
                    note = self.ingredients[iid].note
                    flags.append(
                        f"{item['line']} may contain {', '.join(sorted(may_hit))}"
                        + (f" — {note}" if note else "")
                    )
                soft = self.effective_tags(iid) & flag
                if soft:
                    flags.append(f"{item['line']} → {', '.join(sorted(soft)).replace('_', ' ')}")
        if blocks:
            return Verdict("block", blocks)
        if flags:
            return Verdict("flag", flags)
        return Verdict("pass")

    def exclusion_verdict(self, excluded: str, ingredients: list[dict]) -> Verdict:
        """User-specified exclusion of one ingredient (and its family/compounds)."""
        risky_tags = self.effective_tags(excluded) - {"root_veg"}
        # "No egg" must also catch mayonnaise; "no fish" must catch fish sauce.
        class_tags = risky_tags if risky_tags and risky_tags <= CLASS_DEFINING_TAGS else set()
        blocks, flags = [], []
        for item in ingredients:
            for iid in item["ids"]:
                if excluded in self.lineage(iid) or self.effective_tags(iid) & class_tags:
                    blocks.append(f"{item['line']} → {self.label(excluded)}")
                elif self.effective_may(iid) & risky_tags:
                    flags.append(f"{item['line']} may contain {self.label(excluded)}")
        if blocks:
            return Verdict("block", blocks)
        if flags:
            return Verdict("flag", flags)
        return Verdict("pass")


@lru_cache(maxsize=1)
def load() -> Ontology:
    ingredients = yaml.safe_load((ONTOLOGY_DIR / "ingredients.yaml").read_text(encoding="utf-8"))
    diets = yaml.safe_load((ONTOLOGY_DIR / "diets.yaml").read_text(encoding="utf-8"))
    return Ontology(ingredients, diets)
