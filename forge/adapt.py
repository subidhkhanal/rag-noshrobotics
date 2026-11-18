"""Adapt a recipe to the active constraints by swapping offending ingredients, then re-validating."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .ontology import ONTOLOGY_DIR, Ontology
from .query import Constraints

MAX_SWAPS = 4


@lru_cache(maxsize=1)
def load_substitutions(path: Path = ONTOLOGY_DIR / "substitutions.yaml") -> dict[str, list[dict]]:
    subs = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {k: [{"to": list(o["to"]), "how": o["how"]} for o in v] for k, v in subs.items()}


@dataclass
class Swap:
    line: str
    replace: str
    with_: list[str]
    how: str


@dataclass
class Adaptation:
    adaptable: bool
    swaps: list[Swap] = field(default_factory=list)
    safer: list[Swap] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    remaining_flags: list[str] = field(default_factory=list)
    main_swapped: list[str] = field(default_factory=list)

    def to_dict(self, onto: Ontology) -> dict:
        fmt = lambda s: {"line": s.line, "replace": onto.label(s.replace), "with": [onto.label(w) for w in s.with_],
                         "how": s.how}
        return {
            "adaptable": self.adaptable,
            "swaps": [fmt(s) for s in self.swaps],
            "safer": [fmt(s) for s in self.safer],
            "unresolved": self.unresolved,
            "remaining_flags": self.remaining_flags,
            "main_swapped": [onto.label(x) for x in self.main_swapped],
        }


def _blocking_ids(onto: Ontology, ids: list[str], c: Constraints) -> list[str]:
    block_tags = {t for d in c.diets for t in onto.diets[d].get("block", [])}
    return [i for i in ids if onto.effective_tags(i) & block_tags or any(e in onto.lineage(i) for e in c.exclude)]


def _violations(onto: Ontology, ingredients: list[dict], c: Constraints) -> tuple[list[str], list[str]]:
    blocks, flags = [], []
    for d in sorted(c.diets):
        v = onto.diet_verdict(d, ingredients)
        (blocks if v.status == "block" else flags if v.status == "flag" else []).extend(v.reasons)
    for e in sorted(c.exclude):
        v = onto.exclusion_verdict(e, ingredients)
        (blocks if v.status == "block" else flags if v.status == "flag" else []).extend(v.reasons)
    return blocks, flags


def _candidates(onto: Ontology, iid: str, subs: dict) -> list[dict]:
    """Swaps for the ingredient itself, else for its family (e.g. shallot -> onion swaps)."""
    seen, out = set(), []
    for src in [iid, onto.ingredients[iid].family]:
        for opt in subs.get(src or "", []):
            key = tuple(opt["to"])
            if key not in seen:
                seen.add(key)
                out.append(opt)
    return out


def adapt(onto: Ontology, recipe: dict, c: Constraints) -> Adaptation:
    if not (c.diets or c.exclude):
        return Adaptation(adaptable=False)
    subs = load_substitutions()
    block_tags = {t for d in c.diets for t in onto.diets[d].get("block", [])}
    new_ings = [dict(it, ids=list(it["ids"])) for it in recipe["ingredients"]]
    swaps, unresolved = [], []
    for it in new_ings:
        for bad in _blocking_ids(onto, it["ids"], c):
            chosen = None
            for opt in _candidates(onto, bad, subs):
                ok = all(not onto.effective_tags(t) & block_tags and not any(e in onto.lineage(t) for e in c.exclude)
                         for t in opt["to"])
                if ok:
                    chosen = opt
                    break
            if chosen is None:
                unresolved.append(f"{it['line']} ({onto.label(bad)})")
                continue
            it["ids"] = [x for x in it["ids"] if x != bad] + [t for t in chosen["to"] if t not in it["ids"]]
            swaps.append(Swap(it["line"], bad, chosen["to"], chosen["how"]))
    if unresolved or len(swaps) > MAX_SWAPS:
        return Adaptation(adaptable=False, swaps=swaps, unresolved=unresolved
                          or [f"needs {len(swaps)} swaps — too many to stay the same dish"])
    safer = []
    for it in new_ings:
        for risky in [i for i in it["ids"] if onto.effective_may(i) & block_tags]:
            for opt in _candidates(onto, risky, subs):
                if all(not (onto.effective_tags(t) | onto.effective_may(t)) & block_tags for t in opt["to"]):
                    it["ids"] = [x for x in it["ids"] if x != risky] + [t for t in opt["to"] if t not in it["ids"]]
                    safer.append(Swap(it["line"], risky, opt["to"], opt["how"]))
                    break
    blocks, flags = _violations(onto, new_ings, c)
    if blocks:
        return Adaptation(adaptable=False, swaps=swaps, unresolved=blocks)
    in_name = {m.ingredient for m in onto.match(recipe["name"])}
    main = list(dict.fromkeys(s.replace for s in swaps
                              if s.replace in in_name or onto.ingredients[s.replace].family in in_name))
    return Adaptation(adaptable=bool(swaps or safer), swaps=swaps, safer=safer, remaining_flags=flags[:3],
                      main_swapped=main)
