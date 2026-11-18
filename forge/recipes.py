"""Turn raw dataset rows into normalized recipe records."""
from __future__ import annotations

import re

from .ontology import Ontology

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_QTY = re.compile(r"^[\s\d/.\-+½¼¾⅓⅔]+(to|or|\+)?[\s\d/.\-]*", re.I)
_UNIT = re.compile(
    r"^(cups?|tablespoons?|teaspoons?|tbsp|tsp|grams?|gms?|g|kg|ml|litres?|liters?|inch(es)?|pinch|"
    r"sprigs?|bunch(es)?|cloves?|pieces?|nos?|handful|cans?|packets?|small|medium|large|big|few|"
    r"stalks?|sheets?|drops?|slices?|fl oz|oz|lbs?|pounds?)\b\.?\s*",
    re.I,
)
COURSE_MAP = {
    "breakfast": ["south indian breakfast", "north indian breakfast", "indian breakfast", "world breakfast", "brunch"],
    "lunch": ["lunch", "main course", "one pot dish"],
    "dinner": ["dinner", "main course", "one pot dish"],
    "snack": ["snack", "appetizer"],
    "dessert": ["dessert"],
    "side": ["side dish"],
}


def is_english(text) -> bool:
    return isinstance(text, str) and bool(text.strip()) and not _DEVANAGARI.search(text)


def split_ingredients(raw: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in raw or "":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def display_name(line: str) -> str:
    name = line.split(" - ")[0]
    name = _QTY.sub("", name).strip()
    name = _UNIT.sub("", name).strip()
    return re.sub(r"^of\s+", "", name, flags=re.I) or line


def parse_ingredients(onto: Ontology, raw: str) -> list[dict]:
    out = []
    for line in split_ingredients(raw):
        ids = [m.ingredient for m in onto.match(line)]
        ids = [i for i in ids if "non_food" not in onto.ingredients[i].tags]
        if not ids and any("non_food" in onto.ingredients[m.ingredient].tags for m in onto.match(line)):
            continue
        out.append({"line": line, "name": display_name(line), "ids": ids})
    return out


def course_tags(course: str) -> list[str]:
    c = (course or "").strip().lower()
    return sorted(k for k, v in COURSE_MAP.items() if c in v)
