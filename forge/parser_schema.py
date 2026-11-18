"""Shared output schema for the query parser (rules vs. base LLM vs. QLoRA fine-tune)."""
from __future__ import annotations

import json

DIETS = ["dairy_free", "egg_free", "eggetarian", "fish_free", "gluten_free", "jain", "lactose_free", "no_onion_garlic",
         "nut_free", "peanut_free", "sattvic", "sesame_free", "shellfish_free", "soy_free", "tree_nut_free", "vegan",
         "vegetarian"]
COURSES = ["breakfast", "dessert", "dinner", "lunch", "side", "snack"]
ATTRIBUTES = ["comfort", "creamy", "crispy", "festive", "healthy", "high_protein", "mild", "quick", "spicy", "tangy"]
INTENTS = ["search", "plan", "history", "more", "similar", "different"]
FIELDS = ["diets", "exclude", "include", "max_time", "course", "cuisine", "attributes", "servings", "non_veg", "intent",
          "days"]

SYSTEM = (
    "You convert a cooking-assistant request into JSON constraints. Output only JSON with exactly these keys: "
    "diets, exclude, include, max_time, course, cuisine, attributes, servings, non_veg, intent, days.\n"
    f"diets: subset of {DIETS}. Allergies map to diets (peanut allergy -> peanut_free, nut allergy -> nut_free, "
    "no eggs -> egg_free, no dairy -> dairy_free, lactose intolerant -> lactose_free, no onion no garlic -> "
    "no_onion_garlic). vegan/jain/sattvic imply vegetarian, so don't also list vegetarian.\n"
    "exclude: canonical ingredient ids the user refuses that no listed diet already covers (e.g. mushroom, potato, "
    "coconut, paneer, cashew). include: ids the user has or wants (e.g. chicken, tomato, rice, paneer, dal, peas); "
    "drop any that the diets forbid.\n"
    f"max_time: integer minutes or null. course: one of {COURSES} or null. cuisine: lowercase region like "
    "'south indian', 'punjabi', 'italian' or null.\n"
    f"attributes: subset of {ATTRIBUTES}. servings: integer or null. non_veg: true only if meat/fish/egg dishes are "
    f"explicitly wanted. intent: one of {INTENTS}. days: integer only for a meal plan, else null."
)


def empty() -> dict:
    return {"diets": [], "exclude": [], "include": [], "max_time": None, "course": None, "cuisine": None,
            "attributes": [], "servings": None, "non_veg": False, "intent": "search", "days": None}


def normalize(d) -> dict:
    """Coerce any parser output into the canonical comparable form."""
    out = empty()
    if not isinstance(d, dict):
        return out
    for k in ("diets", "exclude", "include", "attributes"):
        v = d.get(k) or []
        out[k] = sorted({str(x).strip().lower() for x in v if isinstance(x, (str, int))}) if isinstance(v, list) else []
    for k in ("max_time", "servings", "days"):
        v = d.get(k)
        try:
            out[k] = int(v) if v not in (None, "", "null") else None
        except (TypeError, ValueError):
            out[k] = None
    for k in ("course", "cuisine"):
        v = d.get(k)
        out[k] = str(v).strip().lower() if v not in (None, "", "null") else None
    out["non_veg"] = bool(d.get("non_veg"))
    intent = str(d.get("intent") or "search").lower()
    out["intent"] = intent if intent in INTENTS else "search"
    return out


def parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        return normalize(json.loads(text[start:end + 1]))
    except Exception:
        return normalize(None)


def dumps(d: dict) -> str:
    return json.dumps(normalize(d), separators=(",", ":"))
