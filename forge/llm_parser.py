"""Optional Claude parser, combined with the rule parser so the model can only add restrictions.

Enabled when ANTHROPIC_API_KEY is set. CLAUDE_PARSER_MODEL picks the model (default claude-opus-5-5).
Only called for messages with cues the rules may miss (negations, allergies, Hinglish), so plain
queries keep the rules-only latency. Any error or timeout falls back to the rules' result.
"""
from __future__ import annotations

import json
import os
import re

from .ontology import Ontology
from .parser_schema import ATTRIBUTES, COURSES, DIETS, INTENTS, SYSTEM, normalize
from .query import ATTRIBUTE_RES, COURSE_RES, CUISINE_GROUPS, Constraints

TIMEOUT_S = float(os.getenv("CLAUDE_PARSER_TIMEOUT", "6"))
CUES = re.compile(
    r"\b(hold|can'?t|cannot|don'?t|won'?t|doesn'?t|allerg\w*|avoid|without|minus|skip|except|no|not|never|"
    r"nahi|nahin|bina|mat|hate|dislike|stand|free|intoleran\w*|celiac|coeliac|fast|fasting|vrat|ready|"
    r"within|under|less|min|mins|minutes|hour|ke bina|nothing)\b|-free",
    re.I,
)
_NULLABLE_INT = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
SCHEMA = {
    "type": "object",
    "properties": {
        "diets": {"type": "array", "items": {"type": "string", "enum": DIETS}},
        "exclude": {"type": "array", "items": {"type": "string"}},
        "include": {"type": "array", "items": {"type": "string"}},
        "max_time": _NULLABLE_INT,
        "course": {"anyOf": [{"type": "string", "enum": COURSES}, {"type": "null"}]},
        "cuisine": _NULLABLE_STR,
        "attributes": {"type": "array", "items": {"type": "string", "enum": ATTRIBUTES}},
        "servings": _NULLABLE_INT,
        "non_veg": {"type": "boolean"},
        "intent": {"type": "string", "enum": INTENTS},
        "days": _NULLABLE_INT,
    },
    "required": ["diets", "exclude", "include", "max_time", "course", "cuisine", "attributes", "servings",
                 "non_veg", "intent", "days"],
    "additionalProperties": False,
}

_client = None


def enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def should_call(message: str) -> bool:
    return bool(CUES.search(message))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=0)
    return _client


def llm_parse(message: str) -> dict | None:
    import anthropic
    try:
        response = _get_client().messages.create(
            model=os.getenv("CLAUDE_PARSER_MODEL", "claude-opus-5-5"),
            max_tokens=2048,
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": message}],
        )
    except (anthropic.APIConnectionError, anthropic.APIStatusError):
        return None
    if response.stop_reason != "end_turn":
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return normalize(json.loads(text))
    except json.JSONDecodeError:
        return None


def _canonical(onto: Ontology, words: list[str]) -> list[str]:
    """Map the model's ingredient words ("onions", "kaju") to ontology ids; unknown words are dropped."""
    out: list[str] = []
    for w in words:
        ids = [w] if w in onto.ingredients else [m.ingredient for m in onto.match(w.replace("_", " "))]
        out += [i for i in ids if i not in out and "non_food" not in onto.ingredients[i].tags]
    return out


def combine(onto: Ontology, rules: Constraints, model: dict | None) -> Constraints:
    """Union hard constraints (diets, exclusions, tightest time limit); fill soft fields only where rules found none."""
    if not model:
        return rules
    rules.diets |= {d for d in model["diets"] if d in onto.diets}
    if rules.diets & {"vegan", "jain", "sattvic"}:
        rules.diets -= {"vegetarian", "eggetarian"}
    if "vegetarian" in rules.diets:
        rules.diets.discard("eggetarian")
    rules.exclude |= set(_canonical(onto, model["exclude"]))
    times = [t for t in (rules.max_time, model["max_time"]) if t and 0 < t <= 600]
    rules.max_time = min(times) if times else None
    blocked = {t for d in rules.diets for t in onto.diets[d].get("block", [])}
    rules.exclude = {e for e in rules.exclude if not onto.effective_tags(e) & blocked}
    rules.include = [i for i in rules.include if not onto.effective_tags(i) & blocked and i not in rules.exclude]
    if not rules.include:
        rules.include = [i for i in _canonical(onto, model["include"]) if i not in rules.exclude
                         and not onto.effective_tags(i) & blocked]
    rules.course = rules.course or (model["course"] if model["course"] in COURSE_RES else None)
    rules.cuisine = rules.cuisine or (model["cuisine"] if model["cuisine"] in CUISINE_GROUPS else None)
    rules.attributes |= {a for a in model["attributes"] if a in ATTRIBUTE_RES}
    rules.servings = rules.servings or model["servings"]
    rules.prefer_non_veg = rules.prefer_non_veg or (model["non_veg"] and "meat" not in blocked)
    if model["intent"] != "search" and not rules.intents:
        rules.intents.add(model["intent"])
        if model["intent"] == "plan":
            rules.days = rules.days or model["days"] or 7
    return rules
