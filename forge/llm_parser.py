"""Optional fine-tuned LLM parser, combined with the rule parser so the model can only add restrictions.

Enable by pointing at any OpenAI-compatible server hosting the QLoRA model (Ollama, vLLM, llama.cpp):
    PARSER_URL=http://localhost:11434/v1  PARSER_MODEL=recipeforge-parser  [PARSER_KEY=...]
"""
from __future__ import annotations

import os

import httpx

from .ontology import Ontology
from .parser_schema import SYSTEM, parse_json
from .query import ATTRIBUTE_RES, COURSE_RES, CUISINE_GROUPS, Constraints

TIMEOUT_S = 2.5


def enabled() -> bool:
    return bool(os.getenv("PARSER_URL") and os.getenv("PARSER_MODEL"))


def llm_parse(message: str) -> dict | None:
    try:
        r = httpx.post(
            os.environ["PARSER_URL"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('PARSER_KEY', 'none')}"},
            json={"model": os.environ["PARSER_MODEL"], "temperature": 0, "max_tokens": 192,
                  "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]},
            timeout=TIMEOUT_S,
        )
        return parse_json(r.json()["choices"][0]["message"]["content"])
    except Exception:
        return None


def combine(onto: Ontology, rules: Constraints, model: dict | None) -> Constraints:
    """Union hard constraints (diets, exclusions, tightest time limit); fill soft fields only where rules found none."""
    if not model:
        return rules
    rules.diets |= {d for d in model["diets"] if d in onto.diets}
    if rules.diets & {"vegan", "jain", "sattvic"}:
        rules.diets -= {"vegetarian", "eggetarian"}
    rules.exclude |= {e for e in model["exclude"] if e in onto.ingredients}
    times = [t for t in (rules.max_time, model["max_time"]) if t and 0 < t <= 600]
    rules.max_time = min(times) if times else None
    blocked = {t for d in rules.diets for t in onto.diets[d].get("block", [])}
    rules.exclude = {e for e in rules.exclude if not onto.effective_tags(e) & blocked}
    if not rules.include:
        rules.include = [i for i in model["include"] if i in onto.ingredients and i not in rules.exclude
                         and not onto.effective_tags(i) & blocked]
    rules.course = rules.course or (model["course"] if model["course"] in COURSE_RES else None)
    rules.cuisine = rules.cuisine or (model["cuisine"] if model["cuisine"] in CUISINE_GROUPS else None)
    rules.attributes |= {a for a in model["attributes"] if a in ATTRIBUTE_RES}
    rules.servings = rules.servings or model["servings"]
    rules.prefer_non_veg = rules.prefer_non_veg or (model["non_veg"] and not blocked & {"meat"})
    if model["intent"] != "search" and not rules.intents:
        rules.intents.add(model["intent"])
        if model["intent"] == "plan":
            rules.days = rules.days or model["days"] or 7
    return rules
