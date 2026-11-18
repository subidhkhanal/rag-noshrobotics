"""Answers about one dish already on screen ("is it healthy?", "does it have garlic?", "kya ye jain hai?") and
recall of what the chat has shown ("what was the last recipe?").

Every answer is computed from the recipe record (ingredient lines, diet verdicts, times), never generated, and it
says plainly when the data can't answer (there is no nutrition data).
"""
from __future__ import annotations

import re

from .query import Constraints

# "what was the last recipe", "what did you suggest earlier", "pichli recipe kya thi"
RECALL = re.compile(
    r"\b((what|which) (was|were|is|are) (the |your |that )?(last|previous|earlier|recent|first)( few)? "
    r"(recipes?|dish(es)?|suggestions?|ones?|options?|results?)|"
    r"what (did|have) you (just )?(show|suggest|recommend|give|tell)(ed)?( me)?|"
    r"(remind me|go back to|show me again|repeat)( of| to)? (the |your )?(last|previous|earlier)( few)? "
    r"(recipes?|dish(es)?|suggestions?|ones?|options?|results?|list)|"
    r"(last|previous) (recipes?|dish(es)?|suggestions?) (you|i) (showed|suggested|gave|asked|saw)|"
    r"(pichli|pichhli|pehle wali|pichla|pichhla) (recipe|dish|wali|wala)|kaun si recipe (batayi|di) thi)\b", re.I)

# A question about a specific dish: question form + a pointer to it.
QUESTION = re.compile(r"^\s*(is|isn'?t|are|does|doesn'?t|do|has|have|can|could|will|would|how|what|kya|ye|yeh|"
                      r"it|this|that|should)\b|\?\s*$|\b(hai|hain|kya)\s*\??\s*$", re.I)
POINTER = re.compile(r"\b(it|its|this|that|these|those|them|ye|yeh|iska|iski|isme|ismein|isse|"
                     r"(the )?(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th) (one|recipe|dish))\b", re.I)
ALL_SHOWN = re.compile(r"\b(these|those|them|all of them|all (these|those|the) (ones|recipes|dishes))\b", re.I)
ASKS_TIME = re.compile(r"\b(how long|how much time|time|minutes?|mins?|kitna (time|samay)|der)\b", re.I)
ASKS_SERVES = re.compile(r"\b(how many (people|persons|servings?)|serves?|servings?|kitne log|kitne logo)\b", re.I)
ASKS_NUTRITION = re.compile(r"\b(calories?|kcal|nutrition|nutrients?|macros?|carbs?|fat content|sugar content|"
                            r"protein content|grams? of protein|diabetic|weight loss)\b", re.I)


def is_dish_question(message: str) -> bool:
    return bool(QUESTION.search(message) and (POINTER.search(message) or re.search(r"\?\s*$", message)))


def answer(engine, idx: int, q: Constraints, message: str) -> list[str]:
    """Plain-language answers about recipe index `idx` for what the question asks. Empty if it asks nothing we know."""
    onto, r = engine.onto, engine.recipes[idx]
    n = len(r["ingredients"])
    out: list[str] = []
    for d in sorted(q.diets):
        v, label = r["verdicts"][d], onto.diets[d]["label"]
        if v["status"] == "pass":
            out.append(f"Yes, it's {label}: all {n} ingredients checked.")
        elif v["status"] == "block":
            out.append(f"No, it isn't {label}: {v['reasons'][0]}.")
        else:
            out.append(f"Probably {label}, but check: {v['reasons'][0]}.")
    for e in list(dict.fromkeys(list(q.include) + sorted(q.exclude))):
        v, label = onto.exclusion_verdict(e, r["ingredients"]), onto.label(e)
        if v.status == "block":
            out.append(f"Yes, it contains {label} ({v.reasons[0]}).")
        elif v.status == "flag":
            out.append(f"Possibly: {v.reasons[0]}.")
        else:
            out.append(f"No {label} in its ingredient list.")
    feats = engine.features(idx)
    ids = set(r["ingredient_ids"])
    if "healthy" in q.attributes or ASKS_NUTRITION.search(message):
        out.append(_healthiness(engine, idx, feats, ids))
    if "spicy" in q.attributes or "mild" in q.attributes:
        from .search import SPICE_IDS
        hot = sorted(onto.label(x) for x in ids & SPICE_IDS)
        if not hot:
            out.append("It's mild, with no chillies or pepper listed.")
        else:
            level = "quite spicy" if feats["spicy"] >= 0.67 else "medium spicy" if feats["spicy"] >= 0.5 else "mildly spiced"
            out.append(f"It's {level} (heat from {', '.join(hot)}).")
    if "high_protein" in q.attributes:
        from .search import PROTEIN_IDS
        src = sorted(onto.label(x) for x in ids & PROTEIN_IDS)
        out.append(f"Protein comes from {', '.join(src)}." if src else "It has no major protein source (no dal, paneer, "
                   "legumes, eggs or meat).")
    if "creamy" in q.attributes:
        out.append("Yes, it's on the creamy side." if feats["creamy"] >= 0.5 else "Not especially creamy.")
    if "quick" in q.attributes or ASKS_TIME.search(message) or q.max_time:
        t = r["total_min"]
        parts = [f"prep {r['prep_min']} min" if r.get("prep_min") else "", f"cook {r['cook_min']} min" if r.get("cook_min") else ""]
        detail = ", ".join(p for p in parts if p)
        out.append(f"It takes about {t} minutes{f' ({detail})' if detail else ''}." if t else "The recipe doesn't give a time.")
    if ASKS_SERVES.search(message) or q.servings:
        s = r["servings"]
        if s and q.servings and q.servings != s:
            out.append(f"It serves {s}; scale it about {q.servings / s:.1f}× for {q.servings}.")
        else:
            out.append(f"It serves {s}." if s else "The recipe doesn't say how many it serves.")
    return out


def _healthiness(engine, idx: int, feats: dict, ids: set) -> str:
    from .search import HEAVY_IDS, LIGHT_IDS
    onto, r = engine.onto, engine.recipes[idx]
    fried = bool(re.search(r"deep fry|deep-fry|deep fried", r["instructions"].lower()))
    heavy = sorted(onto.label(x) for x in ids & HEAVY_IDS)
    light = sorted(onto.label(x) for x in ids & LIGHT_IDS)
    verdict = ("Fairly light" if feats["healthy"] >= 0.65 else "Middle of the road" if feats["healthy"] >= 0.4
               else "On the heavier side")
    why = []
    why.append("it's deep-fried" if fried else "no deep-frying")
    if light:
        why.append("built on " + ", ".join(light[:3]))
    if heavy:
        why.append("uses " + ", ".join(heavy[:3]))
    return (f"{verdict}: {'; '.join(why)}. I don't have calorie or nutrition data, so this is judged from its "
            f"ingredients and cooking method.")
