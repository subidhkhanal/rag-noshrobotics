"""Generate synthetic parser training data with gold labels known by construction.

Usage: python -m finetune.make_data  ->  finetune/data/train.jsonl, val.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from forge.ontology import load

from forge.parser_schema import SYSTEM, dumps, empty

OUT = Path(__file__).resolve().parent / "data"
onto = load()

DIET_PHRASES = {
    "vegetarian": ["vegetarian", "veg", "pure veg", "shakahari", "veggie", "no meat or eggs", "strictly vegetarian"],
    "vegan": ["vegan", "plant-based", "fully vegan", "no animal products", "100% vegan"],
    "jain": ["jain", "jain-friendly", "for my jain in-laws", "jain style", "suitable for jains"],
    "sattvic": ["sattvic", "satvik", "for my navratri fast", "vrat-friendly", "upvas"],
    "no_onion_garlic": ["no onion no garlic", "without onion and garlic", "bina pyaz lahsun", "onion-garlic free",
                        "no pyaz no lahsun", "skip the onion and garlic", "without any onion or garlic",
                        "no onion, no garlic"],
    "lactose_free": ["lactose free", "I'm lactose intolerant", "lactose-intolerant friendly", "for someone lactose intolerant"],
    "dairy_free": ["dairy free", "no dairy", "without any dairy", "allergic to milk", "dairy allergy"],
    "gluten_free": ["gluten free", "celiac friendly", "no gluten", "wheat allergy", "I can't have gluten", "coeliac"],
    "nut_free": ["nut free", "no nuts", "without any nuts", "nut allergy", "allergic to nuts"],
    "peanut_free": ["peanut allergy", "allergic to peanuts", "no peanuts", "allergic to groundnuts", "peanut-free"],
    "tree_nut_free": ["tree nut allergy", "allergic to tree nuts"],
    "egg_free": ["eggless", "no eggs", "egg free", "allergic to eggs", "without eggs"],
    "eggetarian": ["eggetarian", "vegetarian but eggs are fine"],
    "soy_free": ["soy free", "no soy", "allergic to soy"],
    "sesame_free": ["sesame allergy", "no sesame", "allergic to sesame"],
    "shellfish_free": ["shellfish allergy", "allergic to prawns", "allergic to shellfish"],
}
VEG_FAMILY = {"vegetarian", "vegan", "jain", "sattvic", "eggetarian"}
EXCL = {
    "mushroom": ["mushroom", "mushrooms", "khumb"], "potato": ["potato", "potatoes", "aloo"],
    "coconut": ["coconut", "nariyal"], "paneer": ["paneer"], "tomato": ["tomato", "tomatoes", "tamatar"],
    "brinjal": ["brinjal", "eggplant", "baingan"], "capsicum": ["capsicum", "bell peppers", "shimla mirch"],
    "cauliflower": ["cauliflower", "gobi"], "okra": ["okra", "bhindi"], "cheese": ["cheese"],
    "cream": ["cream", "fresh cream"], "chickpeas": ["chickpeas", "chole"], "sugar": ["sugar"],
    "peas": ["peas", "matar"], "corn": ["corn", "sweet corn"], "curd": ["curd", "dahi", "yogurt"],
    "spinach": ["spinach", "palak"], "cashew": ["cashews", "kaju"], "rice": ["rice", "chawal"],
    "green_chilli": ["green chillies", "hari mirch"], "beetroot": ["beetroot"], "carrot": ["carrot", "carrots", "gajar"],
    "cabbage": ["cabbage", "patta gobi"], "onion": ["onion", "onions", "pyaz"], "garlic": ["garlic", "lahsun"],
}
INCL = {
    "paneer": ["paneer"], "potato": ["potatoes", "aloo"], "peas": ["peas", "matar"], "chicken": ["chicken"],
    "tomato": ["tomatoes", "tomato"], "rice": ["rice", "leftover rice"], "spinach": ["spinach", "palak"],
    "chickpeas": ["chickpeas", "kabuli chana"], "dal": ["dal", "lentils", "moong dal", "toor dal"],
    "egg": ["eggs", "egg"], "mushroom": ["mushrooms"], "cauliflower": ["cauliflower", "gobi"],
    "okra": ["bhindi", "okra"], "corn": ["corn"], "capsicum": ["capsicum"], "cabbage": ["cabbage"],
    "fish": ["fish"], "mutton": ["mutton"], "tofu": ["tofu"], "oats": ["oats"], "besan": ["besan", "gram flour"],
    "coconut": ["coconut"], "curd": ["curd", "dahi"], "brinjal": ["baingan", "brinjal"], "carrot": ["carrots"],
    "soy_chunks": ["soya chunks"], "millet": ["ragi", "millets"], "pasta": ["pasta"], "noodles": ["noodles"],
}
EXCL_T = ["no {x}", "without {x}", "hold the {x}", "skip the {x}", "I hate {x}", "I can't stand {x}", "{x} nahi",
          "not a fan of {x}", "keep {x} out", "{x}-free", "minus the {x}", "don't add {x}", "avoid {x}",
          "nothing with {x}"]
INCL_T = ["with {x}", "using {x}", "I have {x}", "I've got {x} in the fridge", "need to use up {x}", "something with {x}",
          "{x} based", "made with {x}", "{x} wala"]
TIME_T = [("under {n} minutes", None), ("in {n} mins", None), ("within {n} min", None), ("{n} minutes max", None),
          ("ready in {n} minutes", None), ("{n} min mein", None), ("less than {n} minutes", None),
          ("in half an hour", 30), ("under an hour", 60), ("within an hour", 60)]
COURSE_P = {"breakfast": ["breakfast", "nashta", "for breakfast"], "lunch": ["lunch", "lunchbox", "for lunch"],
            "dinner": ["dinner", "for dinner tonight", "raat ka khana"], "snack": ["snack", "evening snack", "starter"],
            "dessert": ["dessert", "something sweet", "mithai"], "side": ["side dish", "sabzi"]}
CUISINES = ["south indian", "north indian", "punjabi", "gujarati", "bengali", "kerala", "tamil", "andhra", "maharashtrian",
            "goan", "rajasthani", "kashmiri", "chettinad", "mughlai", "hyderabadi", "italian", "mexican", "thai",
            "chinese", "indo chinese", "continental", "mediterranean", "karnataka", "parsi"]
CUISINE_T = ["{c}", "{c} style", "{c} food", "something {c}"]
ATTR_P = {"spicy": ["spicy", "teekha", "fiery"], "mild": ["mild", "not too spicy", "kid-friendly"],
          "creamy": ["creamy", "rich and creamy"], "healthy": ["healthy", "light", "low oil"],
          "high_protein": ["high protein", "protein-rich"], "quick": ["quick", "fast", "easy"],
          "comfort": ["comforting", "comfort food"], "festive": ["festive", "for a party"],
          "crispy": ["crispy", "crunchy"], "tangy": ["tangy", "khatta"]}
SERV_T = ["for {n} people", "for {n} guests", "serves {n}", "for a family of {n}"]
NONVEG_P = ["non veg", "non-vegetarian", "something with meat"]
PREFIX = ["", "", "suggest ", "I want ", "kuch ", "can you find ", "any ", "show me ", "what can I make ", "need ",
          "looking for ", "give me "]
SUFFIX = ["", "", " recipes", " options", " ideas", " please", "?", " recipe"]
INTENT_P = {
    "more": ["show me more", "anything else?", "more options", "something else"],
    "similar": ["something similar", "like that one but different", "similar to the first one"],
    "different": ["something different from what I usually make", "surprise me with something new",
                  "something I don't usually cook"],
    "history": ["what did I cook last week", "show my cooking history", "what have I cooked recently"],
}
PLAN_T = ["plan my week", "make a {n} day meal plan", "plan {n} dinners", "weekly menu", "{n}-day plan", "plan meals for the week"]


def blocked_tags(diets):
    return {t for d in diets for t in onto.diets[d].get("block", [])}


def gold_finalize(g):
    if g["diets"] & {"vegan", "jain", "sattvic"}:
        g["diets"] -= {"vegetarian", "eggetarian"}
    if "vegetarian" in g["diets"]:
        g["diets"].discard("eggetarian")
    bt = blocked_tags(g["diets"])
    g["exclude"] = {e for e in g["exclude"] if not onto.effective_tags(e) & bt}
    g["include"] = [i for i in g["include"] if i not in g["exclude"] and not onto.effective_tags(i) & bt]
    if g["diets"] & VEG_FAMILY:
        g["non_veg"] = False
    out = empty()
    out.update({k: (sorted(v) if isinstance(v, set) else v) for k, v in g.items()})
    return out


def sample(rng: random.Random):
    g = {"diets": set(), "exclude": set(), "include": [], "max_time": None, "course": None, "cuisine": None,
         "attributes": set(), "servings": None, "non_veg": False, "intent": "search", "days": None}
    parts = []
    roll = rng.random()
    if roll < 0.06:
        g["intent"] = rng.choice(["more", "similar", "different", "history"])
        parts.append(rng.choice(INTENT_P[g["intent"]]))
        if g["intent"] in ("history",):
            return " ".join(parts), gold_finalize(g)
    elif roll < 0.14:
        g["intent"] = "plan"
        t = rng.choice(PLAN_T)
        n = rng.choice([3, 4, 5, 6, 7]) if "{n}" in t else 7
        g["days"] = n
        parts.append(t.format(n=n))
    slots = rng.sample(["diet", "diet2", "excl", "incl", "time", "course", "cuisine", "attr", "serv", "nonveg"],
                       k=rng.choice([1, 1, 2, 2, 3, 3, 4]))
    for s in slots:
        if s in ("diet", "diet2"):
            d = rng.choice(list(DIET_PHRASES))
            g["diets"].add(d)
            parts.append(rng.choice(DIET_PHRASES[d]))
        elif s == "excl":
            xs = rng.sample(list(EXCL), k=rng.choice([1, 1, 2]))
            g["exclude"] |= set(xs)
            words = [rng.choice(EXCL[x]) for x in xs]
            parts.append(rng.choice(EXCL_T).format(x=" or ".join(words) if len(words) > 1 else words[0]))
        elif s == "incl":
            xs = rng.sample(list(INCL), k=rng.choice([1, 2, 2, 3]))
            g["include"] += [x for x in xs if x not in g["include"]]
            words = [rng.choice(INCL[x]) for x in xs]
            joined = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
            parts.append(rng.choice(INCL_T).format(x=joined))
        elif s == "time":
            t, fixed = rng.choice(TIME_T)
            n = fixed or rng.choice([10, 15, 20, 25, 30, 40, 45])
            g["max_time"] = n
            parts.append(t.format(n=n))
        elif s == "course":
            c = rng.choice(list(COURSE_P))
            g["course"] = c
            parts.append(rng.choice(COURSE_P[c]))
        elif s == "cuisine":
            c = rng.choice(CUISINES)
            g["cuisine"] = c
            parts.append(rng.choice(CUISINE_T).format(c=c))
        elif s == "attr":
            a = rng.choice(list(ATTR_P))
            g["attributes"].add(a)
            parts.append(rng.choice(ATTR_P[a]))
        elif s == "serv":
            n = rng.choice([2, 3, 4, 5, 6, 8, 10])
            g["servings"] = n
            parts.append(rng.choice(SERV_T).format(n=n))
        elif s == "nonveg":
            g["non_veg"] = True
            parts.append(rng.choice(NONVEG_P))
    rng.shuffle(parts)
    glue = rng.choice([" ", ", ", " and ", " — "])
    text = rng.choice(PREFIX) + glue.join(p for p in parts if p) + rng.choice(SUFFIX)
    if rng.random() < 0.3:
        text = text.lower()
    return text.strip(), gold_finalize(g)


def make(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n:
        text, gold = sample(rng)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text},
                                  {"role": "assistant", "content": dumps(gold)}]})
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, n, seed in (("train", 6000, 1), ("val", 300, 2)):
        rows = make(n, seed)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))
    for r in make(8, 3):
        print(r["messages"][1]["content"], "=>", r["messages"][2]["content"])
