"""Nosh One fit: can the robot cook this, what does the user prep by hand, and does it fit the machine?

Hardware figures are from public reports (Gizmocrowd, New Atlas, Trashbox, CES 2026 coverage), NOT confirmed by
Nosh: 4 L pan with automated stirrer, 8 spice containers, oil and water dispensers, 2.5 L ingredient tray. Chopping
and prep are manual; the robot sautés, stirs, simmers and seasons; it cannot flip (roti, dosa, pancakes).
Everything here is a rule over the recipe text with a stated reason, so Nosh can correct it with their own data:
a catalogue imported with its own `robot` fields overrides these inferences (see scripts/import_catalogue.py).
"""
from __future__ import annotations

import os
import re

PAN_LITRES = float(os.getenv("NOSH_PAN_LITRES", "4"))
SPICE_SLOTS = int(os.getenv("NOSH_SPICE_SLOTS", "8"))
# Roughly how many servings of a curry/rice dish fit a 4 L pan in one batch (≈ 450-500 ml cooked per serving).
MAX_BATCH_SERVINGS = int(os.getenv("NOSH_MAX_BATCH_SERVINGS", "6"))

# Dry spices that would sit in the robot's spice containers (salt included: it must be dispensed too).
DRY_SPICES = {
    "ajwain", "allspice", "amchur", "anardana", "asafoetida", "black_pepper", "cardamom", "chaat_masala", "cinnamon",
    "cloves", "coriander_seed", "cumin", "dry_ginger", "fennel", "fenugreek_seed", "garam_masala", "khanda_lasun_masala",
    "long_pepper", "mustard_seed", "nigella", "nutmeg", "poppy_seeds", "red_chilli", "saffron", "salt",
    "sambar_powder", "spice_blend", "spice_misc", "star_anise", "turmeric_powder", "maggi_masala",
}

# Why a dish can't be cooked by the robot, checked in order against the name and method.
NOT_ROBOT = [
    ("needs flipping on a griddle (the robot can't flip)",
     r"\b(rotis?|rotti|rotli|rotlo|chapatis?|chapathi|phulka|paratha|parantha|parotta|thepla|naan|kulcha|bhakri|"
     r"rotla|dosas?|dosai|ut+ap+am|ut+ap+as?|paniyaram|appe|fritters?|(fish|prawns?|tawa|bhindi|paneer) fry|"
     r"cheela|chilla|chila|pesarattu|adai|appam|pancakes?|crepes?|tortillas?|omelette|french toast|poli|puran poli|"
     r"tikki|cutlets?|patties|kebabs?)\b"),
    ("needs deep-frying", r"\b(deep[- ]fr(y|ied|ying)|fry (them|it|the \w+) in (hot )?oil until|pakor?as?|pakodas?|"
                          r"bhajj?is?|bajj?is?|bonda|vadas?|wadas?|puris?|pooris?|bhatura|samosas?|kachoris?|"
                          r"chakli|murukku|jalebi|gulab jamun)\b"),
    ("needs a steamer or mould", r"\b(idl(i|y|is|ies)|dhoklas?|khaman|momos?|modaks?|kozhukattai|steamed|steam(ed)? in|"
                                 r"idli plate|steamer|pressure cook(er)? for \d+ whistles? .* (mould|plate))\b"),
    ("needs an oven, grill or tandoor", r"\b(bake|baked|baking|oven|preheat|grill(ed)?|tandoor|air[- ]?fr(y|ier)|"
                                        r"broil|toaster|cakes?|cupcakes?|cookies?|biscuits?|muffins?|brownies?|"
                                        r"pizzas?|tarts?|pies?|loaf|focaccia|buns?)\b"),
    ("no cooking needed", r"\b(salad|raita|smoothie|juice|lassi|chaas|buttermilk|milkshake|shake|cooler|mocktail|"
                          r"chutney|dip|hummus|sandwich|toast|parfait|overnight oats|ice ?cream|kulfi|sorbet|"
                          r"no[- ]cook|pickle|achar)\b"),
]
NOT_ROBOT = [(why, re.compile(rx, re.I)) for why, rx in NOT_ROBOT]
PREP_VERBS = re.compile(
    r"\b(chop|dice|slice|grate|peel|soak|grind|blend|puree|marinate|knead|mince|wash|rinse|julienne|crush|whisk|"
    r"cut|deseed|de-seed|devein|clean|pound|mash|beat|shred|sprout|keep (it )?aside|set aside|pressure cook|boil "
    r"the (potato|potatoes|dal|chickpeas|rajma))\w*\b", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s*(?=[A-Z])")


def assess(recipe: dict) -> dict:
    """Robot fit for one recipe. A catalogue's own `robot` field (from Nosh's data) always wins over inference."""
    spices = sorted({i for i in recipe.get("ingredient_ids", []) if i in DRY_SPICES})
    if isinstance(recipe.get("robot"), dict) and "fit" in recipe["robot"]:
        given = recipe["robot"]
        return {**given, "spices": spices, "extra_spices": max(0, len(spices) - SPICE_SLOTS),
                "batches_for_recipe": batches(recipe.get("servings"))}
    name, method = recipe.get("name", ""), recipe.get("instructions", "")
    # The dish name decides first (an idli is steamed even if its method mentions frying); then the method decides
    # for techniques the name doesn't show (frying, baking).
    reason = next((why for why, rx in NOT_ROBOT if rx.search(name)), None)
    if reason is None:
        reason = next((why for why, rx in NOT_ROBOT
                       if why in ("needs deep-frying", "needs an oven, grill or tandoor") and rx.search(method)
                       and not (why == "needs an oven, grill or tandoor" and not re.search(
                           r"\b(bake|baked|baking|oven|preheat|grill(ed)?|tandoor|air[- ]?fr(y|ier)|broil)\b", method, re.I))),
                      None)
    steps = [x.strip() for x in SENTENCE.split(method) if len(x.strip()) > 3]
    prep = [x for x in steps if PREP_VERBS.search(x) and not re.search(r"\b(heat|saute|sauté|stir|simmer|cook)\b", x, re.I)]
    servings = recipe.get("servings") or 0
    return {
        "fit": "robot" if reason is None else "manual",
        "reason": reason or "",
        "prep": prep[:8],
        "prep_min": recipe.get("prep_min") or 0,
        "spices": spices,
        "extra_spices": max(0, len(spices) - SPICE_SLOTS),
        "batches_for_recipe": max(1, -(-servings // MAX_BATCH_SERVINGS)) if servings else 1,
    }


def batches(servings: int | None) -> int:
    return max(1, -(-servings // MAX_BATCH_SERVINGS)) if servings else 1
