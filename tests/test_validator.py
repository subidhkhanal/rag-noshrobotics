import json
import re
from pathlib import Path

import pytest

from forge.ontology import load
from forge.recipes import parse_ingredients

ROOT = Path(__file__).resolve().parent.parent
onto = load()


def verdict(diet, *lines):
    return onto.diet_verdict(diet, parse_ingredients(onto, ", ".join(lines))).status


@pytest.mark.parametrize(
    "diet,line,expected",
    [
        ("no_onion_garlic", "1 tablespoon Ginger Garlic Paste", "block"),
        ("jain", "1 tablespoon Ginger Garlic Paste", "block"),
        ("no_onion_garlic", "2 Spring Onion (Bulb & Greens) - chopped", "block"),
        ("no_onion_garlic", "6 Shallots - peeled", "block"),
        ("no_onion_garlic", "1 tablespoon Kalonji (Onion Nigella Seeds)", "pass"),
        ("no_onion_garlic", "2 tablespoons Tomato Ketchup", "flag"),
        ("no_onion_garlic", "1 packet Maggi Masala-ae-Magic", "flag"),
        ("no_onion_garlic", "1 tablespoon Schezwan Sauce", "block"),
        ("no_onion_garlic", "2 कली लहसुन", "block"),
        ("jain", "2 Potatoes (Aloo) - boiled", "block"),
        ("jain", "1 Carrot (Gajjar) - grated", "block"),
        ("jain", "200 grams Button mushrooms", "block"),
        ("jain", "1 teaspoon Honey", "block"),
        ("jain", "1 teaspoon Dry ginger powder", "flag"),
        ("sattvic", "2 Potatoes (Aloo) - boiled", "pass"),
        ("vegan", "1 cup Coconut milk", "pass"),
        ("vegan", "1 cup Almond Milk (Badam Milk)", "pass"),
        ("vegan", "2 tablespoons Peanut butter", "pass"),
        ("vegan", "1 tablespoon Cocoa butter", "pass"),
        ("vegan", "1 tablespoon Ghee", "block"),
        ("vegan", "1 cup Buttermilk", "block"),
        ("vegan", "1 teaspoon Honey", "block"),
        ("vegetarian", "1 Brinjal (Baingan / Eggplant)", "pass"),
        ("vegetarian", "2 tablespoons Eggless mayonnaise", "pass"),
        ("vegetarian", "2 tablespoons Classic Mayonnaise (with egg)", "block"),
        ("vegetarian", "1 tablespoon Fish sauce", "block"),
        ("vegetarian", "12 Pav buns - (ham burger buns could be substituted)", "flag"),
        ("vegetarian", "2 cups Chicken stock", "block"),
        ("lactose_free", "1 tablespoon Ghee", "flag"),
        ("lactose_free", "1/2 cup Paneer (Homemade Cottage Cheese)", "block"),
        ("gluten_free", "1/4 teaspoon Asafoetida (hing)", "flag"),
        ("gluten_free", "1 tablespoon Soy sauce", "block"),
        ("gluten_free", "1 cup Sooji (Semolina/ Rava)", "block"),
        ("gluten_free", "1 tsp rye", "pass"),
        ("nut_free", "10 Cashew nuts", "block"),
        ("nut_free", "1 tablespoon Groundnut oil", "block"),
        ("nut_free", "1 cup Fresh coconut - grated", "pass"),
    ],
)
def test_line_verdicts(diet, line, expected):
    assert verdict(diet, line) == expected


def test_reasons_name_the_offending_line():
    v = onto.diet_verdict("no_onion_garlic", parse_ingredients(onto, "1 cup Methi leaves, 1 tablespoon Ginger Garlic Paste"))
    assert v.status == "block"
    assert "Ginger Garlic Paste" in v.reasons[0]


def test_user_exclusion_covers_family_and_compounds():
    ings = parse_ingredients(onto, "4 Shallots, 1 tbsp Ginger Garlic Paste, 2 tbsp Tomato Ketchup")
    assert onto.exclusion_verdict("onion", ings[:1]).status == "block"
    assert onto.exclusion_verdict("garlic", ings[1:2]).status == "block"
    assert onto.exclusion_verdict("garlic", ings[2:]).status == "flag"
    assert onto.exclusion_verdict("tomato", ings[2:]).status == "block"
    assert onto.exclusion_verdict("chilli", parse_ingredients(onto, "2 Green Chillies")).status == "block"


# ---------- Independent corpus-wide check (regex on raw text, not the ontology) ----------

CORPUS = ROOT / "data" / "processed" / "recipes.jsonl"
SAFE_PHRASES = re.compile(
    r"onion (nigella )?seeds|nigella|kalonji|coconut (milk|cream|yogurt|curd)|almond milk|badam milk|soy milk|oat milk|"
    r"peanut butter|cocoa butter|almond butter|eggless|eggplant|egg replacer|butter ?scotch|dry ginger|ginger powder|"
    r"sonth|saunth|cream of tartar|ice cream cone|ginger ale|honey ?dew",
    re.I,
)
INDEPENDENT = {
    "no_onion_garlic": r"\b(onions?|garlic|shallots?|leeks?|chives|lahsun|lehsun|pyaz|scallions?)\b",
    "jain": r"\b(onions?|garlic|shallots?|potato(es)?|aloo|carrots?|beetroots?|radish|mooli|mushrooms?|ginger|honey|eggs?|chicken|mutton|fish|prawns?)\b",
    "vegan": r"\b(milk|ghee|butter|curd|dahi|paneer|cream|cheese|honey|eggs?|yogurt|khoya|mawa)\b",
}


@pytest.mark.skipif(not CORPUS.exists(), reason="run python -m forge.build_corpus first")
@pytest.mark.parametrize("diet", sorted(INDEPENDENT))
def test_no_passing_recipe_mentions_forbidden_ingredient(diet):
    pattern = re.compile(INDEPENDENT[diet], re.I)
    violations = []
    for raw in CORPUS.read_text(encoding="utf-8").splitlines():
        rec = json.loads(raw)
        if rec["verdicts"][diet]["status"] == "block":
            continue
        for it in rec["ingredients"]:
            text = SAFE_PHRASES.sub(" ", it["line"])
            if pattern.search(text):
                violations.append((rec["name"], it["line"]))
    assert violations == [], f"{len(violations)} leaks, e.g. {violations[:10]}"


def test_method_cross_check_catches_unlisted_garlic():
    from forge.build_corpus import method_only_ingredients
    ings = parse_ingredients(onto, "200 grams Pasta - farfalle, 5 cloves Cloves (Laung) - minced")
    steps = "Cook the pasta. Heat olive oil in a wide pan, add the minced garlic and saute."
    assert method_only_ingredients(onto, "Farfalle Garlic-And-Oil", steps, ings) == ["garlic"]


def test_method_cross_check_ignores_serving_suggestions_and_negations():
    from forge.build_corpus import method_only_ingredients
    ings = parse_ingredients(onto, "1 cup Toor Dal")
    steps = ("To begin making the Dal Recipe in no onion no garlic style, boil the dal. "
             "Serve with egg mayo sandwich recipe and onion raita.")
    assert method_only_ingredients(onto, "Dal", steps, ings) == []
