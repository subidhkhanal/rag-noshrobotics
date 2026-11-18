"""Hand-written evaluation queries with gold labels, phrased independently of the training templates.

Usage: python -m finetune.test_set  ->  finetune/data/test.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from forge.parser_schema import empty

OUT = Path(__file__).resolve().parent / "data" / "test.jsonl"


def g(**kw):
    d = empty()
    d.update(kw)
    return d


TESTS = [
    # --- failures observed in the Nosh Assistant ---
    ("no onion no garlic vegetarian", g(diets=["no_onion_garlic", "vegetarian"])),
    ("Jain food options", g(diets=["jain"])),
    ("I'm lactose intolerant what can I make", g(diets=["lactose_free"])),
    ("recipes without nuts", g(diets=["nut_free"])),
    ("I have chicken tomatoes and rice", g(include=["chicken", "rice", "tomato"])),
    ("I want butter chicken but I'm vegetarian", g(diets=["vegetarian"], include=["butter"])),
    ("something for a dinner party for 6 people", g(course="dinner", servings=6, attributes=["festive"])),
    ("vegetarian breakfast under 20 mins", g(diets=["vegetarian"], course="breakfast", max_time=20)),
    ("high protein non veg under 15 mins", g(attributes=["high_protein"], non_veg=True, max_time=15)),
    ("what did I cook last week", g(intent="history")),
    ("suggest something different from what I usually make", g(intent="different")),
    # --- natural phrasings the rules may miss ---
    ("hold the onions please, and make it vegan", g(diets=["vegan"], exclude=["onion"])),
    ("my daughter can't eat peanuts, need a tiffin idea", g(diets=["peanut_free"], course="lunch")),
    ("my kid is allergic to cashews", g(exclude=["cashew"])),
    ("anything I can whip up in 25 minutes?", g(max_time=25)),
    ("ready in 25", g(max_time=25)),
    ("we're 8 people tonight, something punjabi", g(servings=8, cuisine="punjabi", course="dinner")),
    ("I can't stand mushrooms. paneer dinner?", g(exclude=["mushroom"], include=["paneer"], course="dinner")),
    ("no dairy at all, and nothing with soy either", g(diets=["dairy_free", "soy_free"])),
    ("celiac here — south indian breakfast ideas", g(diets=["gluten_free"], cuisine="south indian", course="breakfast")),
    ("sattvic khana for navratri", g(diets=["sattvic"])),
    ("lahsun pyaz ke bina kuch banao", g(diets=["no_onion_garlic"])),
    ("aloo aur matar hai, kya banau", g(include=["peas", "potato"])),
    ("kuch teekha chahiye", g(attributes=["spicy"])),
    ("kuch spicy banana hai", g(attributes=["spicy"])),
    ("not too spicy please, it's for my kids", g(attributes=["mild"])),
    ("healthy lunch without potatoes", g(attributes=["healthy"], course="lunch", exclude=["potato"])),
    ("egg free cake", g(diets=["egg_free"], include=[])),
    ("eggless dessert for a birthday", g(diets=["egg_free"], course="dessert", attributes=["festive"])),
    ("vegan protein-packed lunch, no peanuts", g(diets=["peanut_free", "vegan"], attributes=["high_protein"], course="lunch")),
    ("give me a jain dinner that takes less than 40 minutes", g(diets=["jain"], course="dinner", max_time=40)),
    ("plan my week, Jain dinners", g(diets=["jain"], course="dinner", intent="plan", days=7)),
    ("5 day vegan lunch plan", g(diets=["vegan"], course="lunch", intent="plan", days=5)),
    ("make me a 3-day meal plan with no gluten", g(diets=["gluten_free"], intent="plan", days=3)),
    ("show me more", g(intent="more")),
    ("something similar but spicier", g(intent="similar", attributes=["spicy"])),
    ("I'm craving something like dal but creamy", g(include=["dal"], attributes=["creamy"])),
    ("leftover rice — what can I do with it", g(include=["rice"])),
    ("gluten free dessert in half an hour", g(diets=["gluten_free"], course="dessert", max_time=30)),
    ("bengali fish curry", g(cuisine="bengali", include=["fish"], non_veg=False)),
    ("thai curry, no onion no garlic", g(cuisine="thai", diets=["no_onion_garlic"])),
    ("I'm allergic to sesame and shellfish", g(diets=["sesame_free", "shellfish_free"])),
    ("quick snack without maida", g(attributes=["quick"], course="snack", exclude=["maida"])),
    ("something with spinach and paneer for dinner", g(include=["paneer", "spinach"], course="dinner")),
    ("kerala style vegetarian lunch", g(cuisine="kerala", diets=["vegetarian"], course="lunch")),
    ("a crispy evening snack, nothing deep fried in maida", g(attributes=["crispy"], course="snack", exclude=["maida"])),
    ("I don't eat eggs or fish", g(diets=["egg_free", "fish_free"])),
    ("strictly vegetarian, no mushrooms, no paneer", g(diets=["vegetarian"], exclude=["mushroom", "paneer"])),
    ("dinner for two under an hour", g(course="dinner", servings=2, max_time=60)),
    ("comfort food for a rainy evening", g(attributes=["comfort"])),
    ("low oil gujarati dinner", g(attributes=["healthy"], cuisine="gujarati", course="dinner")),
    ("tangy chettinad side dish", g(attributes=["tangy"], cuisine="chettinad", course="side")),
    ("I've got tofu and broccoli", g(include=["broccoli", "tofu"])),
    ("no coconut please, I'm allergic", g(exclude=["coconut"])),
    ("diabetic friendly breakfast without sugar", g(course="breakfast", exclude=["sugar"], attributes=["healthy"])),
    ("ragi breakfast for my parents", g(include=["millet"], course="breakfast")),
    ("vegan butter chicken", g(diets=["vegan"], include=[])),
    ("something mexican with beans, dairy free", g(cuisine="mexican", include=["kidney_beans"], diets=["dairy_free"])),
    ("can you plan 4 dinners without potatoes", g(intent="plan", days=4, course="dinner", exclude=["potato"])),
    ("show my cooking history", g(intent="history")),
    ("milk allergy, need a dessert", g(diets=["dairy_free"], course="dessert")),
    ("ghar ka khana, simple dal chawal type", g(include=["dal", "rice"], attributes=["comfort"])),
    ("chicken biryani for 10 guests", g(include=["chicken", "rice"], servings=10)),
    ("mutton curry but less spicy", g(include=["mutton"], attributes=["mild"])),
    ("eggs for breakfast in 10 minutes", g(include=["egg"], course="breakfast", max_time=10)),
    ("fasting food, no grains", g(diets=["sattvic"])),
    ("hold the cheese, italian pasta", g(cuisine="italian", include=["pasta"], exclude=["cheese"])),
    ("something without tomatoes or capsicum", g(exclude=["capsicum", "tomato"])),
    ("vegetarian but eggs are fine — high protein breakfast", g(diets=["eggetarian"], attributes=["high_protein"], course="breakfast")),
    ("kids lunchbox, no nuts allowed at school", g(diets=["nut_free"], course="lunch", attributes=["mild"])),
]


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for q, gold in TESTS:
            f.write(json.dumps({"query": q, "gold": gold}, ensure_ascii=False) + "\n")
    print(len(TESTS), "test queries")
