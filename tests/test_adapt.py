from forge.adapt import adapt, load_substitutions
from forge.ontology import load
from forge.query import parse
from forge.recipes import parse_ingredients

onto = load()


def recipe(name, *lines):
    return {"name": name, "ingredients": parse_ingredients(onto, ", ".join(lines))}


def test_every_substitution_references_known_ingredients():
    for src, opts in load_substitutions().items():
        assert src in onto.ingredients, src
        for o in opts:
            assert all(t in onto.ingredients for t in o["to"]), (src, o)


def test_ginger_garlic_paste_becomes_hing_and_sonth_for_jain():
    r = recipe("Methi Dal", "1 cup Toor Dal", "1 tablespoon Ginger Garlic Paste", "Salt - to taste")
    a = adapt(onto, r, parse(onto, "jain"))
    assert a.adaptable and a.swaps[0].replace == "ginger_garlic_paste"
    assert set(a.swaps[0].with_) == {"asafoetida", "dry_ginger"}


def test_vegan_skips_swaps_that_are_not_vegan():
    r = recipe("Kadai Paneer", "200 grams Paneer", "2 tablespoons Fresh cream", "1 tablespoon Ghee")
    a = adapt(onto, r, parse(onto, "vegan"))
    got = {s.replace: s.with_ for s in a.swaps}
    assert a.adaptable and got["paneer"] == ["tofu"] and got["cream"] == ["cashew"] and got["ghee"] == ["oil"]
    assert a.main_swapped == ["paneer"]


def test_nut_allergy_never_gets_a_nut_swap():
    r = recipe("Korma", "2 tablespoons Fresh cream", "10 Cashew nuts")
    a = adapt(onto, r, parse(onto, "vegan nut free"))
    assert a.adaptable
    assert all(not onto.effective_tags(t) & {"tree_nut", "peanut", "dairy"} for s in a.swaps for t in s.with_)


def test_unswappable_recipe_is_not_adaptable():
    r = recipe("Mutton Chops", *[f"1 Whole Egg {i}" for i in range(6)])
    assert not adapt(onto, r, parse(onto, "vegan")).adaptable


def test_safer_swap_for_brand_dependent_ingredient():
    r = recipe("Pasta", "2 tablespoons Tomato Ketchup", "1 cup Rice noodles")
    a = adapt(onto, r, parse(onto, "no onion no garlic"))
    assert a.safer and a.safer[0].replace == "ketchup"
