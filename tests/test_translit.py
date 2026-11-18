from collections import Counter

import pytest

from forge.ontology import load
from forge.query import parse
from forge.translit import Speller, deva_to_latin

onto = load()


@pytest.mark.parametrize("deva,latin", [
    ("खिचड़ी", "khichdi"), ("बिरयानी", "biryani"), ("राजमा", "rajma"), ("पराठा", "paratha"),
    ("इडली", "idli"), ("सांभर", "sambhar"), ("भरता", "bharta"), ("लड्डू", "laddu"), ("पुलाव", "pulao"),
    ("दाल तड़का", "dal tadka"), ("पालक", "palak"),
])
def test_romanization(deva, latin):
    assert deva_to_latin(deva) == latin


def test_devanagari_refusal_of_unaliased_ingredient_is_enforced():
    # पालक has no Devanagari alias in the ontology; romanization must still turn it into a refusal.
    c = parse(onto, "पालक के बिना दाल")
    assert "spinach" in c.exclude
    assert "spinach" not in c.include


def test_devanagari_dish_after_ke_bina_is_wanted_not_refused():
    c = parse(onto, "पालक के बिना खिचड़ी")
    assert "spinach" in c.exclude
    assert "khichdi" not in c.exclude


def test_devanagari_alias_is_whole_word_only():
    # "राइस" (rice) contains the mustard alias "राइ"; it must not become a mustard request.
    c = parse(onto, "जीरा राइस")
    assert "mustard_seed" not in c.include
    assert "cumin" in c.include


def test_devanagari_onion_garlic_still_parsed_as_diet():
    c = parse(onto, "प्याज़ लहसुन के बिना खाना")
    assert c.diets & {"no_onion_garlic"} or {"onion", "garlic"} <= c.exclude


def test_speller_maps_variants_to_corpus_spelling():
    s = Speller(Counter({"biryani": 40, "sambar": 30, "paneer": 90, "mutton": 20, "matar": 25, "kheer": 10}))
    assert s.correct("biriyani") == "biryani"
    assert s.correct("sambhar") == "sambar"
    assert s.correct("panir") == "paneer"
    assert s.correct("matan") == "mutton"  # same consonants beats the more common neighbour "matar"
    assert s.correct("xyzzy") == "xyzzy"


def test_golden_set_relevance_rules_are_satisfiable():
    from scripts.retrieval_eval import load_golden, n_relevant
    from forge.search import Engine
    e = Engine(use_dense=False)
    for item in load_golden():
        assert n_relevant(e, item) > 0, item["id"]


@pytest.mark.parametrize("query,word", [
    ("balloon-like fried wheat bread for breakfast", "puri"),
    ("thin savoury crepe made from fermented batter", "dosa"),
    ("fried vegetable balls in a rich gravy", "kofta"),
    ("beaten rice breakfast with peanuts", "poha"),
])
def test_english_descriptions_find_the_dish(query, word):
    from forge.search import Engine
    e = Engine()
    names = [r.recipe["name"].lower() for r in e.search(parse(e.onto, query), k=5).results]
    assert any(word in n or (word == "puri" and "poori" in n) for n in names), names
