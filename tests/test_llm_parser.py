from forge.llm_parser import combine
from forge.ontology import load
from forge.parser_schema import normalize
from forge.query import parse

onto = load()


def test_model_can_add_but_not_remove_restrictions():
    rules = parse(onto, "vegan dinner without peanuts in 30 minutes")
    model = normalize({"diets": ["no_onion_garlic"], "exclude": ["mushroom", "not_an_id"], "max_time": 45,
                       "course": "lunch"})
    c = combine(onto, rules, model)
    assert {"vegan", "peanut_free", "no_onion_garlic"} <= c.diets
    assert c.exclude == {"mushroom"} and c.max_time == 30 and c.course == "dinner"


def test_model_fills_rule_gaps():
    c = combine(onto, parse(onto, "hold the onions please, make it vegan"),
                normalize({"diets": ["vegan"], "exclude": ["onion"]}))
    assert c.exclude == {"onion"}


def test_failed_model_call_leaves_rules_untouched():
    rules = parse(onto, "jain breakfast")
    assert combine(onto, rules, None) is rules


def test_model_words_are_mapped_to_ontology_ids():
    c = combine(onto, parse(onto, "dinner"), normalize({"exclude": ["onions", "kaju", "unicorn"], "include": ["chana"]}))
    assert c.exclude == {"onion", "cashew"} and c.include == ["chickpeas"]


def test_gate_skips_plain_queries():
    from forge.llm_parser import should_call
    assert not should_call("paneer butter masala")
    assert should_call("hold the onions please") and should_call("lahsun pyaz ke bina")
