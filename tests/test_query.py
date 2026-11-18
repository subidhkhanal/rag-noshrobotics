import pytest

from forge.chat import merge
from forge.ontology import load
from forge.query import is_refinement, parse

onto = load()


@pytest.mark.parametrize(
    "msg,diets",
    [
        ("no onion no garlic vegetarian", {"no_onion_garlic", "vegetarian"}),
        ("bina pyaz lahsun ka khana", {"no_onion_garlic"}),
        ("Jain food options", {"jain"}),
        ("I'm lactose intolerant what can I make", {"lactose_free"}),
        ("recipes without nuts", {"nut_free"}),
        ("high protein non veg under 15 mins", set()),
        ("vegan recipes", {"vegan"}),
        ("peanut allergy, quick snack", {"peanut_free"}),
    ],
)
def test_diets(msg, diets):
    assert parse(onto, msg).diets == diets


def test_time_course_cuisine():
    c = parse(onto, "south indian breakfast without coconut under 20 mins")
    assert (c.max_time, c.course, c.cuisine, c.exclude) == (20, "breakfast", "south indian", {"coconut"})
    assert parse(onto, "gluten free dessert in half an hour").max_time == 30


def test_hinglish_verb_is_not_banana():
    c = parse(onto, "kuch spicy banana hai")
    assert c.include == [] and "spicy" in c.attributes


def test_conflict_is_swapped_and_text_rewritten():
    c = parse(onto, "I want butter chicken but I'm vegetarian")
    assert c.conflicts == ["chicken"] and c.swaps == {"chicken": "paneer"}
    assert "paneer" in c.include and "paneer" in c.text and "chicken" not in c.text
    v = parse(onto, "vegan butter chicken")
    assert v.swaps["chicken"] != "paneer" and "butter" in v.text


def test_exclusion_covered_by_diet_is_not_duplicated():
    assert parse(onto, "no onion no garlic vegetarian").exclude == set()


def test_multi_turn_keeps_hard_constraints_and_refines():
    s1 = parse(onto, "I have chicken tomatoes and rice")
    st = merge(parse(onto, ""), s1, "I have chicken tomatoes and rice")
    msg2 = "something similar but spicier"
    st = merge(st, parse(onto, msg2), msg2)
    assert st.include == ["chicken", "tomato", "rice"] and "spicy" in st.attributes
    msg3 = "make it under 15 minutes"
    st = merge(st, parse(onto, msg3), msg3)
    assert st.max_time == 15 and st.include == ["chicken", "tomato", "rice"]


def test_new_topic_replaces_soft_but_keeps_diet():
    st = merge(parse(onto, ""), parse(onto, "jain breakfast"), "jain breakfast")
    msg = "paneer curry for dinner"
    new = parse(onto, msg)
    assert not is_refinement(msg, new)
    st = merge(st, new, msg)
    assert st.diets == {"jain"} and st.course == "dinner" and st.include == ["paneer"]


@pytest.mark.parametrize("msg,diets,exclude,include", [
    ("hold the onions please, and make it vegan", {"vegan"}, {"onion"}, []),
    ("lahsun pyaz ke bina kuch banao", {"no_onion_garlic"}, set(), []),
    ("I can't eat mushrooms, paneer dinner", set(), {"mushroom"}, ["paneer"]),
    ("strictly vegetarian, no mushrooms, no paneer", {"vegetarian"}, {"mushroom", "paneer"}, []),
    ("no tomatoes, capsicum or onions please", set(), {"tomato", "capsicum", "onion"}, []),
])
def test_refusals_never_become_wanted_ingredients(msg, diets, exclude, include):
    c = parse(onto, msg)
    assert c.diets == diets and c.exclude == exclude and c.include == include


def test_bare_time_limits():
    assert parse(onto, "ready in 25").max_time == 25
    assert parse(onto, "15 minutes mexican fried rice").max_time == 15


@pytest.mark.parametrize("msg,exclude", [
    ("my kid can't eat cashews", {"cashew"}),
    ("I can't stand mushrooms. paneer dinner?", {"mushroom"}),
    ("I cannot have peas", {"peas"}),
    ("I won't eat eggs", {"egg"}),
])
def test_contraction_refusals(msg, exclude):
    c = parse(onto, msg)
    assert exclude <= c.exclude and not exclude & set(c.include)


def test_claude_gate_skips_refusals_rules_understood():
    from forge.llm_parser import should_call
    from forge.query import normalize_message
    for msg in ["my kid can't eat cashews", "I can't stand mushrooms", "hold the onions please", "no onion no garlic",
                "no mushrooms, no paneer", "without potatoes and without peas"]:
        assert not should_call(normalize_message(onto, msg), parse(onto, msg).understood), msg


def test_repeated_negation_before_dish_word_still_refuses():
    c = parse(onto, "no mushrooms, no paneer dinner")
    assert {"mushroom", "paneer"} <= c.exclude and "paneer" not in c.include and c.course == "dinner"


def test_later_refusal_does_not_match_inside_earlier_word():
    c = parse(onto, "no eggplant or egg")
    assert "egg" in c.exclude and "egg" not in c.text.split()


def test_span_removal_is_whole_word_only():
    from forge.query import remove_spans
    out = remove_spans(" no paneer and paneertikka masala ", ["no paneer", "masala"])
    assert "paneertikka" in out and "masala" not in out and "no paneer" not in out


def test_gate_and_parser_share_normalisation():
    from forge.llm_parser import should_call
    from forge.query import normalize_message
    for msg in ["I'm allergic to sesame", "I've got no eggs, she's vegan", "kuch spicy banana hai"]:
        assert not should_call(normalize_message(onto, msg), parse(onto, msg).understood), msg


def test_dish_word_sets_course_but_stays_searchable():
    c = parse(onto, "carrot halwa dessert")
    assert c.course == "dessert" and "halwa" in c.text and "dessert" not in c.text
