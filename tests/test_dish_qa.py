import pytest

from forge.chat import Assistant
from forge.search import Engine

assistant = Assistant(Engine())


@pytest.fixture(scope="module")
def shown():
    out = assistant.chat(None, "paneer dinner")
    return out["state"], [c["id"] for c in out["results"]], [c["name"] for c in out["results"]]


def ask(state, msg):
    return assistant.chat(state, msg)


@pytest.mark.parametrize("msg", ["is it healthy", "is it healthy?", "kya ye healthy hai", "how many calories does it have"])
def test_health_question_answers_about_the_dish_not_a_new_search(shown, msg):
    state, ids, names = shown
    out = ask(state, msg)
    assert out["mode"] == "detail" and out["results"][0]["id"] == ids[0]
    assert "nutrition data" in out["reply"]  # honest about what it can't know


def test_ordinal_question_targets_that_card(shown):
    state, ids, names = shown
    out = ask(state, "is the second one spicy?")
    assert out["mode"] == "detail" and out["results"][0]["id"] == ids[1] and "spic" in out["reply"].lower()


def test_ingredient_and_diet_questions_use_real_checks(shown):
    state, ids, _ = shown
    garlic = ask(state, "does the second one have garlic")["reply"]
    assert garlic.split(": ", 1)[1].startswith(("Yes", "No", "Possibly"))
    jain = ask(state, "kya ye jain hai")["reply"]
    assert "Jain" in jain


def test_question_about_all_shown_dishes(shown):
    state, ids, names = shown
    out = ask(state, "are these vegan?")
    assert out["mode"] == "detail" and all(n in out["reply"] for n in names)


def test_it_means_the_dish_in_focus(shown):
    state, ids, _ = shown
    state = ask(state, "how to make the third one")["state"]
    out = ask(state, "is it healthy")
    assert out["results"][0]["id"] == ids[2]


@pytest.mark.parametrize("msg", ["what was the last recipe", "what did you suggest earlier",
                                 "remind me of the previous recipes", "pichli recipe kya thi"])
def test_recall_lists_what_was_shown(shown, msg):
    state, ids, names = shown
    out = ask(state, msg)
    assert out["mode"] == "search" and [c["id"] for c in out["results"]] == ids
    assert names[0] in out["reply"]


def test_recall_with_nothing_shown():
    out = assistant.chat(None, "what was the last recipe")
    assert "haven't suggested" in out["reply"] and not out.get("results")


@pytest.mark.parametrize("msg", ["is there anything healthy", "make it spicy", "something healthy for dinner",
                                 "healthy breakfast"])
def test_new_requests_stay_searches(shown, msg):
    state, _, _ = shown
    assert ask(state, msg)["mode"] == "search"
