"""The distilled classifier: answers confident follow-ups about on-screen dishes without Claude, defers otherwise."""
import pytest

from forge import intent_model
from forge.chat import Assistant
from forge.search import Engine

pytestmark = pytest.mark.skipif(not intent_model.available(), reason="no trained intent model")
assistant = Assistant(Engine())


@pytest.fixture(scope="module")
def plan():
    return assistant._chat(None, "plan my week, jain dinners", understand=False)["state"]


@pytest.fixture(scope="module")
def shown():
    return assistant._chat(None, "paneer dinner", understand=False)["state"]


def local(state, msg):
    return assistant._distilled(state, msg)


def test_features_ignore_case_and_spacing():
    assert intent_model.features("explain monday", "plan") == intent_model.features("Explain  Monday", "plan")


def test_plan_day_ingredients(plan):
    out = local(plan, "what do I need for the saturday one")
    sat = dict(zip(plan["last_labels"], plan["last_ids"]))["Sat"]
    assert out and out["mode"] == "detail" and out["results"][0]["id"] == sat
    assert out["detail"]["ingredients"] and not out["detail"]["steps"]
    assert out["trace"]["understood_by"] == "classifier"


def test_ordinal_steps(shown):
    out = local(shown, "how do I make the third one")
    assert out and out["results"][0]["id"] == shown["last_ids"][2] and out["detail"]["steps"]


def test_hindi_ordinal_is_right_or_deferred(shown):
    out = local(shown, "दूसरी वाली में क्या क्या लगेगा")
    assert out is None or (out["results"][0]["id"] == shown["last_ids"][1] and not out["detail"]["steps"])


def test_defers_on_unresolved_reference(plan):
    assert local(plan, "walk me through the end-of-week one") is None


def test_defers_on_new_request(shown):
    assert local(shown, "one more") is None
    assert local(shown, "how to make chole") is None


def test_defers_without_screen():
    assert local(None, "is it healthy") is None


def test_defers_general_questions(shown):
    for msg in ("why is ginger not allowed in jain food?", "tell me a joke", "thanks, this was really helpful!"):
        assert local(shown, msg) is None, msg
