import pytest

from forge.chat import Assistant
from forge.search import Engine

assistant = Assistant(Engine())


@pytest.fixture(scope="module")
def shown():
    out = assistant.chat(None, "food with no salt")
    return out["state"], [c["id"] for c in out["results"]]


@pytest.mark.parametrize("msg,idx,steps,ings", [
    ("how to make the first one", 0, True, True),
    ("what do you need for the first one", 0, False, True),
    ("the second one", 1, True, True),
    ("ingredients for #3", 2, False, True),
    ("recipe 2", 1, True, True),
    ("how do i cook the last one", -1, True, True),
    ("first one kaise banaye", 0, True, True),
])
def test_reference_to_shown_card(shown, msg, idx, steps, ings):
    state, ids = shown
    out = assistant.chat(state, msg)
    assert out["mode"] == "detail"
    assert out["results"][0]["id"] == ids[idx]
    assert bool(out["detail"]["steps"]) == steps
    assert bool(out["detail"]["ingredients"]) == ings
    assert out["state"]["exclude"] == state["exclude"]  # constraints carried, not changed


@pytest.mark.parametrize("msg", ["show me something spicy", "make it spicy", "what can i do with paneer", "one more"])
def test_new_requests_are_not_references(shown, msg):
    state, _ = shown
    assert assistant.chat(state, msg)["mode"] != "detail"


def test_no_reference_without_shown_cards():
    assert assistant.chat(None, "how to make the first one")["mode"] != "detail"


def test_reply_has_no_ruled_out_sentence(shown):
    out = assistant.chat(None, "no onion no garlic dal")
    assert "ruled out" not in out["reply"].lower()
