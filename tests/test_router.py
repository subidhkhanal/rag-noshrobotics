import pytest

from forge import router
from forge.chat import Assistant
from forge.search import Engine

assistant = Assistant(Engine())


def test_validate_rejects_bad_shapes():
    assert router.validate({"intent": "launch_missiles"}, 3) is None
    assert router.validate("followup", 3) is None
    assert router.validate({"intent": "followup", "card": 9, "want": "steps"}, 3)["card"] == 0  # off-screen -> top
    assert router.validate({"intent": "followup", "card": 2, "want": "steps"}, 3) == \
        {"intent": "followup", "card": 1, "want": "steps"}
    assert router.validate({"intent": "followup", "card": 1, "want": "x"}, 0)["intent"] == "search"
    assert router.validate({"intent": "help", "card": 2, "want": "both"}, 3)["card"] is None


@pytest.fixture
def fake_router(monkeypatch):
    calls = []

    def install(answer):
        monkeypatch.setattr(router, "enabled", lambda: True)
        monkeypatch.setattr(router, "route", lambda m, names: calls.append((m, names)) or answer)
        return calls
    return install


@pytest.fixture(scope="module")
def shown():
    out = assistant.chat(None, "jain dinner")
    return out["state"], [c["id"] for c in out["results"]]


def test_unclear_message_is_routed_to_help(fake_router):
    calls = fake_router({"intent": "help", "card": None, "want": "both"})
    out = assistant.chat(None, "so umm what is it you actually do here")
    assert calls and out["mode"] == "chat" and "I recommend recipes" in out["reply"]


def test_pointing_followup_uses_the_named_card(fake_router, shown):
    state, ids = shown
    calls = fake_router({"intent": "followup", "card": 2, "want": "ingredients"})
    out = assistant.chat(state, "what about the one in the middle")  # rules can't tell which card
    assert out["mode"] == "detail" and out["results"][0]["id"] == ids[2]
    assert out["detail"]["ingredients"] and not out["detail"]["steps"]
    assert len(calls[0][1]) == len(ids)  # the router saw the on-screen names


def test_router_cannot_change_constraints(fake_router, shown):
    state, _ = shown
    fake_router({"intent": "reset", "card": None, "want": "both"})
    # Even a (wrong) reset only clears via the explicit reset reply; a followup never edits diets.
    fake_router({"intent": "followup", "card": 0, "want": "both"})
    out = assistant.chat(state, "is this one ok")
    assert out["state"]["diets"] == state["diets"] == ["jain"]


def test_router_search_falls_through_to_pipeline(fake_router, shown):
    state, _ = shown
    fake_router({"intent": "search", "card": None, "want": "both"})
    out = assistant.chat(state, "something like that but with paneer")
    assert out["mode"] == "search" and "jain" in out["state"]["diets"]


def test_clear_requests_never_call_the_router(fake_router, shown):
    state, _ = shown
    calls = fake_router({"intent": "off_topic", "card": None, "want": "both"})
    for m in ["no onion no garlic dal", "one more", "how to make the first one", "what can you do", "hi"]:
        assistant.chat(state, m)
    assert calls == []


def test_router_unavailable_keeps_old_behaviour(monkeypatch):
    monkeypatch.setattr(router, "enabled", lambda: True)
    monkeypatch.setattr(router, "route", lambda m, names: None)
    out = assistant.chat(None, "blorp zzz")
    assert "couldn't find a food request" in out["reply"]
