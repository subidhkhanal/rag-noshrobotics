import pytest

from forge import commander
from forge.chat import Assistant
from forge.search import Engine

assistant = Assistant(Engine())


def test_validate_drops_unknown_and_offscreen():
    assert commander.validate({"action": "launch"}, ["1"]) is None
    v = commander.validate({"action": "show_recipe", "items": ["Fri", "9"], "want": "steps"}, ["1", "2"])
    assert v["items"] == []  # nothing on screen matches: never invent a card
    v = commander.validate({"action": "ask_about", "items": ["all"], "aspects": ["healthy", "bogus"]}, ["Mon", "Tue"])
    assert v["items"] == ["Mon", "Tue"] and v["aspects"] == ["healthy"]


def test_validate_nothing_on_screen_asks_instead_of_guessing():
    v = commander.validate({"action": "show_recipe", "items": ["Mon"]}, [])
    assert v["action"] == "clarify" and v["clarify"]


def test_validate_clarify_with_one_named_item_acts():
    v = commander.validate({"action": "clarify", "items": ["2"], "clarify": "the second?"}, ["1", "2"])
    assert v["action"] == "show_recipe" and v["items"] == ["2"]


def test_validate_clarify_without_question_never_picks_a_card():
    v = commander.validate({"action": "clarify", "items": [], "clarify": ""}, ["1", "2"])
    assert v["action"] == "clarify" and "Which dish" in v["clarify"]


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def install(cmd):
        monkeypatch.setattr(commander, "enabled", lambda: True)
        monkeypatch.setattr(commander, "command", lambda *a, **k: calls.append(a) or cmd)
        return calls
    return install


@pytest.fixture(scope="module")
def plan():
    out = assistant.chat(None, "plan my week, Jain dinners")
    return out["state"]


def test_plan_cards_have_day_labels(plan):
    assert plan["last_labels"][:3] == ["Mon", "Tue", "Wed"]


def test_command_show_recipe_uses_label(fake, plan):
    calls = fake(commander.validate({"action": "show_recipe", "items": ["Fri"], "want": "steps"}, plan["last_labels"]))
    out = assistant.chat(plan, "walk me through the end-of-week one")
    fri = dict(zip(plan["last_labels"], plan["last_ids"]))["Fri"]
    assert calls and out["mode"] == "detail" and out["results"][0]["id"] == fri and out["detail"]["steps"]
    screen = calls[0][1]
    assert ("Fri", next(n for lab, n in screen if lab == "Fri")) in screen  # the model saw labelled cards


def test_command_cannot_touch_restrictions(fake, plan):
    fake(commander.validate({"action": "search", "items": [], "want": "both"}, plan["last_labels"]))
    out = assistant.chat(plan, "forget all that, give me anything")
    assert "jain" in out["state"]["diets"]  # only the chip × removes a restriction


def test_explicit_day_reference_needs_no_model(fake, plan):
    calls = fake(None)
    out = assistant.chat(plan, "explain monday")
    mon = dict(zip(plan["last_labels"], plan["last_ids"]))["Mon"]
    assert out["results"][0]["id"] == mon and calls == []


@pytest.mark.parametrize("msg", ["jain dinner", "paneer tikka", "no onion no garlic dal"])
def test_obvious_requests_skip_the_model(fake, msg):
    calls = fake(None)
    assistant.chat(None, msg)
    assert calls == []


def test_model_unavailable_falls_back_to_rules(fake, plan):
    fake(None)
    out = assistant.chat(plan, "how do I make wednesday's dinner")
    wed = dict(zip(plan["last_labels"], plan["last_ids"]))["Wed"]
    assert out["mode"] == "detail" and out["results"][0]["id"] == wed


def test_recent_turns_are_kept_and_bounded():
    st = None
    for m in ["hi", "jain dinner", "the first one", "is it spicy", "thanks"]:
        st = assistant.chat(st, m)["state"]
    assert len(st["recent"]) == 4 and st["recent"][-1]["u"] == "thanks"
