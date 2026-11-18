import types

import pytest

from forge import agent
from forge.chat import Assistant
from forge.search import Engine

assistant = Assistant(Engine())


@pytest.fixture(scope="module")
def shown():
    return assistant.chat(None, "paneer dinner")["state"]


def test_reply_check_rejects_invented_dish(shown):
    a = assistant.agent
    some_other = next(n for n in assistant._long_names if "paneer" not in n)
    assert a._check_reply(f"You could also try {some_other.title()}.", set(shown["last_ids"]))


def test_reply_check_allows_shown_dishes(shown):
    a = assistant.agent
    name = next(r["name"] for r in assistant.engine.recipes if r["id"] == shown["last_ids"][0])
    assert a._check_reply(f"{name} is the quickest here.", set(shown["last_ids"])) is None


@pytest.mark.parametrize("text", ["It has about 350 kcal per serving.", "This is 100% safe for allergies."])
def test_reply_check_rejects_numbers_and_absolute_safety(shown, text):
    assert assistant.agent._check_reply(text, set(shown["last_ids"]))


class FakeClient:
    """Replays a scripted Claude conversation: each create() returns the next response."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def with_options(self, **_):
        return self

    @property
    def messages(self):
        return self

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)

    def stream(self, **kw):
        """Streaming twin of create(): text blocks arrive as text_delta events, then the final message."""
        resp = self.create(**kw)

        class Stream:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def __iter__(self_inner):
                for b in resp.content:
                    if b.type == "text":
                        yield types.SimpleNamespace(type="content_block_delta",
                                                    delta=types.SimpleNamespace(type="text_delta", text=b.text))

            def get_final_message(self_inner):
                return resp
        return Stream()


def text_msg(t):
    return types.SimpleNamespace(stop_reason="end_turn", content=[types.SimpleNamespace(type="text", text=t)])


def tool_msg(name, args):
    return types.SimpleNamespace(stop_reason="tool_use", content=[
        types.SimpleNamespace(type="tool_use", id="t1", name=name, input=args)])


@pytest.fixture
def fake(monkeypatch):
    def install(script):
        client = FakeClient(script)
        monkeypatch.setattr(agent.llm_parser, "_get_client", lambda: client)
        monkeypatch.setattr(agent.guard, "allow_claude", lambda: True)
        monkeypatch.setattr(agent, "enabled", lambda: True)
        return client
    return install


def test_question_about_a_diet_does_not_lock_it(fake):
    fake([text_msg("Ginger is a root, and Jain practice avoids roots to minimise harm to living things.")])
    out = assistant.agent.turn(None, "why is ginger not allowed in jain food?", None)
    assert out["state"]["diets"] == [] and out["mode"] == "chat"


def test_users_own_restriction_is_locked_even_if_model_ignores_it(fake):
    fake([text_msg("Sure, let me know what you'd like.")])
    out = assistant.agent.turn(None, "I'm vegan and allergic to peanuts", None)
    assert {"vegan", "peanut_free"} <= set(out["state"]["diets"])


def test_search_tool_keeps_locked_restrictions_and_shows_cards(fake):
    state = assistant.chat(None, "jain dinner")["state"]
    client = fake([tool_msg("search_recipes", {"request": "start over, anything with garlic"}),
                   text_msg("Here are a few more Jain dinners.")])
    out = assistant.agent.turn(state, "show me something else", None)
    assert "jain" in out["state"]["diets"]  # "start over" in a tool request can't clear restrictions
    assert out["results"] and all(c["checks"][0]["status"] != "block" for c in out["results"])
    assert "Shown to the user as cards" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_get_recipe_tool_resolves_on_screen_label(fake, shown):
    fake([tool_msg("get_recipe", {"item": "3", "want": "steps"}), text_msg("Here's how to make it.")])
    out = assistant.agent.turn(shown, "walk me through the middle one", None)
    assert out["mode"] == "detail" and out["results"][0]["id"] == shown["last_ids"][2] and out["detail"]["steps"]


def test_invented_dish_is_rewritten_then_falls_back(fake, shown):
    bogus = next(n for n in assistant._long_names if "paneer" not in n).title()
    fake([text_msg(f"Try {bogus}!"), text_msg(f"Really, try {bogus}!")])
    with pytest.raises(agent.AgentError):
        assistant.agent.turn(shown, "anything else you'd suggest?", None)


def test_stream_emits_status_text_then_final(fake, shown):
    fake([tool_msg("check_dishes", {"items": ["2"], "question": "is it spicy?"}), text_msg("It's mildly spiced.")])
    events = list(assistant.chat_stream(shown, "is the second one spicy, like for a kid?"))
    kinds = [e["type"] for e in events]
    assert "status" in kinds and "delta" in kinds and kinds[-1] == "final"
    assert events[-1]["data"]["reply"] == "It's mildly spiced." and events[-1]["data"]["state"]["recent"]


def test_stream_falls_back_and_clears_partial_text(fake, shown):
    bogus = next(n for n in assistant._long_names if "paneer" not in n).title()
    fake([text_msg(f"Try {bogus}!"), text_msg(f"Really, try {bogus}!")])
    events = list(assistant.chat_stream(shown, "anything else you'd suggest?"))
    assert "reset" in [e["type"] for e in events]  # streamed text withdrawn
    assert events[-1]["type"] == "final" and not events[-1]["data"].get("agent")  # deterministic answer instead
