import pytest

from forge.chat import small_talk_kind


@pytest.mark.parametrize("msg", [
    "what can you do", "what can you ddo", "what do you do", "what do you do?", "What else can you do",
    "what can u do for me", "how can you help", "how do I use this", "how does this work", "help",
    "what are you good at", "what is your purpose", "features", "tum kya kar sakte ho", "aap kya karte hain",
    "what should i ask", "hi, what can you do", "ok so what can you do?",
])
def test_help_questions(msg):
    assert small_talk_kind(msg) == "help"


@pytest.mark.parametrize("msg", ["who are you", "what are you", "who made you"])
def test_about_questions(msg):
    assert small_talk_kind(msg) == "about"


@pytest.mark.parametrize("msg", [
    "what can I cook with paneer", "what do you suggest for jain dinner", "no onion no garlic vegetarian",
    "what should I make for dinner", "help me find a vegan lunch without peanuts please",
    "what can i do with paneer", "what do you do with okra", "how do i use this paneer", "help me find vegan lunch",
])
def test_food_requests_are_not_small_talk(msg):
    assert small_talk_kind(msg) is None
