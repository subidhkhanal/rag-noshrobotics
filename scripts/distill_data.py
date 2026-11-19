"""Generate training data for the distilled command classifier (teacher: Claude Sonnet).

Each request asks the teacher for varied user messages for ONE labelled situation, so labels come from
construction, not from the teacher's judgement. data/eval/dialogue.yaml is never shown or used here: it stays a
held-out test set.

    ANTHROPIC_API_KEY=... python scripts/distill_data.py      # writes data/distill/train.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "distill" / "train.jsonl"

# (action, ref, want, screen, description of what the user does)
SITUATIONS = [
    ("show_recipe", "specific", "steps", "plan", "asks how to cook / the method of one day's dish from a weekly plan, naming the day (Monday, mon, somvar, सोमवार...)"),
    ("show_recipe", "specific", "ingredients", "plan", "asks what ingredients are needed for one day's dish in a weekly plan, naming the day"),
    ("show_recipe", "specific", "both", "plan", "asks to explain / show / tell more about one day's dish in a weekly plan, naming the day"),
    ("show_recipe", "specific", "steps", "list", "asks how to make one numbered dish on screen (the first one, #2, the 3rd, the last one, middle one, pehla wala...)"),
    ("show_recipe", "specific", "ingredients", "list", "asks what goes into / ingredients of one numbered dish on screen"),
    ("show_recipe", "specific", "both", "list", "asks to explain, open, show details of one dish on screen, by position or by part of its name (the palak one, the makhani one)"),
    ("show_recipe", "specific", "steps", "focus", "asks how to make 'it' / 'this' / 'that one' (a dish already being discussed)"),
    ("ask_about", "specific", "both", "list", "asks a factual question about ONE dish on screen: is it healthy, spicy, jain, vegan; does it contain garlic/onion/nuts; how long it takes; how many it serves"),
    ("ask_about", "specific", "both", "plan", "asks a factual question about one day's dish in a weekly plan (is monday's dish spicy, how long does friday's take)"),
    ("ask_about", "specific", "both", "focus", "asks a factual question about 'it' / 'this one' (the dish being discussed): is it spicy, healthy, how long, does it have onion"),
    ("ask_about", "all", "both", "list", "asks a comparison or question across ALL dishes on screen: which of these is quickest, which ones are jain, are these vegan, which is least spicy"),
    ("recall", "none", "both", "list", "asks what was suggested or shown earlier in the chat: what was the last recipe, go back to the list, show those again, what did you suggest"),
    ("search", "none", "both", "list", "asks for DIFFERENT or NEW dishes while some are on screen: more options, something lighter, spicier, not these, something with paneer instead, a dessert now"),
    ("search", "none", "both", "none", "asks for recipes or dishes (a new request) with a diet, ingredient, cuisine, time, mood or occasion"),
    ("plan", "none", "both", "any", "asks for a meal plan for several days or a week"),
    ("greeting", "none", "both", "any", "greets the assistant (hi, hello, namaste, good morning), possibly with small talk about their day"),
    ("thanks", "none", "both", "any", "thanks the assistant or says the answer helped"),
    ("help", "none", "both", "any", "asks what the assistant can do or how to use it"),
    ("about", "none", "both", "any", "asks who or what the assistant is, or who made it"),
    ("bye", "none", "both", "any", "says goodbye or ends the conversation"),
    ("off_topic", "none", "both", "any", "says something unrelated to food or cooking (weather, jokes, cricket, random typing)"),
]
STYLES = ["casual English", "Hinglish (Hindi in Roman letters mixed with English)", "Hindi in Devanagari script",
          "short and terse, with typos", "polite and wordy", "Indian English, chatty"]
SYSTEM = """You write realistic messages that users type into a recipe chat app for a home cooking robot.
Return only a JSON object {"messages": [...]} with distinct, varied messages exactly matching the situation.
Vary wording, length and reference style. Do not add messages that fit a different situation."""
SCHEMA = {"type": "object", "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
          "required": ["messages"], "additionalProperties": False}


def generate(client, situation, style, n=25) -> list[dict]:
    action, ref, want, screen, what = situation
    resp = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=4000, system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"Situation: the user {what}.\nStyle: {style}.\n"
                                              f"Write {n} messages."}])
    msgs = json.loads(next(b.text for b in resp.content if b.type == "text"))["messages"]
    return [{"text": m.strip(), "action": action, "ref": ref, "want": want, "screen": screen, "style": style}
            for m in msgs if m.strip()]


def main() -> None:
    client = anthropic.Anthropic(max_retries=3)
    jobs = [(s, st) for s in SITUATIONS for st in STYLES]
    random.seed(0)
    rows = []
    with ThreadPoolExecutor(8) as pool:
        for batch in pool.map(lambda j: generate(client, *j), jobs):
            rows += batch
            print(len(rows), flush=True)
    seen, unique = set(), []
    for r in rows:
        key = r["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in unique) + "\n", encoding="utf-8")
    print(f"wrote {len(unique)} examples to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
