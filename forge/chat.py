"""Multi-turn dialogue state and grounded response generation."""
from __future__ import annotations

import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import cached_property

import httpx

from . import llm_parser, router
from .adapt import adapt
from .plan import make_plan
from .ontology import tokenize
from .profile import build_profile
from .query import (ATTRIBUTE_RES, COMMON_WORDS, COURSE_RES, CUISINE_GROUPS, Constraints, apply_diet_conflicts,
                    is_refinement, normalize_message, parse, rewrite_swapped_text)
from .search import Engine, Result, SearchOutput

MAX_SHOWN = 60
LAST_IDS = 14  # a weekly plan shows up to 14 cards

# Follow-ups about a dish already on screen: "how to make the first one", "what do I need for #2", "that one".
_ORDINALS = {"first": 0, "1st": 0, "one": 0, "1": 0, "second": 1, "2nd": 1, "two": 1, "2": 1, "third": 2, "3rd": 2,
             "three": 2, "3": 2, "fourth": 3, "4th": 3, "four": 3, "4": 3, "fifth": 4, "5th": 4, "five": 4, "5": 4,
             "last": -1, "pehla": 0, "pehli": 0, "pehle": 0, "doosra": 1, "dusra": 1, "doosri": 1, "dusri": 1,
             "teesra": 2, "tisra": 2, "teesri": 2}
REFERENCE = re.compile(
    r"(?<!\w)(?:the\s+)?(?:(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|pehla|pehli|pehle|doosra|dusra|"
    r"doosri|dusri|teesra|tisra|teesri)\s*(?:one|recipe|dish|wala|wali|vala|vali)?|"
    r"(?:#|no\.?\s*|number\s+|option\s+|recipe\s+|dish\s+)(one|two|three|four|five|[1-5])|"
    r"(that|this|it)\s*(?:one|recipe|dish)?)\b(?=\s|$|[?.!,])", re.I)
ASK_STEPS = re.compile(r"\b(how (do|to|can|should) (i |you |we )?(make|cook|prepare)|steps?|method|instructions?|"
                       r"recipe for|procedure|kaise (banaye|banate|banaen|banau|bane)|vidhi|banane ka tarika|"
                       r"show me|tell me|full recipe|details?)\b", re.I)
FOOD_STOP = COMMON_WORDS | {"the", "and", "with", "style", "recipe", "over", "start", "made", "home", "make", "easy",
                            "best", "quick", "simple", "classic", "special", "homemade", "delicious", "done", "umm",
                            "hmm", "lol", "actually", "here", "really", "basically", "stuff", "thing", "things"}
# Words that may point at an on-screen card; when the rules can't resolve them, the router is asked.
POINTING = re.compile(r"(?<!\w)(one|first|second|third|fourth|fifth|last|that|this|those|these|wala|wali|vala|"
                      r"#\s*\d|number\s+\d|option\s+\d|upar|above|top)(?!\w)", re.I)
# Naming a dish counts as a reference only after a clear ask ("show me something spicy" is a new search).
ASK_BY_NAME = re.compile(r"\b(how (do|to|can|should) (i |you |we )?(make|cook|prepare)|(recipe|ingredients?|steps?|"
                         r"method|instructions?) (for|of)|what (do|will|would|should) (i|you|we) need for|"
                         r"kaise (banaye|banate|banaen|banau|bane))\b", re.I)
ASK_INGREDIENTS = re.compile(r"\b(ingredients?|what (do|will|would|should) (i|you|we) need|what goes in(to)?|"
                             r"what('s| is) in|shopping|samagri|kya chahiye|kya kya chahiye|need for|items?)\b", re.I)


@dataclass
class Session:
    """Conversation state. The server is stateless: the client sends this back with every request."""
    state: Constraints = field(default_factory=Constraints)
    shown: list[int] = field(default_factory=list)
    last_top: int | None = None
    last_ids: list[int] = field(default_factory=list)  # the cards of the latest reply, in order ("the first one")

    def to_dict(self) -> dict:
        st = self.state
        return {
            "diets": sorted(st.diets), "exclude": sorted(st.exclude), "include": list(st.include),
            "pantry_mode": st.pantry_mode, "max_time": st.max_time, "course": st.course, "cuisine": st.cuisine,
            "attributes": sorted(st.attributes), "servings": st.servings, "prefer_non_veg": st.prefer_non_veg,
            "text": st.text, "shown": self.shown[-MAX_SHOWN:], "last_top": self.last_top,
            "last_ids": self.last_ids[:LAST_IDS],
        }


def session_from(engine: Engine, data: dict | None) -> Session:
    """Rebuild a session from client-supplied state, keeping only values the server recognises."""
    if not isinstance(data, dict):
        return Session()
    onto = engine.onto

    def ids(key, allowed):
        v = data.get(key)
        return [x for x in v if isinstance(x, str) and x in allowed] if isinstance(v, list) else []

    def bounded_int(key, lo, hi):
        v = data.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi else None

    st = Constraints(
        diets=set(ids("diets", onto.diets)),
        exclude=set(ids("exclude", onto.ingredients)),
        include=list(dict.fromkeys(ids("include", onto.ingredients)))[:10],
        pantry_mode=data.get("pantry_mode") is True,
        max_time=bounded_int("max_time", 1, 600),
        course=data.get("course") if data.get("course") in COURSE_RES else None,
        cuisine=data.get("cuisine") if data.get("cuisine") in CUISINE_GROUPS else None,
        attributes=set(ids("attributes", ATTRIBUTE_RES)),
        servings=bounded_int("servings", 1, 100),
        prefer_non_veg=data.get("prefer_non_veg") is True,
        text=re.sub(r"[^a-z0-9ऀ-ॿ ]", " ", str(data.get("text", ""))[:300].lower()).strip(),
    )
    shown = data.get("shown")
    shown = [x for x in shown if isinstance(x, int) and x in engine.by_id][-MAX_SHOWN:] if isinstance(shown, list) else []
    last = data.get("last_top")
    last_ids = data.get("last_ids")
    last_ids = ([x for x in last_ids if isinstance(x, int) and x in engine.by_id][:LAST_IDS]
                if isinstance(last_ids, list) else [])
    return Session(state=st, shown=shown, last_top=last if isinstance(last, int) and last in engine.by_id else None,
                   last_ids=last_ids)


def merge(prev: Constraints, new: Constraints, message: str, onto=None) -> Constraints:
    if "reset" in new.intents:
        new.intents.discard("reset")
        return new
    merged = _merge(prev, new, message)
    if onto is not None:
        # A diet added later ("actually I'm vegetarian") must swap ingredients wanted in earlier turns.
        apply_diet_conflicts(onto, merged)
        rewrite_swapped_text(onto, merged)
    return merged


def _merge(prev: Constraints, new: Constraints, message: str) -> Constraints:
    merged = Constraints(
        diets=prev.diets | new.diets,
        exclude=prev.exclude | new.exclude,
        intents=new.intents,
        servings=new.servings or prev.servings,
    )
    # "different from what I usually make" is about the user's habits, not the last results: start soft prefs fresh.
    if is_refinement(message, new) and "usual" not in new.intents:
        merged.include = prev.include + [i for i in new.include if i not in prev.include]
        merged.pantry_mode = prev.pantry_mode or new.pantry_mode
        merged.max_time = new.max_time or prev.max_time
        merged.course = new.course or prev.course
        merged.cuisine = new.cuisine or prev.cuisine
        merged.attributes = (prev.attributes | new.attributes) - ({"spicy"} if "mild" in new.attributes else set()) \
            - ({"mild"} if "spicy" in new.attributes else set())
        merged.prefer_non_veg = prev.prefer_non_veg or new.prefer_non_veg
        merged.text = " ".join(t for t in (prev.text, new.text) if t).strip()
    else:
        merged.include, merged.pantry_mode = new.include, new.pantry_mode
        merged.max_time, merged.course, merged.cuisine = new.max_time, new.course, new.cuisine
        merged.attributes, merged.prefer_non_veg, merged.text = new.attributes, new.prefer_non_veg, new.text
    merged.include = [i for i in merged.include if i not in merged.exclude]
    merged.conflicts, merged.swaps = list(new.conflicts), dict(new.swaps)
    merged.generic, merged.unsupported = new.generic, new.unsupported
    return merged


class Assistant:
    def __init__(self, engine: Engine):
        self.engine = engine

    def remove_constraint(self, state: dict | None, kind: str, value: str) -> Session:
        s = session_from(self.engine, state)
        st = s.state
        if kind == "diet":
            st.diets.discard(value)
        elif kind == "exclude":
            st.exclude.discard(value)
        elif kind == "include":
            st.include = [i for i in st.include if i != value]
        elif kind in ("max_time", "course", "cuisine", "servings"):
            setattr(st, kind, None)
        elif kind == "attribute":
            st.attributes.discard(value)
        elif kind == "text":
            st.text = ""
        return s

    def chat(self, state: dict | None, message: str, history: list | None = None) -> dict:
        s = session_from(self.engine, state)
        onto = self.engine.onto
        kind = small_talk_kind(message)
        if kind:
            return self._small_talk_reply(s, kind)
        target = self._referenced_recipe(s, message)
        if target is not None:
            return self._detail_reply(s, target, message)
        profile = build_profile(self.engine, history)
        parsed = parse(onto, message)
        if "reset" in parsed.intents and not self._has_food_content(parsed, ignore_intents=True):
            return self._small_talk_reply(Session(), "reset")
        routed = self._route(s, message, parsed)
        if routed is not None:
            return routed
        used_llm = llm_parser.enabled() and llm_parser.should_call(normalize_message(onto, message), parsed.understood)
        if used_llm:
            parsed = llm_parser.combine(onto, parsed, llm_parser.llm_parse(message))
        if not self._has_food_content(parsed):
            return self._small_talk_reply(s, "unclear")
        s.state = merge(s.state, parsed, message, onto)
        st = s.state

        if "history" in parsed.intents:
            return self._history_reply(s, parsed, profile)
        if "plan" in parsed.intents:
            return self._plan_reply(s, parsed, profile)

        exclude_ids: set[int] = set()
        if st.intents & {"more", "different"}:
            exclude_ids = set(s.shown)
        anchor = s.last_top if "similar" in st.intents else None
        if anchor is not None:
            exclude_ids.add(anchor)

        out = self.engine.search(st, k=5, exclude_ids=exclude_ids, anchor_id=anchor, profile=profile)
        if out.results:
            s.shown += [r.recipe["id"] for r in out.results]
            s.last_top = out.results[0].recipe["id"]
            s.last_ids = [r.recipe["id"] for r in out.results]
        elif out.closest:
            s.last_ids = [out.closest.recipe["id"]]
        for b in out.blocked_examples:
            b["adaptation"] = adapt(onto, self.engine.recipes[self.engine.by_id[b["id"]]], st).to_dict(onto)
        cards = []
        for r in out.results:
            cd = card(r)
            if r.flagged:
                cd["safer"] = adapt(onto, r.recipe, st).to_dict(onto)["safer"]
            cards.append(cd)
        reply = self.compose(st, parsed, out, anchor)
        if st.intents & {"different", "usual"}:
            reply = (f"Based on your history ({profile.summary(onto)}), here's something different. " if not profile.empty
                     else "I don't have your cooking history yet — tap “Cooked it” on dishes you make and I'll learn "
                          "what you usually cook. For now, here's a varied selection. ") + reply
        reply = self.polish(reply, out) or reply
        s.shown = s.shown[-MAX_SHOWN:]
        return {
            "mode": "search",
            "state": s.to_dict(),
            "reply": reply,
            "results": cards,
            "closest": card(out.closest) if out.closest else None,
            "blocked_examples": out.blocked_examples,
            "constraints": chips(onto, st),
            "parsed": parsed.to_dict(),
            "trace": out.trace | {"relaxed": out.relaxed, "claude_parser": used_llm},
        }

    def _referenced_recipe(self, s: Session, message: str) -> int | None:
        """Recipe id a follow-up points at ("how to make the first one", "#2", "the mushroom matar one")."""
        if not s.last_ids:
            return None
        text = message.strip().lower()
        ask = ASK_STEPS.search(text) or ASK_INGREDIENTS.search(text)
        m = REFERENCE.search(text)
        if m:
            word = (m.group(1) or m.group(2) or "").lower()
            if word in _ORDINALS:
                idx = _ORDINALS[word]
                bare = not re.sub(r"\b(the|please|pls|plz|ok|okay|so|and)\b|[?.!,]", " ", text.replace(m.group(0), " ")).strip()
                if (ask or bare) and -len(s.last_ids) <= idx < len(s.last_ids):
                    return s.last_ids[idx]
            elif m.group(3) and ask:  # "that one" / "this recipe" / "it": the top card
                return s.last_ids[0]
        if ASK_BY_NAME.search(text):
            # By name: most of the message's distinctive words appear in one shown recipe's name.
            words = {w for w in tokenize(text) if len(w) > 3} - COMMON_WORDS - {"make", "cook", "recipe", "need",
                                                                                 "ingredient", "step", "method"}
            if words:
                best = max(s.last_ids, key=lambda rid: len(words & self.engine.name_tokens[self.engine.by_id[rid]]))
                hit = words & self.engine.name_tokens[self.engine.by_id[best]]
                if len(hit) >= max(1, (len(words) + 1) // 2):
                    return best
        return None

    def _route(self, s: Session, message: str, parsed: Constraints) -> dict | None:
        """Ask the Claude router only when the rules couldn't place the message. Returns a reply, or None to
        continue with the normal search path."""
        unclear = self._food_signal(parsed) != "strong"
        pointing = bool(s.last_ids) and bool(POINTING.search(message)) and not (
            parsed.intents & {"more", "different", "similar", "plan", "history", "usual"})
        if not (unclear or pointing) or not router.enabled():
            return None
        names = [self.engine.recipes[self.engine.by_id[rid]]["name"] for rid in s.last_ids]
        r = router.route(message, names)
        if r is None:
            return None
        if r["intent"] == "followup":
            return self._detail_reply(s, s.last_ids[r["card"]], message, want=r["want"])
        if r["intent"] in ("help", "about", "greeting", "thanks", "bye"):
            return self._small_talk_reply(s, r["intent"])
        if r["intent"] == "reset":
            return self._small_talk_reply(Session(), "reset")
        if r["intent"] == "off_topic":
            return self._small_talk_reply(s, "unclear")
        return None  # search: the normal pipeline decides (and still says "unclear" if there's nothing to search)

    def _detail_reply(self, s: Session, rid: int, message: str, want: str | None = None) -> dict:
        """Ingredients and/or steps of a dish already shown, straight from the recipe record (no generation)."""
        onto, eng = self.engine.onto, self.engine
        i = eng.by_id[rid]
        r, st = eng.recipes[i], s.state
        if want:
            want_ing, want_steps = want in ("ingredients", "both"), want in ("steps", "both")
        else:
            want_ing = bool(ASK_INGREDIENTS.search(message))
            want_steps = bool(ASK_STEPS.search(message)) or not want_ing
        result = eng._explain(i, 0.0, eng.is_flagged(i, st), st)
        cd = card(result)
        if result.flagged:
            cd["safer"] = adapt(onto, r, st).to_dict(onto)["safer"]
        steps = [x.strip() for x in re.split(r"(?<=[.!?])\s*(?=[A-Z])", r["instructions"]) if len(x.strip()) > 3]
        bits = [f"{r['total_min']} min" if r["total_min"] else "", f"serves {r['servings']}" if r["servings"] else ""]
        meta = ", ".join(b for b in bits if b)
        what = "Here's how to make" if want_steps and not want_ing else (
            "Here's what you need for" if want_ing and not want_steps else "Here's")
        reply = f"{what} {r['name']}{f' ({meta})' if meta else ''}."
        if st.diets or st.exclude:
            reply += (" It passed your checks: " + ", ".join(c["label"] for c in result.checks if c["status"] == "pass")
                      + "." if not result.flagged else " Note: it has a ⚠ flag — see the card.")
        return {
            "mode": "detail",
            "state": s.to_dict(),
            "reply": reply,
            "results": [cd],
            "detail": {"ingredients": [x["line"] for x in r["ingredients"]] if want_ing or want_steps else [],
                       "steps": steps[:30] if want_steps else [], "url": r["url"]},
            "closest": None,
            "blocked_examples": [],
            "constraints": chips(onto, st),
            "trace": {},
        }

    @cached_property
    def _food_vocab(self) -> frozenset:
        """Every word that may signal food: recipe-name words and ontology aliases."""
        words = set().union(*self.engine.name_tokens)
        for ing in self.engine.onto.ingredients.values():
            for a in ing.aliases:
                words.update(tokenize(a))
        return frozenset(w for w in words if len(w) > 2 and w not in FOOD_STOP)

    @cached_property
    def _strong_food_vocab(self) -> frozenset:
        """Unambiguous food words: ontology aliases, and dish words used in at least 5 recipe names.
        A word from one odd recipe name ("umm" from Umm Ali) is only a weak signal."""
        df = Counter(w for toks in self.engine.name_tokens for w in toks)
        words = {w for w, n in df.items() if n >= 5}
        for ing in self.engine.onto.ingredients.values():
            for a in ing.aliases:
                words.update(tokenize(a))
        return frozenset(w for w in words if len(w) > 2 and w not in FOOD_STOP)

    def _food_signal(self, c: Constraints, ignore_intents: bool = False) -> str:
        """"strong" (clearly a food request), "weak" (only a rare word matched) or "none"."""
        if (c.diets or c.exclude or c.include or c.max_time or c.course or c.cuisine or c.attributes or c.servings
                or c.prefer_non_veg or c.conflicts or c.generic or c.unsupported or (c.intents and not ignore_intents)):
            return "strong"
        toks = set(tokenize(c.text))
        # Spelling variants count too ("khir" -> kheer, "chikan" -> chicken), via the search speller.
        toks |= {t for w in toks for t in tokenize(self.engine.speller.correct(w))}
        if toks & self._strong_food_vocab:
            return "strong"
        return "weak" if toks & self._food_vocab else "none"

    def _has_food_content(self, c: Constraints, ignore_intents: bool = False) -> bool:
        return self._food_signal(c, ignore_intents) != "none"

    def _small_talk_reply(self, s: Session, kind: str) -> dict:
        onto = self.engine.onto
        active = describe(onto, s.state) if (s.state.diets or s.state.exclude) else ""
        keeping = f" I'm still keeping {active} in mind for this chat." if active else ""
        replies = {
            "greeting": "Hi! I find recipes that genuinely fit how you eat — every ingredient is checked, not just "
                        "the recipe name. Tell me your diet or allergies, what's in your fridge, or what you're "
                        "craving." + keeping,
            "thanks": "You're welcome! Want something else — another meal, a weekly plan, or a different cuisine?"
                      + keeping,
            "ack": "Got it. What would you like next — another dish, a different cuisine, or a weekly plan?" + keeping,
            "about": "I'm RecipeForge, a recipe assistant built to show ingredient-grounded recommendations: diets and "
                     "allergies are checked in code against every recipe's real ingredients, and a language model "
                     "only helps understand what you ask — it never decides what's safe for you.",
            "override": "I can't switch off diet or allergy checks — they run in code against every recipe's "
                        "ingredients, not as instructions I can be talked out of. Tell me what you'd like to eat and "
                        "I'll find something that fits." + keeping,
            "reset": "Fresh start — I've cleared this chat's diets and preferences. What would you like to cook?",
            "bye": "Happy cooking! Your chats and cooking history stay saved on this device.",
            "help": "I recommend recipes from 6,000+ Indian and world dishes, checking every ingredient against "
                    "your diet (Jain, sattvic, no onion-garlic, vegan, lactose-free, gluten-free) and allergies. "
                    "I can also plan a week of meals, suggest swaps, and learn from dishes you mark as cooked."
                    + keeping,
            "unclear": "I'm a recipe assistant, so I couldn't find a food request in that. Try telling me a diet, "
                       "an ingredient you have, a cuisine, or how much time you've got." + keeping,
        }
        suggestions = ["no onion no garlic vegetarian", "I have paneer and peas", "quick South Indian breakfast",
                       "plan my week, Jain dinners"]
        return {"mode": "chat", "state": s.to_dict(), "reply": replies[kind], "results": [], "closest": None,
                "blocked_examples": [], "constraints": chips(onto, s.state), "parsed": {}, "trace": {},
                "suggestions": [] if kind == "bye" else suggestions}

    def _base_response(self, s: Session, parsed: Constraints, reply: str, results: list[dict], **extra) -> dict:
        return {"state": s.to_dict(), "reply": reply, "results": results, "closest": None, "blocked_examples": [],
                "constraints": chips(self.engine.onto, s.state), "parsed": parsed.to_dict(), "trace": {}, **extra}

    def _history_reply(self, s: Session, parsed: Constraints, profile) -> dict:
        onto, eng = self.engine.onto, self.engine
        cooked = [e for e in reversed(profile.events) if e.action == "cooked"]
        recent = [e for e in cooked if e.id in profile.recent_ids]
        if not cooked:
            reply = ("I don't have any cooking history for you yet. Tap “Cooked it” on a recipe card after you make it, "
                     "and I'll remember it here — on this device only.")
            return self._base_response(s, parsed, reply, [], mode="history")
        shown = recent or cooked
        seen, cards = set(), []
        for e in shown:
            if e.id in seen:
                continue
            seen.add(e.id)
            r = eng.recipes[eng.by_id[e.id]]
            cards.append({**card_from_recipe(r), "why": [f"cooked {_ago(e.t)}"]})
        when = "in the last 7 days" if recent else "so far (nothing in the last 7 days)"
        reply = f"You've cooked {len(seen)} dish{'es' if len(seen) != 1 else ''} {when}. Overall: {profile.summary(onto)}."
        return self._base_response(s, parsed, reply, cards[:10], mode="history")

    def _plan_reply(self, s: Session, parsed: Constraints, profile) -> dict:
        onto, st = self.engine.onto, s.state
        days = parsed.days or 7
        plan = make_plan(self.engine, st, profile, days)
        what = describe(onto, replace(st, course=None))
        course = st.course or "dinner"
        if not plan.days:
            reply = f"I couldn't build a {course} plan for {what} — nothing in the catalogue meets every constraint."
            return self._base_response(s, parsed, reply, [], mode="plan")
        cards = [{**card(d.result), "day": d.day, "base": onto.label(d.base) if d.base in onto.ingredients else d.base}
                 for d in plan.days]
        s.shown += [d.result.recipe["id"] for d in plan.days]
        s.last_ids = [d.result.recipe["id"] for d in plan.days]
        reply = (f"Here's a {len(plan.days)}-day {what if what != 'recipe' else ''} {course} plan. Every dish is verified "
                 f"against your constraints, no main ingredient repeats on consecutive days, cuisines rotate, and at most "
                 f"two rich dishes. The catalogue has no calorie data, so balance here means variety rather than "
                 f"nutrient targets.").replace("  ", " ")
        if plan.shortfall:
            reply += f" I could only fill {len(plan.days)} of {days} days without repeating myself under these constraints."
        if profile is not None and not profile.empty:
            reply += " Dishes you cooked this week were skipped and the plan leans toward what you've liked."
        return self._base_response(s, parsed, reply, cards, mode="plan",
                                   shopping={"items": plan.shopping, "staples": plan.staples})

    # ---------- grounded, template-based reply ----------
    def compose(self, st: Constraints, parsed: Constraints, out: SearchOutput, anchor: int | None) -> str:
        onto = self.engine.onto
        diet_labels = [onto.diets[d]["label"] for d in sorted(st.diets)]
        lines = []
        if parsed.unsupported:
            what = " / ".join(parsed.unsupported)
            lines.append(f"I can't verify “{what}” — this catalogue has no carb, sugar or sodium data — so treat these "
                         f"as ideas rather than a guarantee.")
        swaps = [x for x in st.conflicts if x in st.swaps and x not in STYLE_ONLY]
        if swaps:
            pairs = ", ".join(f"{onto.label(x)} → {onto.label(st.swaps[x])}" for x in swaps)
            lines.append(f"That doesn't fit {', '.join(diet_labels) or 'your diet'}, so I swapped {pairs}.")
        for x in st.conflicts:
            if x not in st.swaps:
                lines.append(f"{onto.label(x).capitalize()} doesn't fit {', '.join(diet_labels) or 'your diet'}, "
                             f"so here are the closest alternatives.")
        if anchor is not None and anchor in self.engine.by_id:
            lines.append(f"Similar to {self.engine.recipes[self.engine.by_id[anchor]]['name']}"
                         + (" but spicier" if "spicy" in st.attributes else "") + ":")
        if not out.results:
            what = describe(onto, st)
            if st.text and (what == "recipe" or not (st.diets or st.exclude)):
                what = f"“{st.text}”" if what == "recipe" else f"{what} “{st.text}”"
            if out.closest:
                cr = out.closest.recipe
                lines.append(f"Nothing in the catalogue matches {what} within {st.max_time} minutes. "
                             f"The quickest option that still meets every other constraint is {cr['name']} "
                             f"({cr['total_min']} min). I won't relax a dietary or allergy rule to fill the gap.")
            else:
                lines.append(f"I couldn't find anything that meets {what}. I won't relax a dietary or allergy rule "
                             f"to fill the gap — try removing a preference like cuisine or time instead.")
            fix = next((b for b in out.blocked_examples if b.get("adaptation", {}).get("adaptable")), None)
            if fix:
                lines.append(f"{fix['name']} can be adapted with {len(fix['adaptation']['swaps'])} swap(s) — see below.")
            return " ".join(lines)
        n = len(out.results)
        dish = {t for t in tokenize(st.text) if len(t) > 3} - {t for i in st.include for t in tokenize(onto.label(i))}
        names = {t for r in out.results for t in tokenize(r.recipe["name"])}
        missing_dish = bool(dish and (st.diets or st.exclude) and not st.swaps and not dish & names)
        if missing_dish:
            adaptable = [b for b in out.blocked_examples if b.get("adaptation", {}).get("adaptable")
                         and not self.engine.is_condiment[self.engine.by_id[b["id"]]]]
            adaptable.sort(key=lambda b: not dish & set(tokenize(b["name"])))
            same_dish = adaptable and dish & set(tokenize(adaptable[0]["name"]))
            lines.append(f"I couldn't find a {describe(onto, st)} “{st.text}” in the catalogue"
                         + (f" — but {adaptable[0]['name']} can be made {describe(onto, st)} with a few swaps "
                            f"(see below)." if same_dish else "."))
        head = (f"Meanwhile, here {'are' if n > 1 else 'is'} {n} {describe(onto, st)} pick{'s' if n > 1 else ''} "
                f"that do fit" if missing_dish else
                f"Here {'are' if n > 1 else 'is'} {n} {describe(onto, st)} pick{'s' if n > 1 else ''}")
        if st.include:
            head += " using " + ", ".join(onto.label(i) for i in st.include)
        if "spicy" in st.attributes and anchor is None:
            head += ", leaning spicy"
        if st.diets or st.exclude:
            # The counts and ruled-out examples live in "How I found these"; the reply stays short.
            head += ", each checked ingredient by ingredient"
        lines.append(head + ".")
        if st.pantry_mode and st.include:
            want = set(st.include)
            best = max(len(want & self.engine.lineages[self.engine.by_id[r.recipe["id"]]]) for r in out.results)
            if best < len(want):
                names = ", ".join(onto.label(i) for i in st.include)
                lines.append(f"No option{' within ' + str(st.max_time) + ' min' if st.max_time else ''} uses all of "
                             f"{names}; these use the most of them.")
        if any(r.flagged for r in out.results):
            lines.append("Items marked ⚠ depend on brand or preparation — check the note before cooking.")
        if st.servings:
            lines.append(f"Quantities are per the original recipe; I've noted the scaling factor for {st.servings} people.")
        return " ".join(lines)

    def polish(self, reply: str, out: SearchOutput) -> str | None:
        """Optional LLM rewrite of the intro sentence. Recipe facts and cards never come from the LLM."""
        base, key, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
        if not (base and key and model and out.results):
            return None
        allowed = {r.recipe["name"].lower() for r in out.results}
        try:
            resp = httpx.post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "temperature": 0.3, "max_tokens": 160, "messages": [
                    {"role": "system", "content": "Rewrite the assistant message to sound warm and concise. "
                     "Keep every fact and number. Do not add recipes, ingredients or health claims."},
                    {"role": "user", "content": reply}]},
                timeout=8,
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
        mentioned = {r["name"].lower() for r in self.engine.recipes if r["name"].lower() in text.lower()}
        if mentioned - allowed - {b["name"].lower() for b in out.blocked_examples}:
            return None
        if not all(num in text for num in re.findall(r"\d[\d,]*", reply)):
            return None
        return text


_SMALL_TALK = [
    ("greeting", r"(hi+|hello+|hey+|hiya|helo|namaste|namaskar|hola|yo|sup|good\s+(morning|afternoon|evening|day))"
                 r"(\s+(there|nosh|bot|recipeforge|ji|friend))?"),
    ("thanks", r"(no,?\s+)?(thanks|thank\s*you|thx|ty|dhanyavad|shukriya)(\s+(so much|a lot|you))?"),
    ("ack", r"(ok(ay)?|cool|great|nice|awesome|perfect|got it|alright|fine|sure|yes|yeah|yup|hmm+|accha|achha|theek hai)"),
    ("bye", r"(bye|goodbye|see you|good\s*night|tata)"),
    ("about", r"(who (made|built|created) (you|this)|who are you|what are you|what is this( app| site)?|"
              r"are you (chatgpt|claude|an ai|a bot))"),
    ("help", r"(help|"
             r"what (else )?(can|do|could|will|should|does) (you|u|it|this( app| site)?|i) (d+o+|offer|help with|ask)"
             r"( for me)?|"
             r"how (do|does|can|should) (you|u|this|it|i) (work|help|use (this|it|you))|how to use (this|it|you)|"
             r"what (are|is) (you|u|this|it) (for|good at|capable of|able to do)|what('s| is) your (job|purpose|use)|"
             r"(your )?(features|capabilities)|"
             r"(tum|aap|ap) kya (kar sakte|karte) (ho|hain|hai)|kya kar sakte ho|kya karte ho)"),
]
# Swaps that only change a cooking fat (butter -> oil) aren't worth announcing.
STYLE_ONLY = {"butter", "ghee"}
OVERRIDE = re.compile(r"(ignore (all |any |the )?(previous|prior|above) (instructions|rules)|^\s*system\s*:|"
                      r"(disable|turn off|remove|bypass|skip) (all |the )?(diet|allergy|safety)? ?(filters?|checks?|rules?)|"
                      r"jailbreak|developer mode|pretend you)", re.I)


def small_talk_kind(message: str) -> str | None:
    if OVERRIDE.search(message):
        return "override"
    text = re.sub(r"[\s!.?,:)(]+$", "", message.strip().lower())
    for kind, pat in _SMALL_TALK:
        # Inside a longer message a help phrase must end it: "hi, what can you do" is help,
        # "what do you do with okra" is a food question.
        if re.fullmatch(rf"{pat}(\s*[!.?,]+\s*{pat})*", text) or (kind == "help" and re.search(rf"\b{pat}$", text)
                                                                     and len(text.split()) <= 6):
            return kind
    return None


def describe(onto, st: Constraints) -> str:
    bits = [onto.diets[d]["label"] for d in sorted(st.diets)]
    bits += [f"no-{onto.label(e)}" for e in sorted(st.exclude)
             if not any(e in ("onion", "garlic") and d == "no_onion_garlic" for d in st.diets)]
    if st.cuisine:
        bits.append(st.cuisine.title())
    if st.course:
        bits.append(st.course)
    return " ".join(bits) if bits else "recipe"


def chips(onto, st: Constraints) -> list[dict]:
    out = [{"kind": "diet", "value": d, "label": onto.diets[d]["label"], "hard": True} for d in sorted(st.diets)]
    out += [{"kind": "exclude", "value": e, "label": f"no {onto.label(e)}", "hard": True} for e in sorted(st.exclude)]
    if st.max_time:
        out.append({"kind": "max_time", "value": str(st.max_time), "label": f"≤ {st.max_time} min", "hard": True})
    out += [{"kind": "include", "value": i, "label": f"with {onto.label(i)}", "hard": False} for i in st.include]
    for kind in ("course", "cuisine"):
        v = getattr(st, kind)
        if v:
            out.append({"kind": kind, "value": v, "label": v, "hard": False})
    out += [{"kind": "attribute", "value": a, "label": a.replace("_", " "), "hard": False} for a in sorted(st.attributes)]
    if st.servings:
        out.append({"kind": "servings", "value": str(st.servings), "label": f"{st.servings} people", "hard": False})
    return out


def card_from_recipe(rec: dict) -> dict:
    return {
        "id": rec["id"], "name": rec["name"], "cuisine": rec["cuisine"], "course": rec["course"],
        "total_min": rec["total_min"], "servings": rec["servings"], "url": rec["url"],
        "ingredients": [i["name"] for i in rec["ingredients"]][:14],
        "checks": [], "why": [], "flagged": False, "score": 0,
    }


def card(r: Result | None) -> dict | None:
    if r is None:
        return None
    return {**card_from_recipe(r.recipe), "checks": r.checks, "why": r.why, "flagged": r.flagged, "score": r.score}


def _ago(t: int) -> str:
    days = int((time.time() - t) // 86400)
    return "today" if days <= 0 else "yesterday" if days == 1 else f"{days} days ago"
