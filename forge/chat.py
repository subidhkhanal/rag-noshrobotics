"""Multi-turn dialogue state and grounded response generation."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field, replace

import httpx

from . import llm_parser
from .adapt import adapt
from .plan import make_plan
from .profile import build_profile
from .query import ATTRIBUTE_RES, COURSE_RES, CUISINE_GROUPS, Constraints, is_refinement, parse
from .search import Engine, Result, SearchOutput

MAX_SHOWN = 60


@dataclass
class Session:
    """Conversation state. The server is stateless: the client sends this back with every request."""
    state: Constraints = field(default_factory=Constraints)
    shown: list[int] = field(default_factory=list)
    last_top: int | None = None

    def to_dict(self) -> dict:
        st = self.state
        return {
            "diets": sorted(st.diets), "exclude": sorted(st.exclude), "include": list(st.include),
            "pantry_mode": st.pantry_mode, "max_time": st.max_time, "course": st.course, "cuisine": st.cuisine,
            "attributes": sorted(st.attributes), "servings": st.servings, "prefer_non_veg": st.prefer_non_veg,
            "text": st.text, "shown": self.shown[-MAX_SHOWN:], "last_top": self.last_top,
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
    return Session(state=st, shown=shown, last_top=last if isinstance(last, int) and last in engine.by_id else None)


def merge(prev: Constraints, new: Constraints, message: str) -> Constraints:
    if "reset" in new.intents:
        new.intents.discard("reset")
        return new
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
    merged.conflicts, merged.swaps = new.conflicts, new.swaps
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
        profile = build_profile(self.engine, history)
        parsed = parse(onto, message)
        if llm_parser.enabled():
            parsed = llm_parser.combine(onto, parsed, llm_parser.llm_parse(message))
        s.state = merge(s.state, parsed, message)
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
            "trace": out.trace | {"relaxed": out.relaxed},
        }

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
        for x in parsed.conflicts:
            alt = parsed.swaps.get(x)
            lines.append(f"{onto.label(x).capitalize()} doesn't fit {', '.join(diet_labels) or 'your diet'}"
                         + (f", so I swapped it for {onto.label(alt)}." if alt else ", so here are the closest alternatives that do."))
        if anchor is not None and anchor in self.engine.by_id:
            lines.append(f"Similar to {self.engine.recipes[self.engine.by_id[anchor]]['name']}"
                         + (" but spicier" if "spicy" in st.attributes else "") + ":")
        if not out.results:
            what = describe(onto, st)
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
        head = f"Here {'are' if n > 1 else 'is'} {n} {describe(onto, st)} pick{'s' if n > 1 else ''}"
        if st.include:
            head += " using " + ", ".join(onto.label(i) for i in st.include)
        if "spicy" in st.attributes and anchor is None:
            head += ", leaning spicy"
        if st.diets or st.exclude:
            removed = sum(out.trace["removed_by_constraint"].values())
            head += f", each verified ingredient by ingredient. I ruled out {removed:,} of {out.trace['corpus']:,} recipes that broke your constraints"
            if out.blocked_examples:
                ex = out.blocked_examples[0]
                head += f" (for example {ex['name']} — {ex['reason'].split(': ', 1)[-1]})"
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
