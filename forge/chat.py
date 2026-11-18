"""Multi-turn dialogue state and grounded response generation."""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field

import httpx

from .query import Constraints, is_refinement, parse
from .search import Engine, Result, SearchOutput

MAX_SESSIONS = 500


@dataclass
class Session:
    id: str
    state: Constraints = field(default_factory=Constraints)
    shown: list[int] = field(default_factory=list)
    last_top: int | None = None
    turns: int = 0


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
    if is_refinement(message, new):
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
        self.sessions: dict[str, Session] = {}

    def session(self, sid: str | None) -> Session:
        if sid and sid in self.sessions:
            return self.sessions[sid]
        if len(self.sessions) >= MAX_SESSIONS:
            self.sessions.pop(next(iter(self.sessions)))
        s = Session(id=uuid.uuid4().hex)
        self.sessions[s.id] = s
        return s

    def remove_constraint(self, sid: str, kind: str, value: str) -> Session:
        s = self.session(sid)
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

    def chat(self, sid: str | None, message: str) -> dict:
        s = self.session(sid)
        onto = self.engine.onto
        parsed = parse(onto, message)
        s.state = merge(s.state, parsed, message)
        st = s.state
        s.turns += 1

        exclude_ids: set[int] = set()
        if st.intents & {"more", "different"}:
            exclude_ids = set(s.shown)
        anchor = s.last_top if "similar" in st.intents else None
        if anchor is not None:
            exclude_ids.add(anchor)

        out = self.engine.search(st, k=5, exclude_ids=exclude_ids, anchor_id=anchor)
        if out.results:
            s.shown += [r.recipe["id"] for r in out.results]
            s.last_top = out.results[0].recipe["id"]
        reply = self.compose(st, parsed, out, anchor)
        reply = self.polish(reply, out) or reply
        return {
            "session_id": s.id,
            "reply": reply,
            "results": [card(r) for r in out.results],
            "closest": card(out.closest) if out.closest else None,
            "blocked_examples": out.blocked_examples,
            "constraints": chips(onto, st),
            "parsed": parsed.to_dict(),
            "trace": out.trace | {"relaxed": out.relaxed},
        }

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


def card(r: Result | None) -> dict | None:
    if r is None:
        return None
    rec = r.recipe
    return {
        "id": rec["id"], "name": rec["name"], "cuisine": rec["cuisine"], "course": rec["course"],
        "total_min": rec["total_min"], "servings": rec["servings"], "url": rec["url"],
        "ingredients": [i["name"] for i in rec["ingredients"]][:14],
        "checks": r.checks, "why": r.why, "flagged": r.flagged, "score": r.score,
    }
