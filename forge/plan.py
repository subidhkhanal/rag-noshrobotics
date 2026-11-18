"""Multi-day meal planner: hard-filtered candidates, greedy selection under variety constraints."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .profile import STAPLES
from .query import Constraints
from .search import MEAT_TAGS

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# The dish's "base": first match wins. Used to avoid paneer three nights running.
BASES = ["chicken", "mutton", "fish", "shellfish", "egg", "paneer", "tofu", "soy_chunks", "chickpeas", "kidney_beans",
         "dal", "mushroom", "potato", "cauliflower", "okra", "brinjal", "gourd", "spinach", "greens", "methi_leaves",
         "cabbage", "beans", "peas", "pumpkin", "jackfruit", "raw_banana", "millet", "pasta", "noodles", "rice"]
MAX_BASE, MAX_CUISINE, MAX_RICH, MAX_SIM = 2, 3, 2, 0.85
# Broad ontology classes: on a shopping list, show the recipe's own wording ("raw mango", "masoor dal") instead.
GENERIC_IDS = {"fruit", "greens", "herbs", "gourd", "dal", "beans", "mixed_vegetables", "spice_blend", "millet",
               "pseudo_grain", "kidney_beans", "chickpeas", "seeds", "rice", "pasta", "noodles", "bread", "cheese",
               "oil", "chutney_green", "chutney_sweet", "syrup", "essence", "aged_cheese", "fish", "shellfish"}


def _shopping_name(display: str) -> str:
    name = display.split("(")[0].strip().lower()
    return name[:40] or display.lower()


@dataclass
class PlanDay:
    day: str
    result: object
    base: str


@dataclass
class Plan:
    days: list[PlanDay] = field(default_factory=list)
    shopping: list[dict] = field(default_factory=list)
    staples: list[str] = field(default_factory=list)
    shortfall: int = 0
    pool: int = 0


def base_of(engine, i: int) -> str:
    lineage = engine.lineages[i]
    return next((b for b in BASES if b in lineage), engine.recipes[i]["cuisine"] or "other")


def make_plan(engine, c: Constraints, profile=None, days: int = 7) -> Plan:
    course = c.course or "dinner"
    keep, _, _ = engine.hard_filter(c)
    excluded = set(profile.disliked_ids | profile.recent_ids) if profile is not None else set()

    def meal_ok(i: int, courses: set[str]) -> bool:
        r = engine.recipes[i]
        return (any(engine._course_ok(i, c) for c in courses) and not r.get("is_component") and not engine.is_baby[i]
                and not engine.is_condiment[i] and r["id"] not in excluded)

    cand = [i for i in keep if meal_ok(i, {course})]
    if len(cand) < days and course in ("dinner", "lunch"):
        cand = [i for i in keep if meal_ok(i, {"dinner", "lunch"})]
    if c.cuisine:
        narrowed = [i for i in cand if engine._cuisine_ok(i, c.cuisine)]
        if len(narrowed) >= days:
            cand = narrowed
    if c.prefer_non_veg:
        narrowed = [i for i in cand if engine.tags[i] & MEAT_TAGS]
        if len(narrowed) >= days:
            cand = narrowed
    plan = Plan(pool=len(cand))
    if not cand:
        plan.shortfall = days
        return plan

    qtext = engine.query_text(c)
    qvec = engine.embed_query(qtext) if engine.emb is not None else None
    scored = engine.score_pool(c, cand, qtext, qvec, None, {}, pool_size=min(250, len(cand)), profile=profile)

    chosen: list[tuple] = []
    bases, cuisines = Counter(), Counter()
    rich, names = 0, set()
    for _ in range(days):
        pick = None
        for item in scored:
            s, i, flags, cover = item
            if any(i == x[1] for x in chosen):
                continue
            base, cuisine = base_of(engine, i), engine.recipes[i]["cuisine"]
            is_rich = engine.features(i)["creamy"] >= 0.75
            if (bases[base] >= MAX_BASE or cuisines[cuisine] >= MAX_CUISINE or (is_rich and rich >= MAX_RICH)
                    or engine._base_name(i) in names
                    or (chosen and base_of(engine, chosen[-1][1]) == base)):
                continue
            if engine.emb is not None and any(float(engine.emb[i] @ engine.emb[x[1]]) > MAX_SIM for x in chosen):
                continue
            pick = (item, base, cuisine, is_rich)
            break
        if pick is None:
            break
        (s, i, flags, cover), base, cuisine, is_rich = pick
        chosen.append((s, i, flags, cover))
        bases[base] += 1
        cuisines[cuisine] += 1
        rich += is_rich
        names.add(engine._base_name(i))
        plan.days.append(PlanDay(DAY_NAMES[len(plan.days) % 7] if days <= 7 else f"Day {len(plan.days) + 1}",
                                 engine._explain(i, s, flags, c, profile), base))
    plan.shortfall = days - len(plan.days)

    need: dict[str, list[str]] = {}
    staples: set[str] = set()
    for d in plan.days:
        r = d.result.recipe
        for it in r["ingredients"]:
            specific = [i for i in it["ids"] if i not in GENERIC_IDS]
            for iid in (specific or it["ids"]):
                if iid in STAPLES:
                    staples.add(engine.onto.label(iid))
                elif "non_food" not in engine.onto.ingredients[iid].tags:
                    label = _shopping_name(it["name"]) if iid in GENERIC_IDS else engine.onto.label(iid)
                    if r["name"] not in need.get(label, []):
                        need.setdefault(label, []).append(r["name"])
    plan.shopping = [{"item": k, "for": v} for k, v in sorted(need.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    plan.staples = sorted(staples)
    return plan
