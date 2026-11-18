import time

import pytest

from forge.chat import Assistant
from forge.plan import base_of, make_plan
from forge.profile import build_profile
from forge.query import parse
from forge.search import Engine


@pytest.fixture(scope="module")
def engine():
    return Engine(profile="fast")


def rid(engine, prefix):
    return next(r["id"] for r in engine.recipes if r["name"].startswith(prefix))


def history(engine):
    now = int(time.time())
    return [{"id": rid(engine, "Kadai Paneer"), "action": "cooked", "t": now - 2 * 86400},
            {"id": rid(engine, "Dal Makhani"), "action": "cooked", "t": now - 30 * 86400},
            {"id": rid(engine, "Matar Paneer"), "action": "like", "t": now - 5 * 86400}]


def test_profile_drops_invalid_events(engine):
    bad = [{"id": -1, "action": "cooked", "t": 1}, {"id": "x"}, "junk",
           {"id": rid(engine, "Kadai Paneer"), "action": "hack", "t": 5}]
    assert build_profile(engine, bad).empty
    p = build_profile(engine, history(engine))
    assert not p.empty and len(p.cooked_ids) == 2 and len(p.recent_ids) == 1


def test_history_reply_uses_real_history_or_says_none(engine):
    a = Assistant(engine)
    assert a.chat(None, "what did I cook last week", history(engine))["results"][0]["name"].startswith("Kadai Paneer")
    empty = a.chat(None, "what did I cook last week", None)
    assert empty["results"] == [] and "don't have any cooking history" in empty["reply"]


def test_something_different_avoids_cooked_dishes(engine):
    a = Assistant(engine)
    h = history(engine)
    out = a.chat(None, "suggest something different from what I usually make", h)
    cooked = {e["id"] for e in h if e["action"] == "cooked"}
    assert out["results"] and not cooked & {r["id"] for r in out["results"]}
    assert out["reply"].startswith("Based on your history")
    assert "don't have your cooking history" in a.chat(None, "suggest something different from what I usually make")["reply"]


def test_plan_intent_and_days():
    from forge.ontology import load
    onto = load()
    assert parse(onto, "5 day vegan lunch plan").days == 5
    assert parse(onto, "plan my week, jain dinners").days == 7


def test_jain_week_plan_is_compliant_varied_and_has_no_baby_food(engine):
    c = parse(engine.onto, "plan my week, jain dinners")
    plan = make_plan(engine, c, None, 7)
    assert len(plan.days) == 7
    ids = [engine.by_id[d.result.recipe["id"]] for d in plan.days]
    assert all(engine.recipes[i]["verdicts"]["jain"]["status"] != "block" for i in ids)
    bases = [base_of(engine, i) for i in ids]
    assert all(a != b for a, b in zip(bases, bases[1:]))
    assert not any(engine.is_baby[i] or engine.is_condiment[i] for i in ids)
    assert plan.shopping


def test_usual_request_drops_previous_soft_prefs_but_keeps_diet(engine):
    a = Assistant(engine)
    first = a.chat(None, "jain paneer curry", history(engine))
    out = a.chat(first["state"], "suggest something different from what I usually make", history(engine))
    assert out["state"]["include"] == [] and out["state"]["diets"] == ["jain"]
    assert sum("paneer" in r["name"].lower() for r in out["results"]) <= 1
