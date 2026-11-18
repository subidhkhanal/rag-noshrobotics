"""Taste profile built from the user's own cooking history (kept client-side, sent with each request)."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

ACTIONS = {"cooked": 0.7, "like": 1.0, "dislike": -1.0}
MAX_EVENTS = 100
RECENT_DAYS = 7
STAPLES = {
    "salt", "water", "oil", "turmeric_powder", "red_chilli", "cumin", "mustard_seed", "sugar", "green_chilli",
    "coriander_leaves", "curry_leaves", "asafoetida", "garam_masala", "coriander_seed", "black_pepper", "ghee",
    "lemon", "chilli", "leavening", "herbs", "fenugreek_seed", "cardamom", "cinnamon", "cloves", "bay_leaf",
}


@dataclass
class Event:
    id: int
    action: str
    t: int


@dataclass
class Profile:
    events: list[Event] = field(default_factory=list)
    liked_vec: np.ndarray | None = None
    disliked_vec: np.ndarray | None = None
    cooked_ids: set[int] = field(default_factory=set)
    recent_ids: set[int] = field(default_factory=set)
    disliked_ids: set[int] = field(default_factory=set)
    cuisines: Counter = field(default_factory=Counter)
    ingredients: Counter = field(default_factory=Counter)

    @property
    def empty(self) -> bool:
        return not any(e.action in ("cooked", "like") for e in self.events)

    def summary(self, onto) -> str:
        n = len(self.cooked_ids)
        bits = []
        if self.cuisines:
            bits.append("mostly " + " and ".join(c for c, _ in self.cuisines.most_common(2)))
        if self.ingredients:
            bits.append("often with " + ", ".join(onto.label(i) for i, _ in self.ingredients.most_common(3)))
        return f"{n} dish{'es' if n != 1 else ''} cooked" + (f", {'; '.join(bits)}" if bits else "")


def build_profile(engine, raw) -> Profile:
    """Validate client-supplied history and summarise it. Unknown recipes/actions are dropped."""
    if not isinstance(raw, list):
        return Profile()
    now = int(time.time())
    events = []
    for e in raw[-MAX_EVENTS:]:
        if not isinstance(e, dict):
            continue
        rid, action, t = e.get("id"), e.get("action"), e.get("t")
        if isinstance(rid, int) and rid in engine.by_id and action in ACTIONS and isinstance(t, int) and 0 < t <= now + 86400:
            events.append(Event(rid, action, t))
    p = Profile(events=events)
    pos_w: dict[int, float] = {}
    neg: set[int] = set()
    for e in events:
        if e.action == "dislike":
            neg.add(e.id)
            pos_w.pop(e.id, None)
        else:
            pos_w[e.id] = pos_w.get(e.id, 0) + ACTIONS[e.action]
            neg.discard(e.id)
        if e.action == "cooked":
            p.cooked_ids.add(e.id)
            if now - e.t <= RECENT_DAYS * 86400:
                p.recent_ids.add(e.id)
    p.disliked_ids = neg
    for rid in pos_w:
        r = engine.recipes[engine.by_id[rid]]
        p.cuisines[r["cuisine"]] += 1
        p.ingredients.update(i for i in r["ingredient_ids"] if i not in STAPLES)
    if engine.emb is not None:
        if pos_w:
            ids = [engine.by_id[r] for r in pos_w]
            v = (engine.emb[ids] * np.array(list(pos_w.values()), dtype=np.float32)[:, None]).sum(0)
            p.liked_vec = v / (np.linalg.norm(v) or 1.0)
        if neg:
            v = engine.emb[[engine.by_id[r] for r in neg]].mean(0)
            p.disliked_vec = v / (np.linalg.norm(v) or 1.0)
    return p
