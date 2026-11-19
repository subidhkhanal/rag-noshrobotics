"""Distilled command classifier: understands most conversational turns in ~1 ms without calling an LLM.

Trained offline (scripts/train_intent.py) on Claude-written, labelled messages (scripts/distill_data.py), the
standard "LLM as teacher, small model as student" pattern. Features are hashed character and word n-grams
(robust to Hinglish spelling and Devanagari) plus what's on screen; the model is multinomial logistic
regression, so inference is a sparse dot product in numpy. Which card a message means ("Friday", "the 2nd",
"it", "the palak one") is resolved by the deterministic resolver in forge/chat.py, not by the model.

When the model isn't confident, the turn goes to Claude as before.
"""
from __future__ import annotations

import re
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "data" / "models" / "intent.npz"
DIM = 1 << 17
_WORD = re.compile(r"[\wऀ-ॿ]+")


def features(text: str, screen: str) -> dict[int, float]:
    """Hashed n-gram counts. `screen`: "plan" | "list" | "focus" | "none" (what's on screen right now)."""
    t = " ".join(text.lower().split())
    feats: dict[int, float] = {}

    def add(key: str, w: float = 1.0) -> None:
        h = zlib.crc32(key.encode("utf-8")) % DIM
        feats[h] = feats.get(h, 0.0) + w

    padded = f" {t} "
    for n in (2, 3, 4):
        for i in range(len(padded) - n + 1):
            add("c" + padded[i:i + n])
    words = _WORD.findall(t)
    for i, w in enumerate(words):
        add("w" + w, 2.0)
        if i:
            add("b" + words[i - 1] + " " + w, 2.0)
    add("screen=" + screen, 3.0)
    for w in words[:3]:  # the opening words carry the intent ("how", "is", "which", "hi")
        add("first=" + w, 1.5)
    if t.endswith("?"):
        add("question_mark", 1.5)
    norm = sum(v * v for v in feats.values()) ** 0.5 or 1.0
    return {k: v / norm for k, v in feats.items()}


@lru_cache(maxsize=1)
def _load():
    if not MODEL.exists():
        return None
    z = np.load(MODEL, allow_pickle=False)
    heads = {}
    for head in ("action", "want", "ref"):
        heads[head] = (z[f"{head}_coef"], z[f"{head}_bias"], [str(x) for x in z[f"{head}_labels"]])
    return heads


def available() -> bool:
    return _load() is not None


def _predict(head, feats) -> tuple[str, float]:
    coef, bias, labels = head
    idx = np.fromiter(feats.keys(), dtype=np.int64)
    val = np.fromiter(feats.values(), dtype=np.float32)
    logits = coef[:, idx] @ val + bias
    p = np.exp(logits - logits.max())
    p /= p.sum()
    k = int(p.argmax())
    return labels[k], float(p[k])


def predict(text: str, screen: str) -> dict | None:
    """{"action", "confidence", "want", "ref"} or None if no model is installed."""
    heads = _load()
    if heads is None:
        return None
    f = features(text, screen)
    action, conf = _predict(heads["action"], f)
    want, _ = _predict(heads["want"], f)
    ref, ref_conf = _predict(heads["ref"], f)
    return {"action": action, "confidence": conf, "want": want, "ref": ref, "ref_confidence": ref_conf}
