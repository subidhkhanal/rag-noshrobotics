"""Train the distilled command classifier on data/distill/train.jsonl and save data/models/intent.npz.

Splits off 15% for validation (stratified), reports accuracy and the confidence/accuracy trade-off used to pick
the routing threshold. The held-out gold set (data/eval/dialogue.yaml) is never used here.

    python scripts/train_intent.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from forge.intent_model import DIM, MODEL, features  # noqa: E402

SCREENS = {"plan": ["plan"], "list": ["list"], "focus": ["focus"], "none": ["none"], "any": ["none", "list", "plan"]}


def expand(rows):
    """One example per plausible screen state (social/plan messages occur with or without cards)."""
    out = []
    for r in rows:
        for screen in SCREENS[r["screen"]]:
            out.append({**r, "screen_state": screen})
    return out


def matrix(rows):
    data, ind, ptr = [], [], [0]
    for r in rows:
        f = features(r["text"], r["screen_state"])
        ind += list(f.keys())
        data += list(f.values())
        ptr.append(len(ind))
    return csr_matrix((np.array(data, dtype=np.float32), np.array(ind), np.array(ptr)), shape=(len(rows), DIM))


def fit(X, y):
    m = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    m.fit(X, y)
    return m


def main():
    rows = [json.loads(l) for l in (ROOT / "data" / "distill" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    random.seed(0)
    train, val = train_test_split(rows, test_size=0.15, random_state=0, stratify=[r["action"] for r in rows])
    tr, va = expand(train), expand(val)
    Xtr, Xva = matrix(tr), matrix(va)
    heads = {}
    for head in ("action", "want", "ref"):
        keep_tr = [i for i, r in enumerate(tr) if head == "action" or r["action"] in ("show_recipe", "ask_about")]
        keep_va = [i for i, r in enumerate(va) if head == "action" or r["action"] in ("show_recipe", "ask_about")]
        ytr = [tr[i][head] for i in keep_tr]
        m = fit(Xtr[keep_tr], ytr)
        pred = m.predict(Xva[keep_va])
        acc = float(np.mean(pred == np.array([va[i][head] for i in keep_va])))
        print(f"{head:7} val accuracy {acc:.3f} on {len(keep_va)} ({Counter(ytr)})")
        heads[head] = m
    # Confidence vs accuracy for the routing threshold
    m = heads["action"]
    proba = m.predict_proba(Xva)
    pred = m.classes_[proba.argmax(1)]
    gold = np.array([r["action"] for r in va])
    conf = proba.max(1)
    print("\nthreshold  coverage  accuracy-when-confident")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        sel = conf >= t
        print(f"   {t:.1f}      {sel.mean():.2f}      {np.mean(pred[sel] == gold[sel]):.3f}")
    # Refit on all data and save
    full = expand(rows)
    X = matrix(full)
    save = {}
    for head in ("action", "want", "ref"):
        keep = [i for i, r in enumerate(full) if head == "action" or r["action"] in ("show_recipe", "ask_about")]
        m = fit(X[keep], [full[i][head] for i in keep])
        coef = m.coef_.astype(np.float16)
        if coef.shape[0] == 1:  # binary heads come back as one row
            coef = np.vstack([-coef, coef])
            bias = np.array([-m.intercept_[0], m.intercept_[0]], dtype=np.float32)
        else:
            bias = m.intercept_.astype(np.float32)
        save[f"{head}_coef"] = coef
        save[f"{head}_bias"] = bias
        save[f"{head}_labels"] = np.array(m.classes_, dtype=str)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL, **save)
    print(f"\nsaved {MODEL.relative_to(ROOT)} ({MODEL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
