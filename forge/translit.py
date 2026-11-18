"""Hindi script and spelling-variant handling for search.

1. deva_to_latin: rule-based Devanagari -> Hinglish romanization with Hindi schwa deletion
   ("खिचड़ी" -> "khichdi", "बिरयानी" -> "biryani", "राजमा" -> "rajma").
2. Speller: maps a word the corpus never uses to the closest corpus word by sound
   ("biriyani" -> "biryani", "sambhar" -> "sambar", "khir" -> "kheer", "chatni" -> "chutney").
Both only change search words; diet and allergen checks never depend on them.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_VIRAMA, _NUKTA = "्", "़"
_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ड़": "d", "ढ़": "dh", "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "फ़": "f",
    "ऱ": "r", "ळ": "l",
}
_NUKTA_FORMS = {"ड": "ड़", "ढ": "ढ़", "क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "फ": "फ़"}
_MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
          "ॅ": "e", "ॉ": "o"}
_VOWEL = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
          "औ": "au", "ऑ": "o"}
_LABIAL = set("pbm")
_DEVA_WORD = re.compile(r"[ऀ-ॿ]+")


def _units(word: str) -> list[list]:
    """Split a Devanagari word into units: [kind, latin, has_explicit_vowel, schwa_suppressed]."""
    out: list[list] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if i + 1 < len(chars) and chars[i + 1] == _NUKTA and ch in _NUKTA_FORMS:
            ch = _NUKTA_FORMS[ch]
            i += 1
        if ch in _CONS:
            out.append(["C", _CONS[ch], None, False])
        elif ch in _MATRA and out and out[-1][0] == "C" and out[-1][2] is None:
            out[-1][2] = _MATRA[ch]
        elif ch == _VIRAMA and out and out[-1][0] == "C":
            out[-1][3] = True
        elif ch in _VOWEL:
            out.append(["V", _VOWEL[ch], None, False])
        elif ch in "ंँ":
            out.append(["N", "n", None, False])
        elif ch == "ः":
            out.append(["V", "h", None, False])
        i += 1
    return out


def _word_to_latin(word: str) -> str:
    units = _units(word)
    # A consonant without its own vowel carries an inherent "a" (schwa) unless a virama suppressed it.
    schwa = [u[0] == "C" and u[2] is None and not u[3] for u in units]
    has_vowel = [u[0] == "V" or (u[0] == "C" and (u[2] is not None or schwa[j])) for j, u in enumerate(units)]
    # Hindi drops the word-final schwa, and a medial schwa between VC_ and _CV ("ki-cha-di" -> "khichdi").
    cons = [j for j, u in enumerate(units) if u[0] == "C"]
    if cons and schwa[cons[-1]] and cons[-1] == len(units) - 1 and len(cons) > 1:
        schwa[cons[-1]] = False
    # Right to left, so a deletion makes the consonant before it keep its own schwa (no two in a row).
    for j in range(len(units) - 2, 0, -1):
        if not schwa[j]:
            continue
        nxt = units[j + 1]
        if has_vowel[j - 1] and nxt[0] == "C" and (nxt[2] is not None or schwa[j + 1]):
            schwa[j] = False
            has_vowel[j] = False
    out = []
    for j, u in enumerate(units):
        kind, lat, vow, _ = u
        if kind == "C":
            out.append(lat + (vow if vow is not None else ("a" if schwa[j] else "")))
        elif kind == "N":
            nxt = next((x[1] for x in units[j + 1:] if x[0] == "C"), "")
            out.append("m" if nxt[:1] in _LABIAL else "n")
        else:
            out.append(lat)
    s = "".join(out)
    return re.sub(r"av$", "ao", s)


def deva_to_latin(text: str) -> str:
    return _DEVA_WORD.sub(lambda m: _word_to_latin(m.group(0)), text)


def soft(word: str) -> str:
    """Spelling-insensitive form: long vowels, aspiration and doubled letters collapse."""
    w = word.lower()
    w = re.sub(r"ck", "k", w)
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = re.sub(r"c(?!h)", "k", w)
    w = w.replace("ph", "f").replace("w", "v").replace("z", "j").replace("q", "k").replace("x", "ks")
    w = re.sub(r"(?<=[bdgjklmnprtv])h", "", w)
    w = re.sub(r"(?<=[aeiou])y(?=[aeiou]|$)", "i", w)
    w = re.sub(r"ey$|ee|ie$", "i", w)
    w = re.sub(r"oo|ou(?=[^aeiou]|$)", "u", w)
    w = re.sub(r"aa", "a", w)
    return re.sub(r"(.)\1+", r"\1", w)


def skeleton(word: str) -> str:
    s = soft(word)
    return s[:1] + re.sub(r"[aeiouy]", "", s[1:])


def _edit(a: str, b: str, cap: int = 3) -> float:
    """Edit distance where vowel changes cost half: romanizations disagree on vowels far more than consonants."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    cost = lambda ch: 0.5 if ch in "aeiouy" else 1.0  # noqa: E731
    prev = [0.0]
    for cb in b:
        prev.append(prev[-1] + cost(cb))
    for ca in a:
        cur = [prev[0] + cost(ca)]
        for j, cb in enumerate(b, 1):
            sub = 0.0 if ca == cb else (0.5 if ca in "aeiouy" and cb in "aeiouy" else 1.0)
            cur.append(min(prev[j] + cost(ca), cur[j - 1] + cost(cb), prev[j - 1] + sub))
        prev = cur
    return prev[-1]


class Speller:
    """Maps out-of-vocabulary words to the most frequent corpus word that sounds the same."""

    def __init__(self, vocab: Counter):
        self.vocab = vocab
        self.by_soft: dict[str, str] = {}
        self.by_skel: dict[str, list[str]] = {}
        for w, _ in vocab.most_common():
            if len(w) < 3 or not w.isalpha():
                continue
            self.by_soft.setdefault(soft(w), w)
            self.by_skel.setdefault(skeleton(w), []).append(w)
        self._cache: dict[str, str] = {}

    def correct(self, word: str) -> str:
        if word in self.vocab or len(word) < 4 or not word.isalpha():
            return word
        if word in self._cache:
            return self._cache[word]
        s = soft(word)
        best = self.by_soft.get(s)
        if best is None:
            sk = skeleton(word)
            cands = []
            for key in [sk] + [k for k in self.by_skel if k[:1] == sk[:1] and abs(len(k) - len(sk)) <= 1
                               and k != sk and _edit(k, sk, 1) <= 1]:
                # Same consonant skeleton ("batar"/"butter") tolerates more vowel drift than a neighbour.
                cap = 2 if key == sk and len(s) >= 4 else max(1, len(s) // 3)
                for w in self.by_skel.get(key, [])[:20]:
                    d = _edit(soft(w), s)
                    if d <= cap:
                        # Consonants survive romanization better than vowels: prefer the same skeleton.
                        # A common corpus word beats a rare one at similar distance ("rais" -> rice, not ras).
                        cands.append((key != sk, d - 0.25 * math.log10(self.vocab[w]), w))
            best = min(cands)[-1] if cands else word
        self._cache[word] = best
        return best

    def correct_text(self, text: str) -> str:
        return re.sub(r"[a-z]+", lambda m: self.correct(m.group(0)), text.lower())
