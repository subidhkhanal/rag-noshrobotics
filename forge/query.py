"""Rule-based query parser: natural language (English/Hinglish) -> structured constraints.

Hard constraints (diets, exclusions, max time) are enforced in code by the search layer.
Soft preferences (ingredients, course, cuisine, attributes, free text) only affect ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ontology import Ontology

DIET_PATTERNS: list[tuple[str, str]] = [
    ("no_onion_garlic",
     r"(no|without|zero|free of|minus|skip)\s+(onions?|pyaa?z|kanda)\s*(,|and|or|&|n|aur|\+)?\s*(no\s+|without\s+)?(garlic|lahsun|lehsun|lasun)"
     r"|(no|without)\s+(garlic|lahsun|lehsun)\s*(,|and|or|&|aur)?\s*(no\s+|without\s+)?(onions?|pyaa?z)"
     r"|onion[- ]?(and[- ]|&[- ]?)?garlic[- ]free|bina\s+(pyaa?z|kanda)\s*(aur\s+)?(lahsun|lehsun)?|(pyaa?z|kanda)\s+(lahsun|lehsun)\s+(nahi|nahin|ke bina)"),
    ("jain", r"\bjain\b"),
    ("sattvic", r"\bsatt?vic\b|\bsatvik\b|\bvrat\b|\bfasting\b|\bnavratri\b|\bupvas\b"),
    ("vegan", r"\bvegan\b|plant[- ]based"),
    ("eggetarian", r"\beggetarian\b|eggs? (are|is) (ok|fine|allowed)"),
    ("vegetarian", r"(?<!non[- ])(?<!non)\b(veg|vegetarian|veggie|shakahari|pure veg)\b(?!etable)"),
    ("lactose_free", r"lactose[- ]?(intoleran\w*|free|sensitive)"),
    ("dairy_free", r"dairy[- ]?free|no dairy|without dairy|milk allergy|allergic to (milk|dairy)|dairy allergy"),
    ("gluten_free", r"gluten[- ]?free|no gluten|without gluten|celiac|coeliac|gluten intoleran\w*|gluten allergy|wheat allergy|allergic to (wheat|gluten)"),
    ("tree_nut_free", r"tree[- ]?nut[- ]?(free|allergy)|allergic to tree ?nuts"),
    ("peanut_free", r"peanut[- ]?(free|allergy)|allergic to (peanuts?|groundnuts?)|no peanuts?|without peanuts?"),
    ("nut_free", r"\bnut[- ]?free|\bno nuts\b|without (any )?nuts|\bnut allergy|allergic to nuts"),
    ("egg_free", r"egg[- ]?free|\bno eggs?\b|without eggs?|eggless|egg allergy|allergic to eggs?"),
    ("soy_free", r"soy[a]?[- ]?free|soy allergy|allergic to soy|no soy"),
    ("sesame_free", r"sesame[- ]?(free|allergy)|allergic to sesame|no sesame"),
    ("shellfish_free", r"shellfish[- ]?(free|allergy)|allergic to (shellfish|prawns?|shrimps?)|seafood allergy|allergic to seafood"),
    ("fish_free", r"fish allergy|allergic to fish|seafood allergy|allergic to seafood"),
    ("dairy_free", r"\bno milk\b|without milk"),
    ("eggetarian", r"\bno meat\b|without meat|meatless"),
]
NON_VEG = re.compile(r"\bnon[- ]?veg(etarian)?\b|\bchicken\b|\bmutton\b|\bfish\b|\bprawns?\b|\bmeat\b(?! masala)", re.I)

EXCLUDE_RE = re.compile(
    r"\b(?:no|without|not|skip|avoid|except|minus|hate|dislike|don'?t (?:like|want|eat)|allergic to|bina|zero)\s+"
    r"((?:[a-zऀ-ॿ]+(?:\s*(?:,|and|or|&|aur|nor)\s*|\s+)?){1,4})",
    re.I,
)
HINDI_EXCLUDE_RE = re.compile(r"\b([a-zऀ-ॿ]+)\s+(?:nahi|nahin|mat|na)\b", re.I)
INCLUDE_RE = re.compile(
    r"\b(?:i have|i've got|i got|have got|got some|with|using|use up|made with|containing|includes?|from)\s+"
    r"((?:[a-zऀ-ॿ]+(?:\s*(?:,|and|&|aur|plus)\s*|\s+)?){1,8})",
    re.I,
)
PANTRY_RE = re.compile(r"\b(i have|i've got|i got|have got|in my (fridge|pantry|kitchen)|leftover|use up)\b", re.I)
STOP_AT = re.compile(r"\b(under|within|in|for|that|which|but|no|without|less|max|quick|and make|recipe|recipes|dish|dishes|something|please)\b.*$", re.I)

TIME_RES = [
    (re.compile(r"(?:under|within|less than|in|max(?:imum)?|below|up ?to|upto|<|no more than|not more than|only)\s*(\d{1,3})\s*(?:min|mins|minutes|minute)\b", re.I), "min"),
    (re.compile(r"(\d{1,3})\s*(?:min|mins|minutes)\s*(?:or less|max|tops|mein|me)", re.I), "min"),
    (re.compile(r"(?:under|within|in|less than)\s*half an hour", re.I), "half"),
    (re.compile(r"(?:under|within|in|less than)\s*(?:an|one|1)\s*(?:hour|hr)\b", re.I), "hour"),
    (re.compile(r"(?:under|within|in|less than)\s*(\d)\s*(?:hours|hrs|hour|hr)\b", re.I), "hours"),
]
COURSE_RES = {
    "breakfast": r"\b(breakfast|nashta|nasta|tiffin|morning meal|brunch)\b",
    "lunch": r"\b(lunch|lunchbox|tiffin box)\b",
    "dinner": r"\b(dinner|supper|raat ka khana)\b",
    "snack": r"\b(snacks?|starters?|appetizers?|evening snack|chai time|tea time|party snack)\b",
    "dessert": r"\b(desserts?|sweets?|mithai|sweet dish|pudding)\b",
    "side": r"\b(side dish|sides?|accompaniment|sabzi|sabji)\b",
}
CUISINE_GROUPS = {
    "south indian": ["South Indian", "Tamil Nadu", "Kerala", "Karnataka", "Andhra", "Chettinad", "Udupi", "Mangalorean",
                     "Coorg", "North Karnataka", "Coastal Karnataka", "South Karnataka", "Malabar", "Kongunadu", "Hyderabadi"],
    "north indian": ["North Indian", "Punjabi", "Kashmiri", "Awadhi", "Mughlai", "Rajasthani", "Uttar Pradesh", "Lucknowi",
                     "Himachal", "Haryana", "Bihari", "Uttarakhand-North Kumaon", "Sindhi"],
    "punjabi": ["Punjabi"], "gujarati": ["Gujarati"], "bengali": ["Bengali"], "kerala": ["Kerala", "Malabar"],
    "tamil": ["Tamil Nadu", "Chettinad", "Kongunadu"], "chettinad": ["Chettinad"], "andhra": ["Andhra", "Hyderabadi"],
    "hyderabadi": ["Hyderabadi"], "karnataka": ["Karnataka", "North Karnataka", "Coastal Karnataka", "South Karnataka", "Udupi", "Mangalorean", "Coorg"],
    "maharashtrian": ["Maharashtrian", "Malvani", "Konkan"], "marathi": ["Maharashtrian"], "goan": ["Goan", "Konkan"],
    "rajasthani": ["Rajasthani"], "kashmiri": ["Kashmiri"], "mughlai": ["Mughlai", "Awadhi", "Lucknowi"],
    "parsi": ["Parsi"], "sindhi": ["Sindhi"], "bihari": ["Bihari"], "odia": ["Oriya"], "oriya": ["Oriya"],
    "assamese": ["Assamese"], "north east": ["North East India", "Assamese", "Nagaland"],
    "chinese": ["Chinese", "Indo Chinese", "Cantonese", "Sichuan", "Hunan", "Shandong"], "indo chinese": ["Indo Chinese"],
    "italian": ["Italian"], "mexican": ["Mexican"], "thai": ["Thai"], "continental": ["Continental", "European", "French", "British"],
    "french": ["French"], "asian": ["Asian", "Thai", "Chinese", "Japanese", "Vietnamese", "Indonesian", "Malaysian", "Korean"],
    "japanese": ["Japanese"], "middle eastern": ["Middle Eastern", "Arab", "Afghan"], "lebanese": ["Middle Eastern"],
    "mediterranean": ["Mediterranean", "Greek", "Middle Eastern"], "greek": ["Greek"], "fusion": ["Fusion"],
    "sri lankan": ["Sri Lankan"], "nepali": ["Nepalese"], "american": ["American"], "african": ["African"],
}
ATTRIBUTE_RES = {
    "spicy": r"\b(spicy|spicier|hot|teekha|tikha|fiery|chatpata|masaledar)\b",
    "mild": r"\b(mild|less spicy|not (too )?spicy|non[- ]spicy|kid[- ]friendly|for kids|bland)\b",
    "creamy": r"\b(creamy|rich|malai(dar)?|buttery)\b",
    "healthy": r"\b(healthy|healthier|healthiest|light|lighter|low[- ]?(cal|calorie|fat|oil)|diet|nutritious|weight loss)\b",
    "high_protein": r"\b(high[- ]?protein|protein[- ]?rich|protein)\b",
    "quick": r"\b(quick|quickly|fast|jaldi|in a hurry|instant|easy|simple|busy)\b",
    "comfort": r"\b(comfort|comforting|homely|ghar ka)\b",
    "festive": r"\b(festive|festival|party|guests|special occasion|celebration|diwali|holi)\b",
    "crispy": r"\b(crispy|crunchy|fried)\b",
    "tangy": r"\b(tangy|sour|khatta)\b",
}
INTENT_RES = {
    "reset": r"\b(start over|reset|new search|clear (all|everything)|forget (it|that|everything))\b",
    "more": r"\b(more|another|other options?|something else|show me (more|others)|anything else)\b",
    "different": r"\b(different|new|something new|change|variety|not the same|other than)\b",
    "similar": r"\b(similar|like (this|that|it|the (first|last) one)|same but|along those lines)\b",
    "plan": r"\b(plan|planner|planning|weekly menu|menu for (the )?week|\d+[- ]day menu|week'?s? (worth of )?(meals|dinners|lunches))\b",
    "history": r"\b(what (did|have) i (cook|cooked|make|made|eat|ate)|my (cooking )?history|what i('ve| have)? cooked)\b",
    "usual": r"\b(usually|normally|always|my usual|what i (usually|normally|always))\b",
}
WORD_NUMS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10, "fourteen": 14}
REFINE_START = re.compile(
    r"^\s*(make it|make them|but|and|also|only|now|what about|how about|instead|same|similar|something similar|"
    r"more|another|something else|less|under|within|no |without|with less|with more|quicker|faster|spicier|healthier|"
    r"can you|could you|any|in )",
    re.I,
)
# Diet-driven swaps, tried in order; the first one the active diets allow wins.
SWAPS = {
    "chicken": ["paneer", "soy_chunks", "tofu"], "chicken_stock": ["stock"], "mutton": ["jackfruit", "soy_chunks"],
    "pork": ["jackfruit", "soy_chunks"], "beef": ["jackfruit", "soy_chunks"], "fish": ["raw_banana", "paneer", "tofu"],
    "shellfish": ["paneer", "tofu", "mushroom"], "egg": ["paneer", "tofu", "besan"], "paneer": ["tofu", "soy_chunks"],
    "cheese": ["tofu", "cashew"], "cream": ["cashew", "coconut_milk"], "milk": ["almond_milk", "coconut_milk", "soy_milk"],
    "curd": ["coconut_milk"], "butter": ["oil"], "ghee": ["oil"], "honey": ["jaggery", "sugar"],
    "potato": ["raw_banana", "pumpkin", "gourd"], "mushroom": ["paneer", "tofu", "capsicum"], "onion": ["cabbage"],
    "maida": ["rice_flour", "millet"], "wheat_flour": ["millet", "besan"], "semolina": ["millet", "rice_flour"],
    "pasta": ["rice_noodles"], "noodles": ["rice_noodles"], "bread": ["corn_tortilla", "batter"],
    "cashew": ["seeds", "coconut"], "almond": ["seeds", "coconut"], "peanut": ["seeds", "coconut"],
}
# Ingredients that also name a dish style ("butter chicken", "malai kofta"): keep the word in the search text.
STYLE_WORDS = {"butter", "cream", "ghee", "cheese", "milk"}
HINGLISH_VERB = re.compile(
    r"\b(banana|banani|banane|banaun|banau|pakana|pakani)\s+(hai|hain|he|h|chahiye|chahta|chahti|chahte|hoon|hu|hun|tha)\b"
)
FILLER = re.compile(
    r"\b(i|im|am|me|my|we|our|you|your|want|wanna|would|like|love|need|craving|crave|feel|feeling|something|some|"
    r"anything|suggest|recommend|give|show|find|make|cook|can|could|please|what|whats|which|should|have|has|got|a|an|the|"
    r"for|to|of|is|are|it|that|this|recipes?|dish(es)?|food|options?|ideas?|today|tonight|now|good|nice|tasty|"
    r"delicious|kuch|khana|bana|hai|chahiye|mujhe|ka|ki|ke|mein|lets|do|be|on|in|at|but|and|or|similar|more|another|"
    r"else|different|instead|also|only|just|then|again|one|ones|any|non veg|non vegetarian|nonveg|other|than|usually|"
    r"from|with|up|bit|little|really|very|so|too|much|kind|type|sort|get|try|eat|ate|not|week|weekly|plan|"
    r"planner|menu|meals?|days?|nights?|usually|normally|always|cook|cooked|made|last)\b",
    re.I,
)


@dataclass
class Constraints:
    diets: set[str] = field(default_factory=set)
    exclude: set[str] = field(default_factory=set)
    include: list[str] = field(default_factory=list)
    pantry_mode: bool = False
    max_time: int | None = None
    course: str | None = None
    cuisine: str | None = None
    attributes: set[str] = field(default_factory=set)
    servings: int | None = None
    prefer_non_veg: bool = False
    conflicts: list[str] = field(default_factory=list)
    swaps: dict[str, str] = field(default_factory=dict)
    text: str = ""
    intents: set[str] = field(default_factory=set)
    days: int | None = None

    def has_soft_content(self) -> bool:
        return bool(self.include or self.course or self.cuisine or self.text.strip() or self.prefer_non_veg)

    def to_dict(self) -> dict:
        return {
            "diets": sorted(self.diets), "exclude": sorted(self.exclude), "include": list(self.include),
            "pantry_mode": self.pantry_mode, "max_time": self.max_time, "course": self.course,
            "cuisine": self.cuisine, "attributes": sorted(self.attributes), "servings": self.servings,
            "prefer_non_veg": self.prefer_non_veg, "conflicts": list(self.conflicts), "swaps": dict(self.swaps), "text": self.text, "intents": sorted(self.intents), "days": self.days,
        }


def _ingredients_in(onto: Ontology, phrase: str) -> list[str]:
    ids = []
    for m in onto.match(phrase):
        if "non_food" in onto.ingredients[m.ingredient].tags:
            continue
        if m.ingredient not in ids:
            ids.append(m.ingredient)
    return ids


def parse(onto: Ontology, message: str) -> Constraints:
    low = " " + message.strip().lower().replace("’", "'") + " "
    low = re.sub(r"n't\b", " not", low)
    low = re.sub(r"(\w)'(m|s|re|ve|ll|d)\b", r"\1", low)
    low = HINGLISH_VERB.sub(" ", low)
    c = Constraints()
    consumed: list[str] = []

    for diet, pat in DIET_PATTERNS:
        for m in re.finditer(pat, low, re.I):
            c.diets.add(diet)
            consumed.append(m.group(0))
    if "vegan" in c.diets or "jain" in c.diets or "sattvic" in c.diets:
        c.diets.discard("vegetarian")
        c.diets.discard("eggetarian")
    if "vegetarian" in c.diets:
        c.diets.discard("eggetarian")
    if NON_VEG.search(low) and not c.diets & {"vegetarian", "vegan", "jain", "sattvic", "eggetarian"}:
        c.prefer_non_veg = bool(re.search(r"\bnon[- ]?veg", low))

    for m in EXCLUDE_RE.finditer(low):
        phrase = STOP_AT.sub("", m.group(1))
        ids = _ingredients_in(onto, phrase)
        if ids:
            c.exclude.update(ids)
            consumed.append(m.group(0))
    for m in HINDI_EXCLUDE_RE.finditer(low):
        ids = _ingredients_in(onto, m.group(1))
        if ids:
            c.exclude.update(ids)
            consumed.append(m.group(0))
    for m in re.finditer(r"\b([a-z]+)[- ]free\b", low):
        ids = _ingredients_in(onto, m.group(1))
        if ids:
            c.exclude.update(ids)
            consumed.append(m.group(0))

    for rx, kind in TIME_RES:
        m = rx.search(low)
        if m:
            c.max_time = {"min": lambda: int(m.group(1)), "half": lambda: 30, "hour": lambda: 60,
                          "hours": lambda: int(m.group(1)) * 60}[kind]()
            consumed.append(m.group(0))
            break

    for course, pat in COURSE_RES.items():
        m = re.search(pat, low)
        if m:
            c.course = course
            consumed.append(m.group(0))
            break
    for key in sorted(CUISINE_GROUPS, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(key)}\b", low)
        if m:
            c.cuisine = key
            consumed.append(m.group(0))
            break
    for attr, pat in ATTRIBUTE_RES.items():
        if re.search(pat, low):
            c.attributes.add(attr)
    if "mild" in c.attributes:
        c.attributes.discard("spicy")
    for intent, pat in INTENT_RES.items():
        m = re.search(pat, low)
        if m:
            c.intents.add(intent)
            if intent in ("plan", "history"):
                consumed.append(m.group(0))
    if "plan" in c.intents:
        m = re.search(r"\b(\d{1,2}|" + "|".join(WORD_NUMS) + r")[- ]?(days?|nights?|dinners|lunches|meals)\b", low)
        if m:
            n = m.group(1)
            c.days = max(1, min(14, int(n) if n.isdigit() else WORD_NUMS[n]))
            consumed.append(m.group(0))
        else:
            c.days = 7
    m = re.search(r"\bfor (\d{1,2})\s*(people|persons|guests|pax|of us|members)?\b", low)
    if m and (m.group(2) or int(m.group(1)) <= 20) and not re.search(rf"for {m.group(1)}\s*(min|hour)", low):
        c.servings = int(m.group(1))
        consumed.append(m.group(0))

    c.pantry_mode = bool(PANTRY_RE.search(low))
    include: list[str] = []
    for m in INCLUDE_RE.finditer(low):
        for iid in _ingredients_in(onto, STOP_AT.sub("", m.group(1))):
            if iid not in include:
                include.append(iid)
    excluded_spans = " ".join(consumed).lower()
    for iid in _ingredients_in(onto, low):
        if iid not in include and iid not in c.exclude and not _ingredients_in(onto, excluded_spans).count(iid):
            include.append(iid)
    blocked_tags = {t for d in c.diets for t in onto.diets[d].get("block", [])}
    c.exclude = {e for e in c.exclude if not onto.effective_tags(e) & blocked_tags}
    c.conflicts = [i for i in include if onto.effective_tags(i) & blocked_tags]
    c.include = [i for i in include if i not in c.exclude and i not in c.conflicts]
    for bad in c.conflicts:
        for alt in SWAPS.get(bad, []):
            if not onto.effective_tags(alt) & blocked_tags and alt not in c.exclude:
                c.swaps[bad] = alt
                if alt not in c.include:
                    c.include.append(alt)
                break

    remainder = low
    for span in sorted(consumed, key=len, reverse=True):
        remainder = remainder.replace(span.lower(), " ")
    remainder = re.sub(r"\b(no|without|not|avoid|bina|nahi|nahin|mat)\b\s+\w+", " ", remainder)
    remainder = FILLER.sub(" ", remainder)
    remainder = re.sub(r"[^a-z0-9ऀ-ॿ ]+", " ", remainder)
    c.text = re.sub(r"\s+", " ", remainder).strip()
    for bad, alt in c.swaps.items():
        if bad in STYLE_WORDS:
            continue
        c.text = re.sub(rf"\b{re.escape(onto.label(bad))}\b", onto.label(alt), c.text)
    return c


def is_refinement(message: str, c: Constraints) -> bool:
    if c.intents & {"more", "different", "similar"}:
        return True
    if REFINE_START.search(message):
        return True
    return not c.has_soft_content()
