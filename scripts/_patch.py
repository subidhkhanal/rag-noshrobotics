import pathlib, sys
p = pathlib.Path("forge/query.py"); s = p.read_text(encoding="utf-8")
R = [
 ('    "lunch": r"\b(lunch|lunchbox|tiffin box)\b",', '    "lunch": r"\b(lunch(es)?|lunchbox(es)?|tiffin box)\b",'),
 ('    "dinner": r"\b(dinner|supper|raat ka khana)\b",', '    "dinner": r"\b(dinners?|suppers?|raat ka khana)\b",'),
 ('    "breakfast": r"\b(breakfast|nashta|nasta|tiffin|morning meal|brunch)\b",', '    "breakfast": r"\b(breakfasts?|nashta|nasta|tiffin|morning meal|brunch)\b",'),
 ('low[- ]?(cal|calorie|fat|oil)|diet|nutritious|weight loss)\b",', 'low[- ]?(cal|calorie|fat|oil)|diet|nutritious|weight loss|halka|not (too )?oily|less oil)\b",'),
 ('    (re.compile(r"\b(\d{1,3})\s*(?:min|mins|minutes)\b", re.I), "min"),', '    (re.compile(r"\b(\d{1,3})[- ]?(?:min|mins|minutes|minute)\b", re.I), "min"),'),
 ("allergic to|bina|zero|", "allergic to|(?<!ke )bina|zero|"),
]
for a, b in R:
    if a not in s:
        sys.exit("MISSING: " + a)
    s = s.replace(a, b, 1)
p.write_text(s, encoding="utf-8")
print("ok")
