import pandas as pd, re, sys, collections
df = pd.read_csv(sys.argv[1])
df = df[df.Ingredients == df.TranslatedIngredients]
QTY = re.compile(r"^[\s\d/.\-½¼¾⅓⅔]+")
UNITS = r"(cups?|cup|tablespoons?|teaspoons?|tbsp|tsp|grams?|gms?|g|kg|ml|litres?|liters?|inch|inches|pinch|sprigs?|sprig|bunch|cloves?|pieces?|nos?|handful|cans?|packets?|small|medium|large|big|few|stalks?|sheets?|drops?|slices?|leaves)\b"
c = collections.Counter()
for s in df.Ingredients:
    for part in str(s).split(","):
        p = part.strip().split(" - ")[0]
        p = QTY.sub("", p).strip()
        p = re.sub(r"^" + UNITS, "", p, flags=re.I).strip()
        p = re.sub(r"^(of)\s+", "", p, flags=re.I)
        if p: c[p.lower()] += 1
tot = sum(c.values())
print("distinct", len(c), "total", tot)
cum = 0
for i, (k, v) in enumerate(c.most_common(1500)):
    cum += v
    print(v, k)
