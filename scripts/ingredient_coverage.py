import collections, sys
import pandas as pd
from forge.ontology import load
from forge.recipes import is_english, parse_ingredients
onto = load()
df = pd.read_csv("data/raw/IndianFoodDatasetCSV.csv")
df = df[df.TranslatedIngredients.map(is_english) & df.TranslatedRecipeName.map(is_english)]
lines = unknown = 0
recipes_with_unknown = 0
unk = collections.Counter()
for raw in df.TranslatedIngredients:
    items = parse_ingredients(onto, raw)
    u = [i for i in items if not i["ids"]]
    lines += len(items); unknown += len(u)
    recipes_with_unknown += bool(u)
    for i in u: unk[i["name"].lower()] += 1
print(f"recipes {len(df)} lines {lines} unknown {unknown} coverage {1-unknown/lines:.4f} recipes_with_unknown {recipes_with_unknown}")
for k, v in unk.most_common(int(sys.argv[1]) if len(sys.argv) > 1 else 80): print(v, k)
