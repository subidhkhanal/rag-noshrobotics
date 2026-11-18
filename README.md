# RecipeForge

An ingredient-grounded recipe assistant. Every dietary claim it makes is verified in code against the recipe's actual ingredient list, never inferred from the recipe's name or tags, and never decided by a language model.

Built as a working answer to failures observed in a production cooking-robot assistant: a "no onion, no garlic" request returning a dal whose ingredients include garlic paste; a Jain filter returning 80+ unranked, unvalidated recipes; and dietary queries answered with alphabetical dumps.

## What it does

- **Understands the request.** English, Hinglish and Devanagari ("bina pyaz lahsun ka khana", "kuch spicy banana hai") are parsed into hard constraints (diet, allergy, "no X", time limit) and soft preferences (ingredients on hand, course, cuisine, spicy/creamy/healthy…).
- **Knows what is in each dish.** Every ingredient line in 6,111 recipes is mapped to one of 248 canonical ingredients through 2,200+ aliases (99.98% line coverage). Compound items are expanded: ginger garlic paste → ginger + garlic, schezwan sauce → garlic + chilli. Brand-dependent items carry explicit "may contain" risk: ketchup and instant-noodle tastemakers (onion/garlic), compounded hing (wheat).
- **Enforces diets in code.** 17 rule sets (vegetarian, eggetarian, vegan, Jain, sattvic, no-onion-no-garlic, lactose-free, dairy-free, gluten-free, nut / tree-nut / peanut / egg / soy / sesame / shellfish / fish-free) give each recipe a pass / flag / block verdict with the offending line as the reason.
- **Ranks instead of dumping.** It filters first, then runs BM25 and dense retrieval (bge-small) over the survivors, merges them with reciprocal rank fusion, reranks with a cross-encoder, applies preference features, and uses MMR so results are varied. You get the top 5, each with the checks it passed.
- **Holds a conversation.** Diets and allergies persist across turns as locked chips. Refinements ("make it under 15 minutes", "something similar but spicier") merge with what came before, and a new request replaces the soft preferences. When a requested ingredient conflicts with the diet, it is swapped (butter chicken + vegetarian → paneer).
- **Fails honestly.** When nothing qualifies, it says so and offers the closest option that still meets every dietary rule. It never relaxes a dietary or allergy constraint to fill the gap.

## Results

| Check | Result |
|---|---|
| Ingredient lines resolved to the ontology | 99.98% of 76,358 |
| Recipes whose source diet label contradicts their own ingredients | 263 (e.g. "Sattvic" tuna sandwich; "Vegan" dishes with honey, ghee, curd) |
| Independent regex audit of every recipe passing no-onion-garlic / Jain / vegan | 0 leaks |
| Constraint violations across 50 constrained queries × top 5 | 0 (see `reports/evaluation.md`) |
| Unit and regression tests | `pytest` — all passing |

Compliance is measured by an independent regex checker over raw ingredient text, deliberately separate from the ontology the system itself uses.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # or source .venv/bin/activate
pip install -r requirements.txt
python -m forge.build_corpus data/raw/IndianFoodDatasetCSV.csv   # only if rebuilding from the raw CSV
python -m forge.search                                  # builds data/processed/embeddings.npy
uvicorn forge.api:app --port 8000                       # open http://localhost:8000
pytest -q
python scripts/evaluate.py                              # writes reports/evaluation.md
```

Docker: `docker build -t recipeforge . && docker run -p 8000:8000 recipeforge`. The image bakes in the ONNX models (no PyTorch), so it runs on a small CPU instance.

Optional: set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` (any OpenAI-compatible endpoint) to let an LLM reword the reply's intro. Its output is discarded if it mentions a recipe that wasn't validated or drops any number. Recipe cards and every compliance decision stay deterministic.

## Layout

```
data/ontology/ingredients.yaml   canonical ingredients: aliases, tags, may-contain, family, components
data/ontology/diets.yaml         diet and allergen rules over those tags
forge/ontology.py                alias matching, tag inheritance, verdicts
forge/build_corpus.py            raw CSV → normalized recipes + per-diet verdicts + label audit
forge/query.py                   message → constraints (rules; English/Hinglish)
forge/search.py                  hard filter → BM25 + dense → RRF → cross-encoder → MMR
forge/chat.py                    dialogue state, swaps, grounded replies
forge/api.py                     FastAPI + static web app
web/                             no-build frontend (deployable separately; set API_BASE in config.js)
```

## Data and licensing

The recipe corpus is the Archana's Kitchen "6000+ Indian Food Recipes" dataset (Mendeley Data, DOI 10.17632/xsphgmmh7b.1, CC BY 4.0). Recipe text originates from archanaskitchen.com, and each result links back to it. This is a demo corpus; a production deployment would run the same pipeline over the product's own recipe catalogue.

## Next

- Fine-tune a small model (Qwen2.5-3B, QLoRA) as the query parser and benchmark it against the rule parser on a labelled set.
- Add a substitution engine on a curated Indian swap table, with every proposal re-validated.
- Build a persistent taste profile from cooking history.
- Add a nutrient-bounded weekly planner.
