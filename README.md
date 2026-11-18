# RecipeForge

An ingredient-grounded recipe assistant. Every dietary claim it makes is verified in code against the recipe's actual ingredients and cooking steps, never inferred from the recipe's name or tags, and never decided by a language model.

Built as a working answer to failures observed in a production cooking-robot assistant: a "no onion, no garlic" request returning a dal whose ingredients include garlic paste; a Jain filter returning 80+ unranked, unvalidated recipes; and dietary queries answered with alphabetical dumps.

## What it does

- **Understands the request.** English, Hinglish and Devanagari ("bina pyaz lahsun ka khana", "kuch spicy banana hai") are parsed into hard constraints (diet, allergy, "no X", time limit) and soft preferences (ingredients on hand, course, cuisine, spicy/creamy/healthy…).
  - A fast rule parser handles every message.
  - When a message has phrasing the rules can miss ("hold the onions", "my kid can't eat peanuts", "ready in 25"), Claude also parses it with structured output.
  - The two results are combined so Claude can only add restrictions, never remove one.
- **Knows what is in each dish.** Every ingredient line in 6,109 recipes is mapped to one of 249 canonical ingredients through 2,200+ aliases (99.98% line coverage). Compound items are expanded: ginger garlic paste → ginger + garlic, schezwan sauce → garlic + chilli. Brand-dependent items carry explicit "may contain" risk: ketchup and instant-noodle tastemakers (onion/garlic), compounded hing (wheat).
- **Cross-checks the cooking steps.** If a method uses garlic, onion, egg, chicken, fish, shellfish or mushroom that the ingredient list omits, the recipe is flagged instead of passing. This catches source-data errors such as a "Garlic-and-Oil" pasta whose list says "cloves (laung)" while the method says "add the minced garlic".
- **Enforces diets in code.** 17 rule sets (vegetarian, eggetarian, vegan, Jain, sattvic, no-onion-no-garlic, lactose-free, dairy-free, gluten-free, nut / tree-nut / peanut / egg / soy / sesame / shellfish / fish-free) give each recipe a pass / flag / block verdict with the offending line as the reason.
- **Adapts near-misses.** A recipe ruled out for your diet is offered with validated swaps where possible: Jain dal makhani (ginger → sonth), no-onion-garlic pad thai (onion → cabbage, garlic → hing), vegan paneer butter masala (paneer → tofu, cream → cashew paste). Every adapted recipe is re-checked against all active constraints, and a swap that changes a dish's named ingredient is called out.
- **Ranks instead of dumping.** It filters first, then runs BM25 and dense retrieval over the survivors and merges them with reciprocal rank fusion. Preference features (pantry overlap, dish-name match, spicy/creamy/healthy, cuisine) adjust the scores, and MMR keeps the results varied. You get the top 5, each with the checks it passed.
- **Holds a conversation.** Diets and allergies persist across turns as locked chips. Refinements ("make it under 15 minutes", "something similar but spicier") merge with what came before, and a new request replaces the soft preferences. A requested ingredient that conflicts with the diet is swapped (butter chicken + vegetarian → paneer).
- **Learns your taste.** "Cooked it", 👍 and 👎 on any dish build a history kept on your device and sent with each request.
  - Rankings lean toward what you like and skip what you cooked this week.
  - "Something different from what I usually make" avoids your usual cuisines and ingredients.
  - "What did I cook last week?" answers from that real history.
  - With no history, it says so instead of pretending.
- **Plans meals.** "Plan my week, Jain dinners" returns N validated dishes:
  - no main ingredient repeated on consecutive days or more than twice;
  - cuisines rotated, at most two rich dishes;
  - no desserts, drinks, condiments or baby food posing as dinner (source course labels are unreliable);
  - a combined shopping list.
- **Fails honestly.** When nothing qualifies, it says so and offers the closest option that still meets every dietary rule. It never relaxes a dietary or allergy constraint to fill the gap.

## Results

| Check | Result |
|---|---|
| Ingredient lines resolved to the ontology | 99.98% of 76,358 |
| Recipes whose source diet label contradicts their own ingredients | 263 (e.g. "Sattvic" tuna sandwich; "Vegan" dishes with honey, ghee, curd) |
| Recipes whose cooking steps use a diet-relevant ingredient missing from their list | 94, flagged |
| Independent regex audit of every recipe passing no-onion-garlic / Jain / vegan | 0 leaks |
| Silent constraint violations, 50 constrained queries × top 5 × 4 configurations | 0 (see `reports/evaluation.md`) |
| Parsed requests that would let a ruled-out recipe through, 70 hand-written queries | rules 10% → rules + Claude **0%** (see `reports/parser_benchmark.md`) |
| Deployed profile latency / engine load | ~20 ms median per query (about 2 s when Claude is consulted) / ~0.7 s |
| Tests | 76 passing |

Compliance is measured by an independent regex checker over raw ingredient text, deliberately separate from the ontology the system itself uses.

## Two profiles

| | Fast (default, deployed) | Full |
|---|---|---|
| Dense retrieval | Model2Vec `potion-base-8M` static embeddings: numpy only, 30 MB, <1 ms/query | `bge-small-en-v1.5` via ONNX |
| Reranking | Preference features + dish-name match | + cross-encoder `ms-marco-MiniLM-L-6` |
| Median latency | ~20 ms | ~830 ms |
| Runs on | Serverless (Vercel), any small CPU | Docker |

Diet safety is identical in both: it is enforced before ranking. Select the profile with `FORGE_PROFILE=fast|full`.

The server is stateless. The browser holds the conversation state and sends it with each request, and the server validates every field, so it scales horizontally and survives cold starts.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # or: source .venv/bin/activate
pip install -r requirements-full.txt
uvicorn forge.api:app --port 8000                       # open http://localhost:8000
pytest -q
python scripts/evaluate.py                              # writes reports/evaluation.md
```

Rebuild the data from the raw CSV:

```bash
python -m forge.build_corpus data/raw/IndianFoodDatasetCSV.csv   # recipes.jsonl + audit.json
python -m forge.search fast       # Model2Vec embeddings
python -m forge.search full       # bge embeddings (full profile)
python -m forge.search index      # precomputed index.pkl for fast cold starts
```

**Deploy to Vercel:** import the repo. `app.py` exposes the FastAPI app, `public/` is served from the CDN, and `.vercelignore` keeps the function bundle small.
**Docker (full profile):** `docker build -t recipeforge . && docker run -p 8000:8000 recipeforge`.

Claude parser: set `ANTHROPIC_API_KEY` (and optionally `CLAUDE_PARSER_MODEL`, default `claude-opus-5-5`). Without it, the app runs rules-only.

Optional: set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` (any OpenAI-compatible endpoint) to let an LLM reword the reply's intro. Its output is discarded if it mentions a recipe that wasn't validated or drops any number. Recipe cards and every compliance decision stay deterministic.

## Layout

```
data/ontology/ingredients.yaml     canonical ingredients: aliases, tags, may-contain, family, components
data/ontology/diets.yaml           diet and allergen rules over those tags
data/ontology/substitutions.yaml   curated swaps used to adapt recipes
forge/ontology.py                  alias matching, tag inheritance, verdicts
forge/build_corpus.py              raw CSV → normalized recipes, verdicts, method cross-check, label audit
forge/query.py                     message → constraints (rules; English/Hinglish)
forge/search.py                    hard filter → BM25 + dense → RRF → features (→ cross-encoder) → MMR
forge/adapt.py                     swap offending ingredients and re-validate
forge/profile.py                   taste profile from client-held cooking history
forge/plan.py                      multi-day planner with variety constraints and shopping list
forge/chat.py                      stateless dialogue state, swaps, grounded replies
forge/api.py                       FastAPI app
public/                            no-build frontend
```

## Data and licensing

The recipe corpus is the Archana's Kitchen "6000+ Indian Food Recipes" dataset (Mendeley Data, DOI 10.17632/xsphgmmh7b.1, CC BY 4.0). Recipe text originates from archanaskitchen.com, and each result links back to it. This is a demo corpus; a production deployment would run the same pipeline over the product's own recipe catalogue. Model2Vec `potion-base-8M` is MIT-licensed.

## Next

- Fine-tune a small model (Qwen2.5-3B, QLoRA) as the query parser and as a method reader that can only add risk flags, never remove them, and benchmark it against the rules.
- Add nutrient targets to the planner once the catalogue has nutrition data (e.g. IFCT-linked).
- Create a human-labelled relevance set and report nDCG.
