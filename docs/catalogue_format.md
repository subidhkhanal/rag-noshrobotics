# Bringing your own recipe catalogue

RecipeForge ships with a demo corpus (Archana's Kitchen, 6,109 recipes). To run it on a robot maker's own
catalogue, such as the Nosh One's recipes, export them as JSON in the format below and import them:

```bash
python scripts/import_catalogue.py your_recipes.json data/catalogues/nosh
FORGE_DATA_DIR=data/catalogues/nosh uvicorn forge.api:app      # or set FORGE_DATA_DIR on Vercel
```

The importer maps every ingredient line to the food ontology, computes diet and allergen verdicts from the real
ingredients, and builds the search index and embeddings. It lists ingredient lines it couldn't recognise, so
you can add aliases to `data/ontology/ingredients.yaml` before going live. Unrecognised lines are never
treated as safe.

## Format

A JSON list. Required fields are `name`, `ingredients` and `steps`; everything else is optional.

```json
[
  {
    "id": 101,
    "name": "Jeera Aloo",
    "cuisine": "North Indian",
    "course": "Dinner",
    "servings": 4,
    "prep_min": 10,
    "cook_min": 20,
    "ingredients": ["3 potatoes, boiled and cubed", "1 tsp cumin seeds", "1/2 tsp turmeric powder", "Salt to taste"],
    "steps": ["Boil and cube the potatoes.", "Heat oil, add cumin seeds.", "Add potatoes and spices, stir for 5 minutes."],
    "url": "https://example.com/recipes/jeera-aloo",
    "robot": {
      "cookable": true,
      "prep_steps": ["Boil and cube the potatoes."],
      "prep_min": 10,
      "reason": ""
    }
  }
]
```

| Field | Meaning |
|---|---|
| `ingredients` | One string per line, as written in the recipe. Commas inside a line are fine. |
| `steps` | A list of steps, or one text block. |
| `robot` | Optional, and the most valuable field. Your ground truth for the robot: whether it can cook the dish, which steps the user does by hand, and why not if it can't. When present it replaces the rule-based guess in `forge/robot.py`. |
| `diet_label` | Your own label, kept for display only. Diet checks always use the ingredient lines. |

## Robot settings

The Nosh One figures used by default come from public reports, not from Nosh. Override them with environment
variables:

| Variable | Default | Used for |
|---|---|---|
| `NOSH_SPICE_SLOTS` | 8 | flagging dishes that need more spices than the containers hold |
| `NOSH_PAN_LITRES` | 4 | documentation of the pan size behind the batch estimate |
| `NOSH_MAX_BATCH_SERVINGS` | 6 | "cook in 2 batches" notes for large groups |
| `NOSH_MODE` | 1 | new chats show only robot-cookable dishes (0 to show everything) |
