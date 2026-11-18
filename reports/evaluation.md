# RecipeForge evaluation

50 constrained queries, top-5 results each. Compliance is checked by an independent regex checker over raw ingredient lines (not the ontology). Relevance proxies: share of requested ingredients present, course/cuisine match rate. Lower intra-list similarity = more varied results.

| Config | Results | Violations | Flagged ⚠ | Empty | Ingredient coverage | Course match | Cuisine match | Intra-list sim | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 only | 250 | **0** | 7 | 0 | 0.91 | 1.00 | 1.00 | nan | 11 | 32 |
| BM25 + dense (RRF) | 250 | **0** | 10 | 0 | 0.84 | 1.00 | 1.00 | 0.779 | 18 | 34 |
| BM25 + dense + cross-encoder | 250 | **0** | 8 | 0 | 0.85 | 1.00 | 1.00 | 0.776 | 702 | 820 |

## Reading the table

- **Compliance is the headline:** 0 violations in 750 results across all three configurations. Diets are enforced before ranking, so the ranker can't introduce a violation.
- **The proxies are coarse.** Ingredient coverage and course/cuisine match reward lexical overlap. They can't see that "what's the healthiest option" should not return tandoori chicken and chocolate cake.
- **Spot checks on vague queries** (BM25-only → full pipeline):
  - "what's the healthiest option": Smoked Tandoori Chicken, Choco Orange Cake → Karela Tikki, Oats Methi Muthia
  - "I have potatoes and peas": Peas Seekh Kebab, Cold Pasta Salad → Aloo Matar Sabzi first
  - "something like dal but creamy": Dal Makhani, Trevti Dal, Creamy Spinach Lentils (stable across configs)
- **The cross-encoder stays** for this semantic gain. It costs about 0.7 s per query on CPU (30 candidates, ms-marco-MiniLM-L-6). A food-domain reranker or a smaller pool would cut that.
- **Next step for relevance:** a human-labelled judgement set (about 200 query–recipe pairs) to report nDCG@5 instead of proxies.
