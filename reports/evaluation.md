# RecipeForge evaluation

50 constrained queries, top-5 results each. Compliance is checked by an independent regex checker over raw ingredient lines (not the ontology). A checker hit is a *silent violation* only if the system called the recipe a clean pass; hits on results the system flagged ⚠ were disclosed to the user. Relevance proxies: share of requested ingredients present, course/cuisine match rate. Lower intra-list similarity = more varied results.

| Config | Results | Silent violations | Checker hits that were flagged ⚠ | Flagged ⚠ | Empty | Ingredient coverage | Course match | Cuisine match | Intra-list sim | p50 ms | p95 ms | Cold start s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 only | 250 | **0** | 0 | 7 | 0 | 0.85 | 0.97 | 1.00 | 0.657 | 11 | 41 |  |
| Fast: BM25 + Model2Vec (deployed) | 250 | **0** | 0 | 19 | 0 | 0.89 | 0.97 | 1.00 | 0.691 | 13 | 15 | 0.4 |
| Full: BM25 + bge-small | 250 | **0** | 0 | 11 | 0 | 0.82 | 0.99 | 1.00 | 0.665 | 35 | 134 |  |
| Full: BM25 + bge-small + cross-encoder | 250 | **0** | 0 | 8 | 0 | 0.81 | 0.99 | 1.00 | 0.672 | 477 | 587 | 1.9 |

## Reading the table

- **Compliance:** 0 silent violations in 1,000 results across all four configurations. Diets are enforced in code before ranking, so no ranker can introduce a violation. "Checker hits that were flagged ⚠" are cases where the strict regex checker disagrees but the system had already warned the user, e.g. aged cheddar under lactose-free, which is flagged as trace lactose rather than blocked.
- **The deployed configuration is the fast profile** (BM25 + Model2Vec static embeddings, no neural reranker). It has about 20 ms median latency and about 0.7 s engine load. The full profile's cross-encoder costs about 0.8 s per query on CPU.
- **Ranking proxies are coarse.** Ingredient coverage and course/cuisine match reward lexical overlap; they can't measure whether "what's the healthiest option" returns genuinely light dishes. Spot checks:
  - Cross-encoder, "healthiest option": Karela Tikki and Oats Methi Muthia instead of tandoori chicken and cake.
  - Fast profile, "healthiest option": healthy-tagged smoothies, salads and ragi dishes.
  - Fast profile, "butter chicken but vegetarian": Paneer Butter Masala first.
  - Fast profile, "dal but creamy": Dal Makhani and Trevti Dal first.
- **Fast-profile name-match boost.** It ranks dishes higher when their name contains the requested ingredients or words. This recovers much of the cross-encoder's precision on dish-name queries without its latency.
- **Next step for relevance:** a human-labelled judgement set (about 200 query–recipe pairs) to report nDCG@5 instead of proxies.
