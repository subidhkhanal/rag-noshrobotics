# RecipeForge evaluation

50 constrained queries, top-5 results each. Compliance is checked by an independent regex checker over raw ingredient lines (not the ontology). A checker hit is a *silent violation* only if the system called the recipe a clean pass; hits on results the system flagged ⚠ were disclosed to the user. Relevance proxies: share of requested ingredients present, course/cuisine match rate. Lower intra-list similarity = more varied results.

| Config | Results | Silent violations | Checker hits that were flagged ⚠ | Flagged ⚠ | Empty | Ingredient coverage | Course match | Cuisine match | Intra-list sim | p50 ms | p95 ms | Cold start s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 only | 250 | **0** | 0 | 11 | 0 | 0.93 | 1.00 | 1.00 | 0.655 | 24 | 37 |  |
| Fast: BM25 + Model2Vec (deployed) | 250 | **0** | 0 | 19 | 0 | 0.94 | 1.00 | 1.00 | 0.704 | 27 | 45 | 1.3 |
| Full: BM25 + bge-small | 250 | **0** | 1 | 14 | 0 | 0.84 | 1.00 | 1.00 | 0.641 | 49 | 84 |  |
| Full: BM25 + bge-small + cross-encoder | 250 | **0** | 0 | 6 | 0 | 0.81 | 1.00 | 1.00 | 0.637 | 1019 | 1151 | 3.8 |

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
