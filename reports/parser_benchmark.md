# Parser benchmark

70 hand-written queries (`finetune/test_set.py`), phrased independently of any training data. **Unsafe** = the parsed request would let through at least one recipe that the correct request blocks (checked over all 6,109 recipes). Claude runs `claude-opus-5-5` at low effort with structured JSON output.

Note: after the first run, the rule parser was fixed for refusal phrasings this set exposed ("hold the …", "… ke bina", "ready in 25"). Its row is therefore optimistic; on that first run the rules alone were unsafe on 10% of queries. Claude's row and the combined rows were unaffected.

| System | Same recipes allowed as gold | Unsafe (lets a ruled-out recipe through) | Over-blocking | Hard-constraint exact | Full exact | include | course | cuisine | attributes | intent |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule parser | 96% | **3%** | 0.1% | 91% | 74% | 93% | 94% | 100% | 93% | 100% |
| Claude Opus 5.5 | 99% | **1%** | 0.1% | 97% | 76% | 96% | 96% | 97% | 90% | 99% |
| Rules ∪ Claude Opus 5.5 | 97% | **0%** | 0.3% | 91% | 70% | 96% | 96% | 97% | 90% | 99% |
| Deployed: rules + Claude when gated | 97% | **0%** | 0.3% | 91% | 74% | 96% | 96% | 100% | 90% | 100% |
| Rules ∪ Deployed: rules + Claude when gated | 97% | **0%** | 0.3% | 91% | 74% | 96% | 96% | 100% | 90% | 100% |

Rule parser: unsafe queries — `no dairy at all, and nothing with soy either` (262 recipes); `a crispy evening snack, nothing deep fried in maida` (573 recipes)

Claude Opus 5.5: unsafe queries — `fasting food, no grains` (2170 recipes)
