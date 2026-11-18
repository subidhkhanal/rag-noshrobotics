| System | Same recipes allowed as gold | Unsafe (lets a ruled-out recipe through) | Over-blocking | Hard-constraint exact | Full exact | include | course | cuisine | attributes | intent |
|---|---|---|---|---|---|---|---|---|---|---|
| Rule parser | 89% | **10%** | 0.1% | 84% | 67% | 89% | 94% | 100% | 93% | 100% |
| Claude Opus 5.5 | 99% | **1%** | 0.1% | 97% | 76% | 96% | 96% | 97% | 90% | 99% |
| Rules ∪ Claude Opus 5.5 | 97% | **0%** | 0.3% | 91% | 70% | 96% | 96% | 97% | 90% | 99% |
| Deployed: rules + Claude when gated | 97% | **0%** | 0.3% | 91% | 74% | 96% | 96% | 100% | 90% | 100% |
| Rules ∪ Deployed: rules + Claude when gated | 97% | **0%** | 0.3% | 91% | 74% | 96% | 96% | 100% | 90% | 100% |

Rule parser: unsafe queries — `hold the onions please, and make it vegan` (1089 recipes); `ready in 25` (5033 recipes); `no dairy at all, and nothing with soy either` (262 recipes); `lahsun pyaz ke bina kuch banao` (3164 recipes); `a crispy evening snack, nothing deep fried in maida` (573 recipes); `strictly vegetarian, no mushrooms, no paneer` (238 recipes); `hold the cheese, italian pasta` (519 recipes)

Claude Opus 5.5: unsafe queries — `fasting food, no grains` (2170 recipes)
