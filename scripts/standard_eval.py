"""Standard RAG evaluation, retrieval and generation measured separately.

Part 1, retrieval: TREC-style metrics with ranx (metrics validated against trec_eval) on the 180-query golden set
plus 20 held-out descriptions, comparing configurations with a paired Student's t-test.

Part 2, generation: RAGAS-method metrics with an LLM judge (Claude Opus; the assistant itself runs on Sonnet, so
the model doesn't grade itself):
  faithfulness       = supported claims / factual claims, checked against the exact tool results the assistant saw
                       (claims that are general cooking knowledge are counted separately, as RAGAS has no context
                       for them)
  answer relevancy   = does the reply address the question (1-5, reported as mean/5)
  context precision  = share of retrieved dishes the judge rates as fitting the request

    python scripts/standard_eval.py retrieval
    ANTHROPIC_API_KEY=... python scripts/standard_eval.py generation
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))  # helper modules (retrieval_eval)

METRICS = ["hit_rate@5", "precision@5", "mrr@10", "map@10", "ndcg@10"]


def retrieval() -> dict:
    from ranx import Qrels, Run, compare, evaluate

    import retrieval_eval as R
    from forge import search as S
    from forge.query import parse
    from forge.search import Engine

    e = Engine()
    golden = R.load_golden()
    holdout = yaml.safe_load((ROOT / "data" / "eval" / "golden_holdout.yaml").read_text(encoding="utf-8"))
    queries = [(f"{g['id']}:{lang}", g[lang], g) for g in golden for lang in ("en", "hinglish", "devanagari")]
    queries += [(f"{g['id']}:en", g["en"], g) for g in holdout]
    qrels = {qid: {str(e.recipes[i]["id"]): 1 for i in range(len(e.recipes)) if R.relevant(e, i, g)}
             for qid, _, g in queries}

    def run_config(name: str, setup) -> Run:
        saved = (e.emb, e.fusion, e.w_desc, e.aka_as_name, e.desc_features, S.DISH_SYNONYMS)
        setup()
        out = {}
        for qid, text, _ in queries:
            res = e.search(parse(e.onto, text), k=20).results
            out[qid] = {str(r.recipe["id"]): 1.0 / (rank + 1) for rank, r in enumerate(res)} or {"none": 0.0}
        e.emb, e.fusion, e.w_desc, e.aka_as_name, e.desc_features, S.DISH_SYNONYMS = saved
        return Run(out, name=name)

    def bm25_only():
        e.emb = None

    def plain_hybrid():
        e.fusion, e.w_desc, e.aka_as_name, e.desc_features = "rrf", 0.0, False, False
        S.DISH_SYNONYMS = []

    runs = [run_config("bm25_only", bm25_only), run_config("hybrid_rrf_baseline", plain_hybrid),
            run_config("current", lambda: None)]
    q = Qrels(qrels)
    report = compare(qrels=q, runs=runs, metrics=METRICS, max_p=0.05, stat_test="student")
    print(report)
    # Per slice for the current system
    slices = {"english": lambda k: k.endswith(":en") and not k.startswith("h_"),
              "hinglish": lambda k: k.endswith(":hinglish"), "devanagari": lambda k: k.endswith(":devanagari"),
              "held-out descriptions": lambda k: k.startswith("h_")}
    cur = runs[2]
    out = {}
    print("\ncurrent system by slice:")
    for name, keep in slices.items():
        sub_q = Qrels({k: v for k, v in qrels.items() if keep(k)})
        sub_r = Run({k: v for k, v in cur.to_dict().items() if keep(k)})
        scores = evaluate(sub_q, sub_r, METRICS)
        out[name] = scores
        print(f"  {name:22} n={len(sub_q.keys()):3}  " + "  ".join(f"{m}={v:.3f}" for m, v in scores.items()))
    return {"compare": str(report), "slices": out}


# ---------------------------------------------------------------- generation (RAGAS method, LLM judge)
GEN_SET = [
    # (setup turns, question)
    ([], "suggest a light vegetarian dinner and tell me why it's light"),
    ([], "I'm tired after work, what's easy to cook tonight?"),
    ([], "my husband eats chicken but I'm vegetarian, what works for both of us?"),
    ([], "what can I make with paneer and peas?"),
    ([], "kuch teekha aur jaldi banne wala batao"),
    ([], "something for a kids' birthday party"),
    ([], "I'm allergic to peanuts, suggest a south indian breakfast"),
    ([], "why is ginger not allowed in jain food?"),
    ([], "is ghee vegan?"),
    ([], "how do I store leftover dal?"),
    (["paneer dinner"], "which of these is quickest and why?"),
    (["paneer dinner"], "is the second one spicy?"),
    (["paneer dinner"], "how do I make the first one?"),
    (["paneer dinner"], "can I swap paneer for tofu in the third one?"),
    (["paneer dinner"], "these look heavy, show me something lighter"),
    (["paneer dinner"], "which ones are jain?"),
    (["jain dinner"], "explain the first one to me"),
    (["jain dinner"], "does the second one have potato?"),
    (["jain dinner"], "replace tomato in the first one with potato"),
    (["plan my week, Jain dinners"], "explain monday's dish"),
    (["plan my week, Jain dinners"], "which day is the quickest?"),
    (["plan my week, Jain dinners"], "swap friday's dish for something with dal"),
    (["vegan lunch"], "what was the last recipe you showed me?"),
    (["vegan lunch"], "what side dish goes well with the first one?"),
    ([], "can the nosh one make roti?"),
    ([], "masala dosa"),
    ([], "what's a good high protein vegetarian dinner?"),
    ([], "aaj kuch accha sa banana hai, kya suggest karoge?"),
    (["no onion no garlic dinner"], "is the first one good for weight loss?"),
    (["no onion no garlic dinner"], "how long does the third one take?"),
]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "verdict": {"type": "string", "enum": ["supported", "unsupported", "general_knowledge"]}},
            "required": ["claim", "verdict"], "additionalProperties": False}},
        "answer_relevancy": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "relevant_dishes": {"type": "integer"},
        "total_dishes": {"type": "integer"},
        "notes": {"type": "string"},
    },
    "required": ["claims", "answer_relevancy", "relevant_dishes", "total_dishes", "notes"],
    "additionalProperties": False,
}
JUDGE_SYSTEM = """You evaluate one turn of a recipe assistant, following the RAGAS method.
You get: the user's question, the CONTEXT (exactly what the app's tools returned to the assistant this turn, plus
the dishes already on screen), and the assistant's ANSWER.
1. Split the ANSWER into atomic factual claims (ignore greetings, offers and questions back to the user).
   For each claim: "supported" if the CONTEXT states or directly implies it; "general_knowledge" if it is
   general cooking/food knowledge not about a specific dish in the catalogue (e.g. why Jain food avoids roots,
   how to store dal); otherwise "unsupported" (including any detail about a specific dish that the CONTEXT does
   not contain, such as invented ingredients, times or steps).
2. answer_relevancy 1-5: does the answer address what the user asked? 5 = fully, 1 = not at all.
3. Context precision: count the dishes listed in the CONTEXT's search results this turn (total_dishes) and how
   many genuinely fit the user's request (relevant_dishes). If no search ran this turn, both are 0.
Be strict and literal."""


def generation() -> dict:
    import anthropic

    from forge.chat import Assistant
    from forge.search import Engine

    os.environ.setdefault("CLAUDE_PER_IP_HOUR", "100000")
    a = Assistant(Engine())
    client = anthropic.Anthropic(max_retries=3)
    rows = []
    for setup, question in GEN_SET:
        state = None
        for m in setup:
            state = a.chat(state, m)["state"]
        screen = [f"{lab}: {a.engine.recipes[a.engine.by_id[rid]]['name']}"
                  for lab, rid in zip((state or {}).get("last_labels") or [], (state or {}).get("last_ids") or [])]
        from forge.agent import SYSTEM as AGENT_SYSTEM
        from forge.chat import session_from
        out = a.chat(state, question)
        tools = (out.get("trace") or {}).get("tools", [])
        product = AGENT_SYSTEM.split("How to work:")[0].strip()  # facts the assistant is given about the Nosh One
        if out.get("agent"):
            context = (f"Product facts given to the assistant:\n{product}\n\n"
                       f"App state given to the assistant:\n{a.agent._context(session_from(a.engine, state))}\n\n"
                       "Tool results this turn:\n"
                       + ("\n\n".join(f"[{t['tool']} {json.dumps(t['input'], ensure_ascii=False)}]\n{t['result']}"
                                      for t in tools) or "(no tool was called)"))
        else:  # deterministic path: the reply is a template over these records
            def card_text(c):
                checks = "; ".join(f"{x['label']}: {x['status']} ({x['detail']})" for x in c.get("checks", []))
                rb = c.get("robot") or {}
                robot = (f"Nosh One cookable, hand-prep steps {len(rb.get('prep', []))}" if rb.get("fit") == "robot"
                         else f"not robot-cookable: {rb.get('reason')}") if rb else ""
                return (f"- {c['name']}: {c.get('total_min')} min, serves {c.get('servings')}; checks: {checks or 'none'}; "
                        f"{robot}; notes: {'; '.join(c.get('why', []))}")
            ruled = "\n".join(f"- {b['name']}: {b['reason']}" for b in out.get("blocked_examples", []))
            context = (f"App state: {json.dumps({k: (out.get('state') or {}).get(k) for k in ('diets', 'exclude', 'robot_only', 'attributes', 'max_time')})}\n"
                       "Records the templated reply was built from:\n" + "\n".join(card_text(c) for c in out.get("results", []))
                       + (f"\nRuled out (closest matches that failed a check):\n{ruled}" if ruled else ""))
        context += "\n\nOn screen before this turn:\n" + ("\n".join(screen) or "nothing")
        resp = client.messages.create(
            model="claude-opus-5-5", max_tokens=4000, system=JUDGE_SYSTEM,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
            messages=[{"role": "user", "content": f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{out['reply']}"}])
        j = json.loads(next(b.text for b in resp.content if b.type == "text"))
        sup = sum(c["verdict"] == "supported" for c in j["claims"])
        uns = sum(c["verdict"] == "unsupported" for c in j["claims"])
        gen = sum(c["verdict"] == "general_knowledge" for c in j["claims"])
        rows.append({"question": question, "setup": setup, "agent": bool(out.get("agent")), "supported": sup,
                     "unsupported": uns, "general": gen, "relevancy": j["answer_relevancy"],
                     "dishes": j["total_dishes"], "relevant_dishes": j["relevant_dishes"],
                     "unsupported_claims": [c["claim"] for c in j["claims"] if c["verdict"] == "unsupported"],
                     "notes": j["notes"], "reply": out["reply"]})
        print(f"  {question[:58]:58} faithful {sup}/{sup + uns}  relevancy {j['answer_relevancy']}  "
              f"dishes {j['relevant_dishes']}/{j['total_dishes']}", flush=True)
    sup, uns = sum(r["supported"] for r in rows), sum(r["unsupported"] for r in rows)
    dish_t, dish_r = sum(r["dishes"] for r in rows), sum(r["relevant_dishes"] for r in rows)
    summary = {
        "turns": len(rows),
        "faithfulness": sup / (sup + uns) if sup + uns else 1.0,
        "turns_fully_faithful": sum(r["unsupported"] == 0 for r in rows) / len(rows),
        "answer_relevancy": sum(r["relevancy"] for r in rows) / (5 * len(rows)),
        "context_precision": dish_r / dish_t if dish_t else None,
        "claims": {"supported": sup, "unsupported": uns, "general_knowledge": sum(r["general"] for r in rows)},
    }
    print("\n" + json.dumps(summary, indent=1))
    for r in rows:
        if r["unsupported_claims"]:
            print(f"\n  UNSUPPORTED in {r['question']!r}:")
            for c in r["unsupported_claims"]:
                print(f"    - {c}")
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    result = retrieval() if part == "retrieval" else generation()
    out = ROOT / "reports" / f"standard_eval_{part}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nsaved {out.relative_to(ROOT)}")
