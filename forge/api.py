"""HTTP API + static web app. Stateless: the client sends conversation state with each request.

Run locally: uvicorn forge.api:app --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .adapt import adapt
from .chat import Assistant, chips, session_from
from .search import DATA, Engine

ROOT = Path(__file__).resolve().parent.parent
engine = Engine()
engine.embed_query("warm up")
assistant = Assistant(engine)
AUDIT = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))

app = FastAPI(title="RecipeForge", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    state: dict | None = None
    history: list | None = Field(default=None, max_length=200)


class RemoveIn(BaseModel):
    state: dict | None = None
    kind: str = Field(max_length=20)
    value: str = Field(default="", max_length=60)


class StateIn(BaseModel):
    state: dict | None = None


@app.get("/api/health")
def health():
    return {"ok": True, "recipes": len(engine.recipes), "profile": engine.profile, "dense": engine.emb is not None}


@app.post("/api/chat")
def chat(body: ChatIn):
    return assistant.chat(body.state, body.message, body.history)


@app.post("/api/constraints/remove")
def remove(body: RemoveIn):
    s = assistant.remove_constraint(body.state, body.kind, body.value)
    return {"state": s.to_dict(), "constraints": chips(engine.onto, s.state)}


@app.post("/api/recipe/{rid}")
def recipe(rid: int, body: StateIn):
    if rid not in engine.by_id:
        raise HTTPException(404, "recipe not found")
    r = engine.recipes[engine.by_id[rid]]
    onto = engine.onto
    st = session_from(engine, body.state).state
    block_tags = {t for d in st.diets for t in onto.diets[d].get("block", [])}
    lines = []
    for it in r["ingredients"]:
        status, note = "ok", ""
        for iid in it["ids"]:
            tags, may = onto.effective_tags(iid), onto.effective_may(iid)
            if tags & block_tags or any(e in onto.lineage(iid) for e in st.exclude):
                status, note = "block", ", ".join(sorted(tags & block_tags)) or "excluded"
            elif status != "block" and may & block_tags:
                status, note = "flag", "may contain " + ", ".join(sorted(may & block_tags))
        if not it["ids"]:
            status, note = "unknown", "not recognised"
        lines.append({"line": it["line"], "matched": [onto.label(x) for x in it["ids"]], "status": status, "note": note})
    return {
        "id": r["id"], "name": r["name"], "cuisine": r["cuisine"], "course": r["course"],
        "total_min": r["total_min"], "prep_min": r["prep_min"], "cook_min": r["cook_min"],
        "servings": r["servings"], "url": r["url"], "source_diet_label": r["source_diet_label"],
        "ingredients": lines, "instructions": r["instructions"],
        "method_only": [onto.label(x) for x in r.get("method_only", [])],
        "verdicts": {d: {"label": onto.diets[d]["label"], **v} for d, v in r["verdicts"].items()},
    }


@app.post("/api/adapt/{rid}")
def adapt_recipe(rid: int, body: StateIn):
    if rid not in engine.by_id:
        raise HTTPException(404, "recipe not found")
    st = session_from(engine, body.state).state
    return adapt(engine.onto, engine.recipes[engine.by_id[rid]], st).to_dict(engine.onto)


@app.get("/api/audit")
def audit():
    return AUDIT


web = ROOT / "public"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
