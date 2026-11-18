"""HTTP API + static web app.  Run: uvicorn forge.api:app --port 8000"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chat import Assistant, chips
from .search import DATA, Engine

ROOT = Path(__file__).resolve().parent.parent
engine = Engine()
assistant = Assistant(engine)
AUDIT = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))

app = FastAPI(title="RecipeForge", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, max_length=64)


class RemoveIn(BaseModel):
    session_id: str = Field(max_length=64)
    kind: str = Field(max_length=20)
    value: str = Field(default="", max_length=60)


@app.get("/api/health")
def health():
    return {"ok": True, "recipes": len(engine.recipes), "dense": engine.emb is not None}


@app.post("/api/chat")
def chat(body: ChatIn):
    return assistant.chat(body.session_id, body.message)


@app.post("/api/constraints/remove")
def remove(body: RemoveIn):
    s = assistant.remove_constraint(body.session_id, body.kind, body.value)
    return {"session_id": s.id, "constraints": chips(engine.onto, s.state)}


@app.get("/api/recipe/{rid}")
def recipe(rid: int, session_id: str | None = None):
    if rid not in engine.by_id:
        raise HTTPException(404, "recipe not found")
    i = engine.by_id[rid]
    r = engine.recipes[i]
    onto = engine.onto
    st = assistant.sessions[session_id].state if session_id in assistant.sessions else None
    block_tags = {t for d in (st.diets if st else []) for t in onto.diets[d].get("block", [])}
    lines = []
    for it in r["ingredients"]:
        status, note = "ok", ""
        for iid in it["ids"]:
            tags, may = onto.effective_tags(iid), onto.effective_may(iid)
            if tags & block_tags or (st and any(e in onto.lineage(iid) for e in st.exclude)):
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
        "verdicts": {d: {"label": onto.diets[d]["label"], **v} for d, v in r["verdicts"].items()},
    }


@app.get("/api/audit")
def audit():
    return AUDIT


web = ROOT / "web"
if web.exists():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
