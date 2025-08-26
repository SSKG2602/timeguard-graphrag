from __future__ import annotations
import os
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from tg_graphrag import GraphRAG
from timeguard import parse_time_hint

app = FastAPI(title="TimeGuard Graph-RAG API", version="1.0")

RAG = None        # created in background
STATUS = "starting"
ERR = None


def _init_rag():
    global RAG, STATUS, ERR
    try:
        RAG = GraphRAG(model_name=os.getenv("QWEN_MODEL") or None)
        STATUS = "ready"
    except Exception as e:
        ERR = str(e)
        STATUS = "error"


# kick off background initialization so the API can come up immediately
threading.Thread(target=_init_rag, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"status": STATUS, "error": ERR, "model": os.getenv("QWEN_MODEL") or ""}


def _ensure_ready():
    if STATUS == "ready" and RAG is not None:
        return
    if STATUS == "error":
        raise HTTPException(status_code=500, detail={
                            "status": STATUS, "error": ERR})
    # still starting
    raise HTTPException(status_code=503, detail={"status": STATUS})


class Provenance(BaseModel):
    uri: Optional[str] = None
    observed_at: Optional[str] = None
    source_type: Optional[str] = None


class IngestDoc(BaseModel):
    external_id: str
    text: str
    provenance: Optional[Provenance] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None


class IngestRequest(BaseModel):
    documents: List[IngestDoc]


class RetrieveRequest(BaseModel):
    query: str
    k: int = 12
    time_hint: Optional[Dict[str, Any]] = None


class AnswerRequest(BaseModel):
    query: str
    k: int = 8
    time_hint: Optional[Dict[str, Any]] = None
    strict_time: bool = False


@app.post("/ingest")
def ingest(req: IngestRequest):
    _ensure_ready()
    docs = [d.model_dump() for d in req.documents]
    return RAG.ingest(docs)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    _ensure_ready()
    hint = req.time_hint or parse_time_hint(req.query)
    return RAG.retrieve(req.query, k=req.k, time_hint=hint)


@app.post("/answer")
def answer(req: AnswerRequest):
    _ensure_ready()
    hint = req.time_hint or parse_time_hint(req.query)
    return RAG.answer(req.query, k=req.k, time_hint=hint, strict_time=req.strict_time)


@app.get("/graph/summary")
def graph_summary():
    _ensure_ready()
    return RAG.graph.summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
