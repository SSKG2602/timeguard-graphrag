# FastAPI backend server for TimeGuard Graph-RAG system
# Provides REST API endpoints for document ingestion and time-aware querying

from __future__ import annotations
import os
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from tg_graphrag import GraphRAG
from timeguard import parse_time_hint

# Initialize FastAPI application
app = FastAPI(title="TimeGuard Graph-RAG API", version="1.0")

# Global variables for managing RAG system state
RAG = None        # GraphRAG instance - created in background thread
STATUS = "starting"  # Current initialization status
ERR = None        # Error message if initialization fails


def _init_rag():
    """
    Initialize the GraphRAG system in background thread
    Prevents API startup blocking while models are loading
    """
    global RAG, STATUS, ERR
    try:
        # Create GraphRAG instance with configured or auto-selected model
        RAG = GraphRAG(model_name=os.getenv("QWEN_MODEL") or None)
        STATUS = "ready"
    except Exception as e:
        ERR = str(e)
        STATUS = "error"


# Start RAG initialization in background thread
# API becomes available immediately while models load
threading.Thread(target=_init_rag, daemon=True).start()


@app.get("/healthz")
def healthz():
    """
    Health check endpoint
    Returns current system status and configuration
    """
    return {"status": STATUS, "error": ERR, "model": os.getenv("QWEN_MODEL") or ""}


def _ensure_ready():
    """
    Ensure RAG system is fully initialized before processing requests
    Raises appropriate HTTP exceptions if system is not ready
    """
    if STATUS == "ready" and RAG is not None:
        return
    if STATUS == "error":
        raise HTTPException(status_code=500, detail={
                            "status": STATUS, "error": ERR})
    # Still starting - service temporarily unavailable
    raise HTTPException(status_code=503, detail={"status": STATUS})


# Pydantic models for request/response validation

class Provenance(BaseModel):
    """Document source and metadata information"""
    uri: Optional[str] = None
    observed_at: Optional[str] = None
    source_type: Optional[str] = None


class IngestDoc(BaseModel):
    """Document structure for ingestion"""
    external_id: str  # Unique identifier for the document
    text: str         # Document content
    provenance: Optional[Provenance] = None  # Source metadata
    valid_from: Optional[str] = None         # Document validity start date
    valid_to: Optional[str] = None           # Document validity end date


class IngestRequest(BaseModel):
    """Batch document ingestion request"""
    documents: List[IngestDoc]


class RetrieveRequest(BaseModel):
    """Document retrieval request"""
    query: str                           # Search query
    k: int = 12                         # Number of results to return
    time_hint: Optional[Dict[str, Any]] = None  # Optional time constraint


class AnswerRequest(BaseModel):
    """Question answering request"""
    query: str                           # User question
    k: int = 8                          # Number of evidence chunks to use
    time_hint: Optional[Dict[str, Any]] = None  # Optional time constraint
    strict_time: bool = False           # Whether to apply hard time filtering


@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    Ingest documents into the RAG system
    Processes text, extracts entities, creates embeddings, and builds graph
    """
    _ensure_ready()
    # Convert Pydantic models to dictionaries for processing
    docs = [d.model_dump() for d in req.documents]
    return RAG.ingest(docs)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """
    Retrieve relevant documents for a query
    Uses vector search and graph-based multi-hop reasoning
    """
    _ensure_ready()
    # Parse time constraints from query if not explicitly provided
    hint = req.time_hint or parse_time_hint(req.query)
    return RAG.retrieve(req.query, k=req.k, time_hint=hint)


@app.post("/answer")
def answer(req: AnswerRequest):
    """
    Generate answers to time-aware questions
    Combines retrieval with language model generation
    """
    _ensure_ready()
    # Parse time constraints from query if not explicitly provided
    hint = req.time_hint or parse_time_hint(req.query)
    return RAG.answer(req.query, k=req.k, time_hint=hint, strict_time=req.strict_time)


@app.get("/graph/summary")
def graph_summary():
    """
    Get summary statistics of the knowledge graph
    Returns node/edge counts and entity information
    """
    _ensure_ready()
    return RAG.graph.summary()


# Run server directly if script is executed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
