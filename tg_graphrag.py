# Core GraphRAG engine for TimeGuard system
# Combines vector search, graph reasoning, and time-aware retrieval with LLM generation

from __future__ import annotations
import os
import json
import hashlib
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

import psutil
try:
    import faiss  # High-performance vector search library
except Exception:  # Fallback to numpy if faiss unavailable
    faiss = None
import torch
import spacy
from tqdm import tqdm

# Fix for tqdm compatibility across versions
# Some libraries expect tqdm._lock attribute that was removed in newer versions
if not hasattr(tqdm, "_lock"):
    tqdm._lock = threading.RLock()

from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from transformers import AutoTokenizer, AutoModelForCausalLM

from timeguard import parse_time_hint, time_compat, hard_pass
from graph_store import GraphStore

# File paths for persistent storage
IDX_DIR = "indexes"
MAP_PATH = os.path.join(IDX_DIR, "chunks.jsonl")  # Chunk metadata storage
FAISS_PATH = os.path.join(IDX_DIR, "faiss.index")  # FAISS vector index
NPY_PATH = os.path.join(IDX_DIR, "vectors.npy")   # Numpy vector fallback

# Ensure index directory exists
os.makedirs(IDX_DIR, exist_ok=True)


def _hash(text: str) -> str:
    """Generate SHA256 hash of text for unique identification"""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    """
    Data structure representing a document chunk with temporal and entity information
    """
    chunk_id: str                           # Unique identifier
    text: str                              # Chunk content
    source_uri: Optional[str] = None       # Source document URI
    observed_at: Optional[str] = None      # When information was observed
    valid_from: Optional[str] = None       # Validity period start
    valid_to: Optional[str] = None         # Validity period end
    entities: List[str] = field(default_factory=list)  # Extracted entities


class GraphRAG:
    """
    Main GraphRAG system combining vector search, graph reasoning, and LLM generation
    Handles document ingestion, retrieval, and answer generation with temporal awareness
    """
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize embedding system
        self.embedder = TextEmbedding()
        self.dim = self._infer_embed_dim()
        self.chunks: List[Chunk] = []
        self.lock = threading.Lock()  # Thread safety for concurrent access
        
        # Initialize vector storage (FAISS preferred, numpy fallback)
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)  # Inner product index for cosine similarity
            self.vecs = None
        else:
            self.index = None
            # Single array storage for efficient similarity computation
            self.vecs = np.empty((0, self.dim), dtype="float32")

        # Initialize reranking system for relevance refinement
        self.reranker = Ranker()

        # Initialize NER system for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback to blank model with NER pipeline
            self.nlp = spacy.blank("en")
            if "ner" not in self.nlp.pipe_names:
                self.nlp.add_pipe("ner")

        # Initialize graph layer for entity relationships
        self.graph = GraphStore()

        # Load any existing persisted data
        self._load()

        # Initialize language model with automatic selection
        self.model_name = model_name or self._auto_select_qwen()
        self.device, self.dtype = self._pick_device_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        # Single thread for CPU inference to avoid oversubscription
        torch.set_num_threads(1)

    def _infer_embed_dim(self) -> int:
        """
        Determine embedding dimension by probing the embedding model
        Handles different fastembed API versions gracefully
        """
        # Try direct dimension method if available
        try:
            dim = getattr(
                self.embedder, "get_sentence_embedding_dimension", None)
            if callable(dim):
                return int(dim())
        except Exception:
            pass
        
        # Universal fallback: embed probe string and measure vector length
        try:
            vec = next(iter(self.embedder.embed(["__dim_probe__"])))
            return int(len(vec))
        except Exception as e:
            raise RuntimeError(
                f"Failed to infer embedding dimension from fastembed: {e}")

    # ---------------------- Model Selection ----------------------

    def _auto_select_qwen(self) -> str:
        """
        Automatically select appropriate Qwen model based on available system resources
        Considers both RAM and VRAM to choose optimal model size
        """
        # Check for explicit model specification in environment
        env = os.getenv("QWEN_MODEL", "").strip()
        if env:
            return env
        
        # Assess available system memory
        try:
            avail_ram = int(psutil.virtual_memory().available / (1024**3))
        except Exception:
            avail_ram = 4  # Conservative fallback
        
        # Check GPU availability and memory
        has_cuda = torch.cuda.is_available()
        vram_gb = 0
        if has_cuda:
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            except Exception:
                vram_gb = 0

        # Select model based on available resources
        # Prefer larger models only when resources are ample
        if has_cuda and vram_gb >= 16:
            return "Qwen/Qwen2.5-7B-Instruct"      # Large model for high-end GPUs
        if avail_ram >= 20:
            return "Qwen/Qwen2.5-3B-Instruct"      # Medium model for high RAM
        if avail_ram >= 8:
            return "Qwen/Qwen2.5-1.5B-Instruct"    # Small model for moderate RAM
        return "Qwen/Qwen2.5-0.5B-Instruct"        # Minimal model for low resources

    def _pick_device_dtype(self):
        """
        Select optimal device and data type based on available hardware
        Prioritizes GPU with mixed precision, falls back to CPU
        """
        if torch.cuda.is_available():
            return "cuda", torch.float16      # CUDA with half precision
        if torch.backends.mps.is_available():
            return "mps", torch.float16       # Apple Metal Performance Shaders
        return "cpu", torch.float32           # CPU fallback with full precision

    # ---------------------- Persistence ----------------------
    
    def _load(self):
        """
        Load persisted chunks and vector index from disk
        Handles both FAISS and numpy storage formats
        """
        # Load chunk metadata from JSONL file
        if os.path.exists(MAP_PATH):
            with open(MAP_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    self.chunks.append(Chunk(**json.loads(line)))
        
        # Load vector index (FAISS preferred)
        if faiss is not None and os.path.exists(FAISS_PATH):
            self.index = faiss.read_index(FAISS_PATH)
        elif self.index is None and os.path.exists(NPY_PATH):
            # Load numpy vectors as fallback
            try:
                self.vecs = np.load(NPY_PATH)
            except Exception:
                self.vecs = np.empty((0, self.dim), dtype="float32")

    def _save(self):
        """
        Persist current state to disk
        Saves both vector index and chunk metadata
        """
        # Save vector index
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, FAISS_PATH)
        elif self.vecs is not None:
            np.save(NPY_PATH, self.vecs)
        
        # Save chunk metadata as JSONL
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.__dict__) + "\n")

    # ---------------------- Ingest ----------------------
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text using spaCy NER
        Focuses on entity types relevant to business/temporal queries
        """
        ents = []
        try:
            doc = self.nlp(text)
            # Extract relevant entity types
            for e in doc.ents:
                if e.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"):
                    ents.append(e.text.strip())
        except Exception:
            pass
        
        # Deduplicate while preserving order
        seen = set()
        out = []
        for x in ents:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out[:12]  # Limit to top 12 entities per chunk

    def ingest(self, docs: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system
        Processes in batches to manage memory usage during embedding
        
        Args:
            docs: List of document dictionaries with text and metadata
            batch_size: Number of chunks to process at once
        
        Returns:
            Statistics about ingestion process
        """
        batch_texts: List[str] = []
        batch_meta: List[Dict[str, Any]] = []
        ingested = 0
        ents_added = 0

        def _process_batch():
            """Process current batch of texts through embedding and storage"""
            nonlocal batch_texts, batch_meta, ingested
            if not batch_texts:
                return
            
            # Generate embeddings for batch
            vectors = np.array([v for v in self.embedder.embed(batch_texts)], dtype="float32")
            
            with self.lock:
                # Add to vector index
                if self.index is not None and faiss is not None:
                    faiss.normalize_L2(vectors)  # Normalize for cosine similarity
                    self.index.add(vectors)
                else:
                    # Normalize and store in numpy array
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
                    vectors = vectors / norms
                    self.vecs = np.vstack([self.vecs, vectors]) if self.vecs.size else vectors
                
                # Store chunk metadata and update graph
                for m in batch_meta:
                    ch = Chunk(**m)
                    self.chunks.append(ch)
                    self.graph.add_claim(ch.chunk_id, ch.source_uri, ch.entities)
                    
            ingested += len(batch_meta)
            batch_texts, batch_meta = [], []

        # Process each document
        for d in docs:
            text = d["text"]
            ext_id = d.get("external_id", "doc")
            max_chars = 3200  # Chunk size for manageable processing
            
                            # Split document into chunks
            for i in range(0, len(text), max_chars):
                t = text[i:i+max_chars].strip()
                if not t:
                    continue
                
                # Generate unique chunk ID based on content and position
                cid = f"chunk:{_hash(ext_id+'|'+str(i)+'|'+_hash(t))[:12]}"
                entities = self._extract_entities(t)
                
                # Prepare chunk metadata
                meta = {
                    "chunk_id": cid,
                    "text": t,
                    "source_uri": d.get("provenance", {}).get("uri"),
                    "observed_at": d.get("provenance", {}).get("observed_at"),
                    "valid_from": d.get("valid_from"),
                    "valid_to": d.get("valid_to"),
                    "entities": entities,
                }
                ents_added += len(entities)
                
                # Add to current batch
                batch_texts.append(t)
                batch_meta.append(meta)
                
                # Process batch when full
                if len(batch_texts) >= batch_size:
                    _process_batch()

        # Process final batch
        _process_batch()

        # Persist all changes to disk
        with self.lock:
            self._save()

        return {"ingested": ingested, "entities_added": ents_added}

    # ---------------------- Retrieval / Multi-hop ----------------------
    
    def _initial_retrieve(self, query: str, k: int, time_hint: Dict[str, Any]):
        """
        First hop retrieval using vector similarity search
        Returns both top-k results and larger candidate set for potential escalation
        """
        # Generate query embedding
        qvec = np.array(
            [v for v in self.embedder.embed([query])], dtype="float32")
        
        # Search vector index
        if self.index is not None and faiss is not None:
            faiss.normalize_L2(qvec)
            scores, idxs = self.index.search(qvec, max(k*3, k))
            idxs = idxs[0]
            scs = scores[0]
        else:
            # Numpy fallback search
            qvec /= (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
            if self.vecs.size == 0:
                return [], []
            scs = self.vecs @ qvec[0]  # Cosine similarity
            idxs = np.argsort(-scs)[:max(k*3, k)]
            scs = scs[idxs]
        
        # Convert indices to chunk data
        base = []
        for i, s in zip(idxs, scs):
            if i < 0:
                continue
            ch = self.chunks[i]
            base.append({
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "score_retrieval": float(s),
                "source": {"uri": ch.source_uri},
                "valid_from": ch.valid_from,
                "valid_to": ch.valid_to,
                "observed_at": ch.observed_at,
                "entities": ch.entities
            })
        
        # Apply reranking for relevance refinement
        req = RerankRequest(query=query, passages=[
                            {"id": b["chunk_id"], "text": b["text"]} for b in base])
        rer = self.reranker.rerank(req)
        rmap = {x["id"]: x.get("relevance_score", x.get("score", 0.0))
                for x in rer}
        
        # Calculate final scores combining retrieval, reranking, and temporal compatibility
        for b in base:
            w_t = time_compat(b["valid_from"], b["valid_to"], time_hint)
            b["score_rerank"] = float(rmap.get(b["chunk_id"], 0.0))
            b["score_time"] = float(w_t)
            # Weighted combination of all scores
            b["score_final"] = 0.34*b["score_retrieval"] + \
                0.46*b["score_rerank"] + 0.20*b["score_time"]
        
        base.sort(key=lambda x: x["score_final"], reverse=True)
        return base[:k], base

    def _should_escalate(self, topk: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine if retrieval should escalate to multi-hop search
        Escalates on low confidence or temporal conflicts in evidence
        """
        if not topk:
            return {"escalate": False, "reason": "no_results"}
        
        # Check median score quality
        finals = [x["score_final"] for x in topk]
        med = sorted(finals)[len(finals)//2]
        
        # Check for temporal conflicts among top evidence
        time_ranges = [(x.get("valid_from"), x.get("valid_to"))
                       for x in topk[:6]]
        
        def overlap(a, b):
            """Check if two time ranges overlap"""
            from timeguard import _to_date_safe
            af, at = _to_date_safe(a[0]), _to_date_safe(a[1])
            bf, bt = _to_date_safe(b[0]), _to_date_safe(b[1])
            if not af and not at:
                return True  # No time constraint
            if not bf and not bt:
                return True
            A = (af or bf, at or bt)
            B = (bf or af, bt or at)
            return not (A[1] and B[0] and A[1] < B[0] or B[1] and A[0] and B[1] < A[0])
        
        # Detect temporal conflicts
        conflict = False
        for i in range(len(time_ranges)):
            for j in range(i+1, len(time_ranges)):
                if not overlap(time_ranges[i], time_ranges[j]):
                    conflict = True
        
        # Escalate if scores are low or conflicts exist
        return {"escalate": (med < 0.25) or conflict, 
                "reason": "low_score" if med < 0.25 else ("conflict" if conflict else "ok")}

    def _second_hop(self, query: str, base_all: List[Dict[str, Any]], time_hint: Dict[str, Any], k: int):
        """
        Second hop retrieval using entity-expanded queries
        Leverages graph structure for multi-hop reasoning
        """
        # Extract top entities from initial results
        chunk_entities = {b["chunk_id"]: b.get(
            "entities", []) for b in base_all[:12]}
        topents = self.graph.top_entities_from_chunks(chunk_entities, top_n=5)
        
        # Create entity-expanded query variations
        expansions = [f"{query} {e}" for e in topents]
        bag = []
        
        # Search with each expanded query
        for q2 in expansions:
            qvec = np.array(
                [v for v in self.embedder.embed([q2])], dtype="float32")
            
            # Vector search for expanded query
            if self.index is not None and faiss is not None:
                faiss.normalize_L2(qvec)
                scores, idxs = self.index.search(qvec, k)
                iter_pairs = zip(idxs[0], scores[0])
            else:
                qvec /= (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
                if self.vecs.size == 0:
                    iter_pairs = []
                else:
                    scs = self.vecs @ qvec[0]
                    idxs = np.argsort(-scs)[:k]
                    iter_pairs = zip(idxs, scs[idxs])
            
            # Collect results from expanded search
            for i, s in iter_pairs:
                if i < 0:
                    continue
                ch = self.chunks[i]
                bag.append({
                    "chunk_id": ch.chunk_id, "text": ch.text,
                    "score_retrieval": float(s),
                    "source": {"uri": ch.source_uri},
                    "valid_from": ch.valid_from, "valid_to": ch.valid_to,
                    "observed_at": ch.observed_at, "entities": ch.entities
                })
        
        # Deduplicate results by chunk ID
        dedup = {}
        for b in bag:
            dedup[b["chunk_id"]] = b
        bag = list(dedup.values())
        
        # Rerank expanded results against original query
        req = RerankRequest(query=query, passages=[
                            {"id": b["chunk_id"], "text": b["text"]} for b in bag])
        rer = self.reranker.rerank(req)
        rmap = {x["id"]: x.get("relevance_score", x.get("score", 0.0))
                for x in rer}
        
        # Calculate final scores with adjusted weights for second hop
        for b in bag:
            w_t = time_compat(b["valid_from"], b["valid_to"], time_hint)
            b["score_rerank"] = float(rmap.get(b["chunk_id"], 0.0))
            b["score_time"] = float(w_t)
            # Higher weight on reranking for second hop
            b["score_final"] = 0.28*b["score_retrieval"] + \
                0.52*b["score_rerank"] + 0.20*b["score_time"]
        
        bag.sort(key=lambda x: x["score_final"], reverse=True)
        return bag[:k]

    def retrieve(self, query: str, k: int = 12, time_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main retrieval method with adaptive multi-hop capability
        Uses escalation logic to determine when additional hops are needed
        """
        time_hint = time_hint or parse_time_hint(query)
        
        # First hop retrieval
        topk, base_all = self._initial_retrieve(query, k, time_hint)
        
        # Decide whether to escalate to second hop
        gate = self._should_escalate(topk)
        hops_used = 1
        
        if gate["escalate"]:
            # Perform second hop search
            topk2 = self._second_hop(query, base_all, time_hint, k)
            
            # Blend first and second hop results
            blend = {x["chunk_id"]: x for x in topk}
            for b in topk2:
                if b["chunk_id"] not in blend:
                    blend[b["chunk_id"]] = b
                else:
                    # Keep result with higher score
                    if b["score_final"] > blend[b["chunk_id"]]["score_final"]:
                        blend[b["chunk_id"]] = b
            
            # Re-rank blended results
            topk = sorted(blend.values(),
                          key=lambda x: x["score_final"], reverse=True)[:k]
            hops_used = 2
        
        return {"passages": topk, "time_filters": time_hint, 
                "controller": {"hops_used": hops_used, "reason": gate["reason"]}}

    # ---------------------- Answering ----------------------
    
    def answer(self, query: str, k: int = 8, time_hint: Optional[Dict[str, Any]] = None, strict_time: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive answers using retrieved evidence and language model
        Applies temporal filtering and conflict detection for reliable responses
        """
        # Retrieve relevant evidence
        ev = self.retrieve(query, k=max(k, 12), time_hint=time_hint)
        hint = ev["time_filters"]
        passages = ev["passages"]

        # Filter and prepare evidence for answer generation
        context, evidence = [], []
        for p in passages:
            # Apply strict temporal filtering if requested
            if (not strict_time) or hard_pass(p["valid_from"], p["valid_to"], hint):
                context.append(p["text"])
                evidence.append({
                    "chunk_id": p["chunk_id"],
                    "uri": (p["source"] or {}).get("uri"),
                    "quote": (p["text"][:240]+"...") if len(p["text"]) > 240 else p["text"],
                    "valid_from": p.get("valid_from"),
                    "valid_to": p.get("valid_to"),
                    "score": round(p["score_final"], 4)
                })
            if len(context) >= k:
                break

        # Detect temporal conflicts in evidence
        conflicts = []
        for i in range(min(4, len(evidence))):
            for j in range(i+1, min(4, len(evidence))):
                a, b = evidence[i], evidence[j]
                if a["valid_from"] and a["valid_to"] and b["valid_from"] and b["valid_to"]:
                    from timeguard import _to_date_safe
                    af, at = _to_date_safe(
                        a["valid_from"]), _to_date_safe(a["valid_to"])
                    bf, bt = _to_date_safe(
                        b["valid_from"]), _to_date_safe(b["valid_to"])
                    # Check for non-overlapping periods
                    if at and bf and at < bf or bt and af and bt < af:
                        conflicts.append((a["chunk_id"], b["chunk_id"]))

        # Prepare system message for LLM
        sys_msg = (
            "You are TimeGuard Graph-RAG.\n"
            "Use passages below. Respect temporal intent. Avoid stale claims.\n"
            "If evidence conflicts, present both and explain the time difference.\n"
            "Cite as [#chunk_id]."
        )
        
        # Construct prompt with context
        prompt = (
            f"<|im_start|>system\n{sys_msg}\n<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {query}\n\nPassages:\n" +
            "\n\n".join(f"[{i+1} #{e['chunk_id']}] {c}" for i, (c, e) in enumerate(zip(context, evidence))) +
            "\n\nAnswer clearly, with 1–2 short paragraphs and citations.\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Generate answer using language model
        device = self.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=420, do_sample=False,
                temperature=0.2, eos_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        answer = text.split("<|im_start|>assistant")[-1].strip()

        return {
            "answer": answer,
            "attribution_card": {
                "evidence": evidence,
                "time": ev["time_filters"],
                "conflicts": [{"a": a, "b": b} for a, b in conflicts]
            },
            "controller_stats": ev["controller"]
        }
