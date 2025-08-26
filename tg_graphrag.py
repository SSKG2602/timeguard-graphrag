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
    import faiss  # optional high-performance vector search
except Exception:  # pragma: no cover - faiss may not be available on all platforms
    faiss = None
import torch
import spacy
from tqdm import tqdm

from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from transformers import AutoTokenizer, AutoModelForCausalLM

from timeguard import parse_time_hint, time_compat, hard_pass
from graph_store import GraphStore

IDX_DIR = "indexes"
MAP_PATH = os.path.join(IDX_DIR, "chunks.jsonl")
FAISS_PATH = os.path.join(IDX_DIR, "faiss.index")
NPY_PATH = os.path.join(IDX_DIR, "vectors.npy")

os.makedirs(IDX_DIR, exist_ok=True)


def _hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_uri: Optional[str] = None
    observed_at: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    entities: List[str] = field(default_factory=list)


class GraphRAG:
    def __init__(self, model_name: Optional[str] = None):
        # Embeddings
        self.embedder = TextEmbedding()
        self.dim = self._infer_embed_dim()
        self.chunks: List[Chunk] = []
        self.lock = threading.Lock()
        self.vecs: List[np.ndarray] = []  # fallback vectors if faiss unavailable
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = None

        # Reranker
        self.reranker = Ranker()

        # spaCy NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")
            if "ner" not in self.nlp.pipe_names:
                self.nlp.add_pipe("ner")

        # Graph layer
        self.graph = GraphStore()

        # Load persisted index
        self._load()

        # Qwen model - keep original tech, auto-select size if not specified
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
        torch.set_num_threads(1)

    def _infer_embed_dim(self) -> int:
        # Handle all fastembed versions robustly
        try:
            # Some versions may actually have this:
            dim = getattr(
                self.embedder, "get_sentence_embedding_dimension", None)
            if callable(dim):
                return int(dim())
        except Exception:
            pass
    # Universal fallback: embed a probe string and read vector length
        try:
            vec = next(iter(self.embedder.embed(["__dim_probe__"])))
            return int(len(vec))
        except Exception as e:
            raise RuntimeError(
                f"Failed to infer embedding dimension from fastembed: {e}")

    # ---------------------- Model Selection ----------------------

    def _auto_select_qwen(self) -> str:
        env = os.getenv("QWEN_MODEL", "").strip()
        if env:
            return env
        # RAM/GPU probing
        try:
            avail_ram = int(psutil.virtual_memory().available / (1024**3))
        except Exception:
            avail_ram = 4
        has_cuda = torch.cuda.is_available()
        vram_gb = 0
        if has_cuda:
            try:
                vram_gb = torch.cuda.get_device_properties(
                    0).total_memory // (1024**3)
            except Exception:
                vram_gb = 0

        # Upgrade path first
        if has_cuda and vram_gb >= 16:
            return "Qwen/Qwen2.5-7B-Instruct"
        if avail_ram >= 20:
            return "Qwen/Qwen2.5-3B-Instruct"
        if avail_ram >= 8:
            return "Qwen/Qwen2.5-1.5B-Instruct"
        return "Qwen/Qwen2.5-0.5B-Instruct"

    def _pick_device_dtype(self):
        if torch.cuda.is_available():
            return "cuda", torch.float16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    # ---------------------- Persistence ----------------------
    def _load(self):
        if os.path.exists(MAP_PATH):
            with open(MAP_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    self.chunks.append(Chunk(**json.loads(line)))
        if faiss is not None and os.path.exists(FAISS_PATH):
            self.index = faiss.read_index(FAISS_PATH)
        elif os.path.exists(NPY_PATH):
            try:
                self.vecs = np.load(NPY_PATH).tolist()
            except Exception:
                self.vecs = []

    def _save(self):
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, FAISS_PATH)
        else:
            np.save(NPY_PATH, np.array(self.vecs, dtype="float32"))
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.__dict__) + "\n")

    # ---------------------- Ingest ----------------------
    def _extract_entities(self, text: str) -> List[str]:
        ents = []
        try:
            doc = self.nlp(text)
            for e in doc.ents:
                if e.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"):
                    ents.append(e.text.strip())
        except Exception:
            pass
        # de-dup small list
        seen = set()
        out = []
        for x in ents:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out[:12]

    def ingest(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts, metas, ents_map = [], [], {}
        for d in docs:
            text = d["text"]
            ext_id = d.get("external_id", "doc")
            max_chars = 3200
            for i in range(0, len(text), max_chars):
                t = text[i:i+max_chars].strip()
                if not t:
                    continue
                cid = f"chunk:{_hash(ext_id+'|'+str(i)+'|'+_hash(t))[:12]}"
                entities = self._extract_entities(t)
                meta = {
                    "chunk_id": cid, "text": t,
                    "source_uri": d.get("provenance", {}).get("uri"),
                    "observed_at": d.get("provenance", {}).get("observed_at"),
                    "valid_from": d.get("valid_from"),
                    "valid_to": d.get("valid_to"),
                    "entities": entities
                }
                texts.append(t)
                metas.append(meta)
                ents_map[cid] = entities

        # embed
        vectors = np.array(
            [v for v in self.embedder.embed(texts)], dtype="float32")
        if self.index is not None and faiss is not None:
            faiss.normalize_L2(vectors)
        else:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            vectors = vectors / norms

        with self.lock:
            if self.index is not None and faiss is not None:
                self.index.add(vectors)
            else:
                self.vecs.extend(vectors)
            for m in metas:
                ch = Chunk(**m)
                self.chunks.append(ch)
                # add to graph
                self.graph.add_claim(ch.chunk_id, ch.source_uri, ch.entities)
            self._save()

        return {"ingested": len(metas), "entities_added": sum(len(v) for v in ents_map.values())}

    # ---------------------- Retrieval / Multi-hop ----------------------
    def _initial_retrieve(self, query: str, k: int, time_hint: Dict[str, Any]):
        # embed query
        qvec = np.array(
            [v for v in self.embedder.embed([query])], dtype="float32")
        if self.index is not None and faiss is not None:
            faiss.normalize_L2(qvec)
            scores, idxs = self.index.search(qvec, max(k*3, k))
            idxs = idxs[0]
            scs = scores[0]
        else:
            qvec /= (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
            if not self.vecs:
                return [], []
            mat = np.array(self.vecs)
            scs = mat @ qvec[0]
            idxs = np.argsort(-scs)[:max(k*3, k)]
            scs = scs[idxs]
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
        # FR rerank
        req = RerankRequest(query=query, passages=[
                            {"id": b["chunk_id"], "text": b["text"]} for b in base])
        rer = self.reranker.rerank(req)
        rmap = {x["id"]: x["relevance_score"] for x in rer}
        for b in base:
            w_t = time_compat(b["valid_from"], b["valid_to"], time_hint)
            b["score_rerank"] = float(rmap.get(b["chunk_id"], 0.0))
            b["score_time"] = float(w_t)
            b["score_final"] = 0.34*b["score_retrieval"] + \
                0.46*b["score_rerank"] + 0.20*b["score_time"]
        base.sort(key=lambda x: x["score_final"], reverse=True)
        return base[:k], base

    def _should_escalate(self, topk: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not topk:
            return {"escalate": False, "reason": "no_results"}
        # signals: low median score_final, time conflict among evidence
        finals = [x["score_final"] for x in topk]
        med = sorted(finals)[len(finals)//2]
        time_ranges = [(x.get("valid_from"), x.get("valid_to"))
                       for x in topk[:6]]
        # conflict if at least two non-overlapping windows exist

        def overlap(a, b):
            from timeguard import _to_date_safe
            af, at = _to_date_safe(a[0]), _to_date_safe(a[1])
            bf, bt = _to_date_safe(b[0]), _to_date_safe(b[1])
            if not af and not at:
                return True
            if not bf and not bt:
                return True
            A = (af or bf, at or bt)
            B = (bf or af, bt or at)
            return not (A[1] and B[0] and A[1] < B[0] or B[1] and A[0] and B[1] < A[0])
        conflict = False
        for i in range(len(time_ranges)):
            for j in range(i+1, len(time_ranges)):
                if not overlap(time_ranges[i], time_ranges[j]):
                    conflict = True
        return {"escalate": (med < 0.25) or conflict, "reason": "low_score" if med < 0.25 else ("conflict" if conflict else "ok")}

    def _second_hop(self, query: str, base_all: List[Dict[str, Any]], time_hint: Dict[str, Any], k: int):
        # gather top entities and build expanded queries
        chunk_entities = {b["chunk_id"]: b.get(
            "entities", []) for b in base_all[:12]}
        topents = self.graph.top_entities_from_chunks(chunk_entities, top_n=5)
        expansions = [f"{query} {e}" for e in topents]
        bag = []
        for q2 in expansions:
            qvec = np.array(
                [v for v in self.embedder.embed([q2])], dtype="float32")
            if self.index is not None and faiss is not None:
                faiss.normalize_L2(qvec)
                scores, idxs = self.index.search(qvec, k)
                iter_pairs = zip(idxs[0], scores[0])
            else:
                qvec /= (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
                if not self.vecs:
                    iter_pairs = []
                else:
                    mat = np.array(self.vecs)
                    scs = mat @ qvec[0]
                    idxs = np.argsort(-scs)[:k]
                    iter_pairs = zip(idxs, scs[idxs])
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
        # merge + rerank
        dedup = {}
        for b in bag:
            dedup[b["chunk_id"]] = b
        bag = list(dedup.values())
        req = RerankRequest(query=query, passages=[
                            {"id": b["chunk_id"], "text": b["text"]} for b in bag])
        rer = self.reranker.rerank(req)
        rmap = {x["id"]: x["relevance_score"] for x in rer}
        for b in bag:
            w_t = time_compat(b["valid_from"], b["valid_to"], time_hint)
            b["score_rerank"] = float(rmap.get(b["chunk_id"], 0.0))
            b["score_time"] = float(w_t)
            b["score_final"] = 0.28*b["score_retrieval"] + \
                0.52*b["score_rerank"] + 0.20*b["score_time"]
        bag.sort(key=lambda x: x["score_final"], reverse=True)
        return bag[:k]

    def retrieve(self, query: str, k: int = 12, time_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        time_hint = time_hint or parse_time_hint(query)
        topk, base_all = self._initial_retrieve(query, k, time_hint)
        gate = self._should_escalate(topk)
        hops_used = 1
        if gate["escalate"]:
            topk2 = self._second_hop(query, base_all, time_hint, k)
            # blend with first
            blend = {x["chunk_id"]: x for x in topk}
            for b in topk2:
                if b["chunk_id"] not in blend:
                    blend[b["chunk_id"]] = b
                else:
                    # keep the better one
                    if b["score_final"] > blend[b["chunk_id"]]["score_final"]:
                        blend[b["chunk_id"]] = b
            topk = sorted(blend.values(),
                          key=lambda x: x["score_final"], reverse=True)[:k]
            hops_used = 2
        return {"passages": topk, "time_filters": time_hint, "controller": {"hops_used": hops_used, "reason": gate["reason"]}}

    # ---------------------- Answering ----------------------
    def answer(self, query: str, k: int = 8, time_hint: Optional[Dict[str, Any]] = None, strict_time: bool = False) -> Dict[str, Any]:
        ev = self.retrieve(query, k=max(k, 12), time_hint=time_hint)
        hint = ev["time_filters"]
        passages = ev["passages"]

        context, evidence = [], []
        for p in passages:
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

        # minimal conflict note (non-overlap in top-4)
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
                    if at and bf and at < bf or bt and af and bt < af:
                        conflicts.append((a["chunk_id"], b["chunk_id"]))

        sys_msg = (
            "You are TimeGuard Graph-RAG.\n"
            "Use passages below. Respect temporal intent. Avoid stale claims.\n"
            "If evidence conflicts, present both and explain the time difference.\n"
            "Cite as [#chunk_id]."
        )
        prompt = (
            f"<|im_start|>system\n{sys_msg}\n<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {query}\n\nPassages:\n" +
            "\n\n".join(f"[{i+1} #{e['chunk_id']}] {c}" for i, (c, e) in enumerate(zip(context, evidence))) +
            "\n\nAnswer clearly, with 1–2 short paragraphs and citations.\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

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
