# TimeGuard Graph-RAG 🕒

A  **temporal-aware Graph-RAG (Retrieval-Augmented Generation)** system that understands time-sensitive queries and provides accurate, contextual answers.

Author: **SSKG** (Shreyas Shashi Kumar Gowda)

---

## 🎯 Key Capabilities

- **Time-Aware Retrieval**: Handles queries like *“CEO as of today”*, *“Q4 FY23 revenue”*, *“between 2019 and 2021”*.
- **Graph-Enhanced RAG**: Uses entity relationships for multi-hop reasoning.
- **Flexible Model Support**: Optimized for Qwen family but easily switchable to any LLM.
- **Deployment Ready**: Works locally, on Colab, or cloud (Render).

---

## 🏗️ System Overview

```
Documents ──▶ TimeGuard Graph-RAG ──▶ Answers (time-aware)
                 │
                 ▼
         ┌─────────────────────┐
         │ Core Components:    │
         │  • Chunking         │
         │  • Entity Extraction│
         │  • Vector Store     │
         │  • Graph Store      │
         │  • Time Parser      │
         │  • Multi-hop Search │
         │  • LLM Generation   │
         └─────────────────────┘
```

---

## 📁 Project Structure

```
timeguard-graphrag/
├── app.py          # Streamlit UI frontend
├── server.py       # FastAPI backend APIs
├── run.py          # Zero-dependency bootstrapper
├── tg_graphrag.py  # Core Graph-RAG engine
├── timeguard.py    # Time intelligence parser
├── graph_store.py  # Graph operations (NetworkX + Neo4j)
├── requirements.txt# Dependencies
├── Procfile        # Heroku deployment config
├── render.yaml     # Render deployment config
└── .gitignore      # Exclude models, logs, data
```

### File Responsibilities

- ``: Main engine — embeddings, retrieval, generation, FAISS integration, multi-hop reasoning, temporal filters.
- ``: Temporal parser — fiscal years, relative dates, explicit ranges, before/after constraints.
- ``: Knowledge graph — entity types, relationships, optional Neo4j persistence.
- ``: REST API endpoints with health checks.
- ``: Streamlit frontend for interactive Q&A.
- ``: Automates environment setup, backend startup, and frontend launch.

---

## 🚀 Quick Start

### Method 1: One-Click Bootstrap (Recommended)

```bash
git clone https://github.com/SSKG2602/timeguard-graphrag.git
cd timeguard-graphrag
python run.py
```

`run.py` will:

1. Ensure Python 3.10+
2. Create `.venv` virtual environment
3. Install requirements & spaCy model
4. Launch FastAPI backend (port 8000)
5. Launch Streamlit frontend (port 8501)

### Method 2: Manual Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python server.py &
streamlit run app.py --server.port 8501
```

---

## ⚙️ Configuration

Environment variables (via `.env`):

```bash
TIMEZONE=Asia/Kolkata
DAYFIRST=1
FISCAL_YEAR_START_MONTH=4
FISCAL_YEAR_START_DAY=1
QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

---

## 🌐 Deployment

- **Render**: Configured in `render.yaml`
- **Heroku**: Uses `Procfile`
- **Colab**: Example notebook provided for testing
- **Docker**: Supported with custom Dockerfile

---

## 📊 Example Usage

### Ingest Documents

```bash
POST http://localhost:8000/ingest
{
  "documents": [{
    "external_id": "q4_2023",
    "text": "Q4 2023 Results: Revenue $156M, new CEO Sarah Johnson Dec 1, 2023.",
    "valid_from": "2023-10-01",
    "valid_to": "2023-12-31"
  }]
}
```

### Query with Time Awareness

```bash
POST http://localhost:8000/answer
{
  "query": "Who is the CEO as of today?",
  "k": 8
}
```

Response:

```json
{
  "answer": "Sarah Johnson is the CEO as of today.",
  "time": {"operator": "AS_OF", "at": "2025-08-28"}
}
```

---

## 🧠 Advanced Features

- Intelligent vs Hard Temporal Filters【50†TGfinalDeploying.pdf】
- Conflict detection (e.g., multiple CEOs overlapping)
- Multi-hop retrieval only when needed【47†Graph Rag detailed (text‑only).pdf】
- Self-learning loop: converts incidents into updated policies

---

## 📈 Performance & Scaling

- Vector search powered by **FAISS** (with NumPy fallback).
- Automatic Qwen model selection based on system RAM/VRAM.
- Memory-aware batch processing.
- Supports caching for repeated queries.

---

## 🔒 Security & Compliance

- PII scrubbing at ingest
- RBAC roles for APIs
- Audit trails for policy changes

---

## 👤 Author

Developed and maintained by **SSKG**\
*(Shreyas Shashi Kumar Gowda)*

