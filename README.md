# ⏳ TimeGuard Graph-RAG 🕒  
A temporal-aware **Graph-RAG (Retrieval-Augmented Generation)** system that understands **time-sensitive queries** and provides accurate, contextual answers.  

**Author:** [SSKG (Shreyas Shashi Kumar Gowda)](https://github.com/SSKG2602)  

---

## 📦 Features  

### Core Capabilities  
- ⏳ **Time-Aware Retrieval**  
  Handles queries like *“CEO as of today”*, *“Q4 FY23 revenue”*, *“between 2019 and 2021”*.  

- 🕸 **Graph-Enhanced RAG**  
  Uses entity relationships for multi-hop reasoning instead of blind keyword search.  

- 🔄 **Flexible Model Support**  
  Optimized for **Qwen family** but easily switchable to any LLM.  

- 🚀 **Deployment Ready**  
  Works locally, in Colab, or in the cloud (**Render, Heroku, Docker**).  

- 🧠 **Self-Learning Loop**  
  Converts repeat failures into updated policies automatically.  

---

## 🏗 System Overview  

**Flow:**  


```
Documents ──▶ TimeGuard Graph-RAG ──▶ Answers (time-aware)
│
▼
┌─────────────────────┐
│   Core Components:  │
│   • Chunking        │
│   • Entity Extraction│
│   • Vector Store     │
│   • Graph Store      │
│   • Time Parser      │
│   • Multi-hop Search │
│   • LLM Generation   │
└─────────────────────┘
```




---

## 📁 Project Structure  


```
timeguard-graphrag/
├── app.py             # Streamlit UI frontend
├── server.py          # FastAPI backend APIs
├── run.py             # Zero-dependency bootstrapper
├── tg_graphrag.py     # Core Graph-RAG engine
├── timeguard.py       # Time intelligence parser
├── graph_store.py     # Graph operations (NetworkX + Neo4j)
├── requirements.txt   # Dependencies
├── Procfile           # Heroku deployment config
├── render.yaml        # Render deployment config
└── .gitignore         # Exclude models, logs, data
```


---

## 🚀 Quick Start  

### Method 1: One-Click Bootstrap (Recommended)  

```bash
git clone https://github.com/SSKG2602/timeguard-graphrag.git
cd timeguard-graphrag
python run.py


What run.py does automatically:
Checks Python 3.10+
Creates .venv virtual environment
Installs dependencies + spaCy model
Launches FastAPI backend (port 8000)
Launches Streamlit frontend (port 8501)


⚙ Configuration
.env Example:

TIMEZONE=Asia/Kolkata
DAYFIRST=1
FISCAL_YEAR_START_MONTH=4
FISCAL_YEAR_START_DAY=1
QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct


Example Usage
1️⃣ Ingest Document

POST http://localhost:8000/ingest
{
  "documents": [{
    "external_id": "q4_2023",
    "text": "Q4 2023 Results: Revenue $156M, new CEO Sarah Johnson Dec 1, 2023.",
    "valid_from": "2023-10-01",
    "valid_to": "2023-12-31"
  }]
}


2️⃣ Ask Time-Aware Question

POST http://localhost:8000/answer
{
  "query": "Who is the CEO as of today?",
  "k": 8
}


Response:

{
  "answer": "Sarah Johnson is the CEO as of today.",
  "time": {"operator": "AS_OF", "at": "2025-08-28"}
}



Real-World Applications


Financial Analysis – Query company reports with fiscal-year alignment (e.g., “Revenue in Q2 FY25”).
Policy/Legal Compliance – Ensure regulations are applied as of the right time period.
Customer Support Bots – Provide version-sensitive answers tied to policy/document changes.
Historical Research – Explore leadership changes, events, or timelines with evidence.
Enterprise Knowledge Bases – Keep QA systems free of stale or outdated citations.



Performance & Scaling

Vector search with FAISS (NumPy fallback).
Auto-selects Qwen model by RAM/VRAM.
Batch ingestion + caching for repeat queries.


Author

Developed and maintained by SSKG (Shreyas Shashi Kumar Gowda)
https://www.linkedin.com/in/shreyasshashi/

