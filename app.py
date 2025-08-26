import os
import json
import requests
import streamlit as st
import time

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="TimeGuard Graph-RAG (Qwen)", layout="wide")
st.title("🕒 TimeGuard Graph-RAG — Qwen")


def backend_status():
    try:
        r = requests.get(f"{BACKEND}/healthz", timeout=3)
        return r.json()
    except Exception as e:
        return {"status": "down", "error": str(e)}


status = backend_status()
ready = (status.get("status") == "ready")

with st.sidebar:
    st.subheader("Backend")
    st.write(f"Status: **{status.get('status','unknown')}**")
    if status.get("error"):
        st.error(status["error"])

    st.subheader("Ingest")
    files = st.file_uploader(
        "Upload .txt / .md", type=["txt", "md"], accept_multiple_files=True, disabled=not ready)
    if st.button("Ingest", disabled=not ready):
        if not files:
            st.warning("No files selected.")
        else:
            docs = []
            for f in files:
                txt = f.read().decode("utf-8", "ignore")
                docs.append({"external_id": f.name, "text": txt,
                            "provenance": {"uri": f"name://{f.name}"}})
            try:
                r = requests.post(f"{BACKEND}/ingest",
                                  json={"documents": docs}, timeout=180)
                st.success(r.json())
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.subheader("Settings")
    strict_time = st.checkbox(
        "Hard time filter (strict)", value=False, disabled=not ready)
    st.caption(f"Backend: {BACKEND}")

if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if ready:
    q = st.chat_input(
        "Ask a question (e.g., 'CEO as of today', 'Q4 FY23 revenue', 'between 2019 and 2021')…")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            st.markdown("_thinking…_")
            try:
                r = requests.post(f"{BACKEND}/answer", json={"query": q,
                                  "k": 8, "strict_time": strict_time}, timeout=600)
                data = r.json()
                st.empty()
                st.markdown(data.get("answer", "(no answer)"))
                with st.expander("Attribution Card"):
                    st.json(data.get("attribution_card", {}))
                st.session_state.chat.append(
                    {"role": "assistant", "content": data.get("answer", "")})
            except Exception as e:
                st.error(f"Answer failed: {e}")
else:
    st.info("Backend is starting up (downloading/loading models). The chat will unlock when it’s ready.")
