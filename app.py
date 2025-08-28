# Streamlit frontend for TimeGuard Graph-RAG system
# Provides web UI for document ingestion and time-aware Q&A

import os
import json
import requests
import streamlit as st
import time

# Backend service URL - defaults to local development server
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Configure Streamlit page settings
st.set_page_config(page_title="TimeGuard Graph-RAG (Qwen)", layout="wide")
st.title("🕒 TimeGuard Graph-RAG — Qwen")


def backend_status():
    """
    Check if backend service is running and ready
    Returns status dict with health information
    """
    try:
        r = requests.get(f"{BACKEND}/healthz", timeout=3)
        return r.json()
    except Exception as e:
        return {"status": "down", "error": str(e)}


# Get current backend health status
status = backend_status()
ready = (status.get("status") == "ready")

# Sidebar for backend status and document ingestion
with st.sidebar:
    # Display backend connection status
    st.subheader("Backend")
    st.write(f"Status: **{status.get('status','unknown')}**")
    if status.get("error"):
        st.error(status["error"])

    # Document upload and ingestion interface
    st.subheader("Ingest")
    files = st.file_uploader(
        "Upload .txt / .md", type=["txt", "md"], accept_multiple_files=True, disabled=not ready)
    
    # Process uploaded files when ingest button clicked
    if st.button("Ingest", disabled=not ready):
        if not files:
            st.warning("No files selected.")
        else:
            # Convert uploaded files to document format
            docs = []
            for f in files:
                txt = f.read().decode("utf-8", "ignore")
                docs.append({"external_id": f.name, "text": txt,
                            "provenance": {"uri": f"name://{f.name}"}})
            try:
                # Send documents to backend for processing
                r = requests.post(f"{BACKEND}/ingest",
                                  json={"documents": docs}, timeout=180)
                st.success(r.json())
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    # Query configuration settings
    st.subheader("Settings")
    strict_time = st.checkbox(
        "Hard time filter (strict)", value=False, disabled=not ready)
    st.caption(f"Backend: {BACKEND}")

# Initialize chat history in session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display existing chat messages
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Main chat interface - only active when backend is ready
if ready:
    # Query input field with example prompts
    q = st.chat_input(
        "Ask a question (e.g., 'CEO as of today', 'Q4 FY23 revenue', 'between 2019 and 2021')…")
    
    # Process user query
    if q:
        # Add user message to chat history
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            st.markdown("_thinking…_")
            try:
                # Send query to backend for processing
                r = requests.post(f"{BACKEND}/answer", json={"query": q,
                                  "k": 8, "strict_time": strict_time}, timeout=600)
                data = r.json()
                
                # Clear thinking indicator and show response
                st.empty()
                st.markdown(data.get("answer", "(no answer)"))
                
                # Show detailed attribution information in expandable section
                with st.expander("Attribution Card"):
                    st.json(data.get("attribution_card", {}))
                
                # Add assistant response to chat history
                st.session_state.chat.append(
                    {"role": "assistant", "content": data.get("answer", "")})
            except Exception as e:
                st.error(f"Answer failed: {e}")
else:
    # Show loading message when backend is not ready
    st.info("Backend is starting up (downloading/loading models). The chat will unlock when it's ready.")
