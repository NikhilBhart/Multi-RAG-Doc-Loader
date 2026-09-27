import streamlit as st

from vector_store import get_collection
from ingest import ingest_file
from strategies.factory import run as run_strategy

st.set_page_config(page_title="RAG Doc Assistant", page_icon="📄", layout="wide")

# ---- session state -----------------------------------------------------
if "collection" not in st.session_state:
    st.session_state.collection = get_collection()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # list of {role, content, sources?, trace?}
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

# ---- sidebar: upload + mode switcher -----------------------------------
with st.sidebar:
    st.title("📄 RAG Doc Assistant")

    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_names:
                with st.spinner(f"Indexing {f.name}..."):
                    chunk_count = ingest_file(st.session_state.collection, f)
                st.session_state.uploaded_names.append(f.name)
                st.success(f"{f.name} - {chunk_count} chunks indexed")

    if st.session_state.uploaded_names:
        st.caption("Indexed files:")
        for name in st.session_state.uploaded_names:
            st.markdown(f"- {name}")

    st.divider()
    st.subheader("Retrieval mode")
    mode = st.radio(
        "Choose how questions get answered:",
        ["Simple RAG", "CRAG", "Self-RAG"],
        captions=[
            "Fastest - one retrieval pass, no self-checking",
            "Grades retrieval, corrects with web search if weak",
            "Decides if retrieval's needed, critiques its own answer",
        ],
    )

# ---- main: chat ----------------------------------------------------------
st.header("Chat with your documents")

if not st.session_state.uploaded_names:
    st.info("Upload at least one document in the sidebar to get started.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['metadata']['document_name']}** · relevance {s['score']}")
                    st.caption(s["text"][:300] + "...")
        if msg.get("trace"):
            with st.expander(f"Pipeline ({msg.get('mode', '')})"):
                for step in msg["trace"]:
                    icon = {"done": "✅", "skipped": "⏭️", "running": "⏳"}.get(step["status"], "•")
                    st.markdown(f"{icon} **{step['step']}** — {step['detail']}")

query = st.chat_input("Ask about your documents...")
if query:
    if not st.session_state.uploaded_names:
        st.warning("Upload a document first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner(f"Running {mode}..."):
                result = run_strategy(mode, st.session_state.collection, query)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.markdown(f"**{s['metadata']['document_name']}** · relevance {s['score']}")
                        st.caption(s["text"][:300] + "...")
            with st.expander(f"Pipeline ({mode})"):
                for step in result["trace"]:
                    icon = {"done": "✅", "skipped": "⏭️", "running": "⏳"}.get(step["status"], "•")
                    st.markdown(f"{icon} **{step['step']}** — {step['detail']}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "trace": result["trace"],
            "mode": mode,
        })
