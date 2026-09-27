# RAG Doc Assistant (Streamlit)

Upload documents, then chat with them using a switchable retrieval
strategy — Simple RAG, CRAG, or Self-RAG — picked per question in the sidebar.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

That's the only key needed. CRAG's web-search fallback uses DuckDuckGo, which requires no API key or signup.

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## How it's organized

```
app.py                 Streamlit UI - upload, mode switcher, chat
ingest.py               parse (pdf/docx/txt) -> chunk -> embed -> store
vector_store.py         Chroma wrapper (local, on-disk, no server needed)
llm_client.py           wraps Claude API - complete() and score_relevance()
strategies/
  naive.py               Simple RAG: embed -> top-k -> generate
  crag.py                 + relevance grading + web-search correction
  self_rag.py             + retrieval-needed check + self-critique + revise
  factory.py              maps the sidebar's mode name -> strategy module
```

## Notes

- **Embeddings run locally** (sentence-transformers `all-MiniLM-L6-v2`), so
  you only need an API key for generation (Claude), not for embedding. This
  keeps cost down and avoids a second vendor dependency.
- **Chroma persists to `./chroma_db`** on disk between runs — delete that
  folder to reset your document library.
- **Adding a 4th strategy** (RAG-Fusion, HyDE, ...): add a new file in
  `strategies/` with a `run(collection, query) -> dict` function, register
  it in `strategies/factory.py`'s `STRATEGIES` dict, and add its name to the
  `st.radio` options in `app.py`. Nothing else changes.
- CRAG's web-search correction step uses DuckDuckGo (`duckduckgo-search`),
  which needs no API key. It's an unofficial endpoint, so it can occasionally
  rate-limit or fail — if that happens, CRAG fails soft and just answers from
  the retrieved chunks, same as graded naive RAG.
