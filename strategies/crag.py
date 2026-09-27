"""
Corrective RAG: grades each retrieved chunk's relevance. If overall
confidence is below the threshold, falls back to a web search before
generating, instead of confidently answering off weak context.

Web search fallback uses DuckDuckGo (via duckduckgo-search) - no API key
or signup required, so this correction step just works out of the box.
"""
import llm_client
from vector_store import similarity_search

TOP_K = 6
CONFIDENCE_THRESHOLD = 0.7


def _web_search(query: str, max_results: int = 3) -> list[str]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [r["body"] for r in results if r.get("body")]
    except Exception:
        # DuckDuckGo's unofficial endpoint can rate-limit or fail intermittently;
        # fail soft so CRAG still returns an answer from the retrieved chunks alone.
        return []


def run(collection, query: str) -> dict:
    trace = []

    # 1. Retrieve
    hits = similarity_search(collection, query, top_k=TOP_K)
    trace.append({"step": "Retrieve", "status": "done", "detail": f"{len(hits)} chunks from vector store"})

    # 2. Evaluate each chunk's relevance
    scores = [llm_client.score_relevance(query, h["text"]) for h in hits] if hits else [0.0]
    avg_relevance = sum(scores) / len(scores)
    label = "correct" if avg_relevance >= CONFIDENCE_THRESHOLD else "ambiguous"
    trace.append({"step": "Evaluate", "status": "done", "detail": f"Confidence: {label} ({avg_relevance:.2f})"})

    # 3. Correct: fall back to web search when retrieval is weak
    supplemental = []
    if avg_relevance < CONFIDENCE_THRESHOLD:
        supplemental = _web_search(query)
        detail = f"Low confidence -> pulled {len(supplemental)} web results" if supplemental \
            else "Low confidence, but web search returned nothing - continuing without it"
        trace.append({"step": "Correct", "status": "done" if supplemental else "skipped", "detail": detail})
    else:
        trace.append({"step": "Correct", "status": "skipped", "detail": "Retrieved context was sufficient"})

    # 4. Generate using graded + corrected context
    context = "\n\n---\n\n".join(h["text"] for h in hits) + "\n\n---\n\n" + "\n\n---\n\n".join(supplemental)

    system_prompt = f"""Answer using the context below. Some of it may come from a
corrective web search rather than the user's own documents - if so, say so.

Context:
{context}"""

    answer = llm_client.complete(system_prompt, query)
    trace.append({"step": "Generate", "status": "done", "detail": "Answer composed from graded + corrected context"})

    return {"answer": answer, "sources": hits, "trace": trace}
