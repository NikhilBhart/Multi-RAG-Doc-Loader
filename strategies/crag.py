
"""
Corrective RAG: grades retrieved chunks and falls back to web search
when retrieval confidence is below the threshold.
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
        # Continue with document context if web search fails.
        return []


def run(
    collection,
    query: str,
    *,
    provider: str,
    model: str,
    api_key: str,
) -> dict:
    trace = []

    # 1. Retrieve
    hits = similarity_search(collection, query, top_k=TOP_K)
    trace.append({
        "step": "Retrieve",
        "status": "done",
        "detail": f"{len(hits)} chunks from vector store",
    })

    # 2. Evaluate relevance using the selected provider
    scores = [
        llm_client.score_relevance(
            query,
            hit["text"],
            provider=provider,
            model=model,
            api_key=api_key,
        )
        for hit in hits
    ] if hits else [0.0]

    avg_relevance = sum(scores) / len(scores)
    label = (
        "correct"
        if avg_relevance >= CONFIDENCE_THRESHOLD
        else "ambiguous"
    )

    trace.append({
        "step": "Evaluate",
        "status": "done",
        "detail": f"Confidence: {label} ({avg_relevance:.2f})",
    })

    # 3. Correct: use web search if document retrieval is weak
    supplemental = []

    if avg_relevance < CONFIDENCE_THRESHOLD:
        supplemental = _web_search(query)

        detail = (
            f"Low confidence -> pulled {len(supplemental)} web results"
            if supplemental
            else "Low confidence, but web search returned nothing - "
                 "continuing without it"
        )

        trace.append({
            "step": "Correct",
            "status": "done" if supplemental else "skipped",
            "detail": detail,
        })
    else:
        trace.append({
            "step": "Correct",
            "status": "skipped",
            "detail": "Retrieved context was sufficient",
        })

    # 4. Generate using document and supplemental web context
    document_context = "\n\n---\n\n".join(
        hit["text"] for hit in hits
    )
    web_context = "\n\n---\n\n".join(supplemental)

    context_parts = []
    if document_context:
        context_parts.append(
            "USER DOCUMENTS:\n" + document_context
        )
    if web_context:
        context_parts.append(
            "SUPPLEMENTAL WEB RESULTS:\n" + web_context
        )

    context = "\n\n---\n\n".join(context_parts) or (
        "No relevant document or web context was found."
    )

    system_prompt = f"""Answer using the context below.
If the answer is not supported by the available context, say you don't know.
Clearly distinguish information from the user's documents from information
obtained through supplemental web search.

Context:
{context}"""

    answer = llm_client.complete(
        system_prompt,
        query,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    trace.append({
        "step": "Generate",
        "status": "done",
        "detail": "Answer composed from graded and corrected context",
    })

    return {
        "answer": answer,
        "sources": hits,
        "trace": trace,
    }
