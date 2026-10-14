"""Simple RAG: one retrieval pass, one generation call. Fast and cheap."""

import llm_client
from vector_store import similarity_search

TOP_K = 6


def run(
    collection,
    query: str,
    *,
    provider: str,
    model: str,
    api_key: str,
) -> dict:
    hits = similarity_search(collection, query, top_k=TOP_K)
    context = "\n\n---\n\n".join(h["text"] for h in hits)

    system_prompt = f"""Answer the user's question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}"""

    answer = llm_client.complete(
        system_prompt,
        query,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    trace = [
        {
            "step": "Retrieve",
            "status": "done",
            "detail": f"{len(hits)} chunks from vector store",
        },
        {
            "step": "Generate",
            "status": "done",
            "detail": "Answered directly from retrieved context",
        },
    ]

    return {
        "answer": answer,
        "sources": hits,
        "trace": trace,
    }