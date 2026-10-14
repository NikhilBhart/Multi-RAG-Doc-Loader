
"""
Self-RAG: decides whether retrieval is needed, drafts an answer,
critiques the draft against retrieved sources, and revises if needed.
"""

import llm_client
from vector_store import similarity_search

TOP_K = 6
SUPPORT_THRESHOLD = 0.6


def _needs_retrieval(
    query: str,
    *,
    provider: str,
    model: str,
    api_key: str,
) -> bool:
    prompt = f"""Does answering this question require looking up specific documents,
or can it be answered from general knowledge/conversation alone?
Reply with ONLY "yes" or "no".

Question: {query}"""

    reply = llm_client.complete(
        "You are deciding whether retrieval is necessary.",
        prompt,
        max_tokens=5,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    return reply.strip().lower().startswith("yes")


def run(
    collection,
    query: str,
    *,
    provider: str,
    model: str,
    api_key: str,
) -> dict:
    trace = []

    # 1. Reflect
    needs_retrieval = _needs_retrieval(
        query,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    trace.append({
        "step": "Reflect",
        "status": "done",
        "detail": (
            "Retrieval needed"
            if needs_retrieval
            else "Answerable without documents"
        ),
    })

    # 2. Retrieve
    hits, context = [], ""

    if needs_retrieval:
        hits = similarity_search(
            collection, query, top_k=TOP_K
        )
        context = "\n\n---\n\n".join(
            hit["text"] for hit in hits
        )

        trace.append({
            "step": "Retrieve",
            "status": "done",
            "detail": f"{len(hits)} chunks from vector store",
        })
    else:
        trace.append({
            "step": "Retrieve",
            "status": "skipped",
            "detail": "Not needed for this query",
        })

    # 3. Draft
    if needs_retrieval:
        draft_system = f"""Answer using ONLY the context below.
If the answer is not supported by the context, say you don't know.

Context:
{context}"""
    else:
        draft_system = "Answer the user's question directly."

    draft = llm_client.complete(
        draft_system,
        query,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    trace.append({
        "step": "Draft",
        "status": "done",
        "detail": "Initial answer generated",
    })

    # 4. Critique
    if needs_retrieval and context:
        support = llm_client.score_relevance(
            draft,
            context,
            provider=provider,
            model=model,
            api_key=api_key,
        )
    else:
        support = 1.0

    is_supported = support >= SUPPORT_THRESHOLD

    trace.append({
        "step": "Critique",
        "status": "done",
        "detail": (
            "Draft is well-supported"
            if is_supported
            else f"Draft under-supported ({support:.2f})"
        ),
    })

    # 5. Revise if the draft is under-supported
    final_answer = draft

    if needs_retrieval and not is_supported:
        revise_prompt = f"""Revise the draft to state only what the
context supports. Explicitly identify anything uncertain.

Context:
{context}

Draft:
{draft}"""

        final_answer = llm_client.complete(
            "You are revising an under-supported answer.",
            revise_prompt,
            provider=provider,
            model=model,
            api_key=api_key,
        )

        trace.append({
            "step": "Revise",
            "status": "done",
            "detail": "Answer tightened to match supported claims",
        })
    else:
        trace.append({
            "step": "Revise",
            "status": "skipped",
            "detail": "Draft already well-supported",
        })

    return {
        "answer": final_answer,
        "sources": hits,
        "trace": trace,
    }
