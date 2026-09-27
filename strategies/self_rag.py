"""
Self-RAG: decides whether retrieval is even needed, drafts an answer,
critiques the draft against the retrieved sources, and revises if the
critique finds it under-supported. Most LLM calls of the three strategies.
"""
import llm_client
from vector_store import similarity_search

TOP_K = 6
SUPPORT_THRESHOLD = 0.6


def _needs_retrieval(query: str) -> bool:
    prompt = f"""Does answering this question require looking up specific documents,
or can it be answered from general knowledge/conversation alone?
Reply with ONLY "yes" or "no".

Question: {query}"""
    reply = llm_client.complete("You are deciding whether retrieval is necessary.", prompt, max_tokens=5)
    return "yes" in reply.strip().lower()


def run(collection, query: str) -> dict:
    trace = []

    # 1. Reflect
    needs_retrieval = _needs_retrieval(query)
    trace.append({"step": "Reflect", "status": "done",
                  "detail": "Retrieval needed" if needs_retrieval else "Answerable without documents"})

    hits, context = [], ""
    if needs_retrieval:
        hits = similarity_search(collection, query, top_k=TOP_K)
        context = "\n\n---\n\n".join(h["text"] for h in hits)
        trace.append({"step": "Retrieve", "status": "done", "detail": f"{len(hits)} chunks from vector store"})
    else:
        trace.append({"step": "Retrieve", "status": "skipped", "detail": "Not needed for this query"})

    # 2. Draft
    draft_system = f"Answer using this context:\n{context}" if needs_retrieval else "Answer the question directly."
    draft = llm_client.complete(draft_system, query)
    trace.append({"step": "Draft", "status": "done", "detail": "Initial answer generated"})

    # 3. Critique: is the draft supported by the context?
    support = llm_client.score_relevance(draft, context) if needs_retrieval else 1.0
    is_supported = support >= SUPPORT_THRESHOLD
    trace.append({"step": "Critique", "status": "done",
                  "detail": "Draft is well-supported" if is_supported else f"Draft under-supported ({support:.2f})"})

    # 4. Revise if flagged
    final_answer = draft
    if needs_retrieval and not is_supported:
        revise_prompt = f"""The draft below isn't well supported by the context. Revise it to
only state what the context actually supports, and note explicitly what's uncertain.

Context:
{context}

Draft:
{draft}"""
        final_answer = llm_client.complete("You are revising an under-supported answer.", revise_prompt)
        trace.append({"step": "Revise", "status": "done", "detail": "Answer tightened to match supported claims"})
    else:
        trace.append({"step": "Revise", "status": "skipped", "detail": "Draft already well-supported"})

    return {"answer": final_answer, "sources": hits, "trace": trace}
