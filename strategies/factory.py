
"""Maps retrieval modes to their strategy modules."""

from strategies import naive, crag, self_rag

STRATEGIES = {
    "Simple RAG": naive,
    "CRAG": crag,
    "Self-RAG": self_rag,
}


def run(
    mode: str,
    collection,
    query: str,
    *,
    provider: str,
    model: str,
    api_key: str,
) -> dict:
    strategy = STRATEGIES.get(mode)

    if strategy is None:
        raise ValueError(f"Unknown RAG mode: {mode}")

    return strategy.run(
        collection,
        query,
        provider=provider,
        model=model,
        api_key=api_key,
    )