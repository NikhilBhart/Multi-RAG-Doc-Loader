"""Maps the sidebar's mode selection to the matching strategy module."""
from strategies import naive, crag, self_rag

STRATEGIES = {
    "Simple RAG": naive,
    "CRAG": crag,
    "Self-RAG": self_rag,
}


def run(mode: str, collection, query: str) -> dict:
    strategy = STRATEGIES.get(mode)
    if strategy is None:
        raise ValueError(f"Unknown RAG mode: {mode}")
    return strategy.run(collection, query)
