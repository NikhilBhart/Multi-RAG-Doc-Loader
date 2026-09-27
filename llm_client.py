"""
Wraps the Gemini API (Google's official google-genai SDK). Every strategy's
generation, relevance-grading and critique calls go through this one module.
"""
import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads GEMINI_API_KEY (and any other vars) from a .env file, if present

MODEL = "gemini-3.6-flash"
_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def complete(system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
    response = _client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text or ""


def score_relevance(query_or_answer: str, passage: str) -> float:
    """
    Asks the model to rate how well `passage` supports/relates to
    `query_or_answer`, returns a 0.0-1.0 float. Used by CRAG (chunk grading)
    and Self-RAG (draft-vs-source support check).
    """
    prompt = f"""Rate how well the PASSAGE supports or answers the QUERY, on a
scale from 0.0 (irrelevant/unsupported) to 1.0 (directly relevant/fully supported).
Reply with ONLY the number, nothing else.

QUERY: {query_or_answer}

PASSAGE: {passage}"""

    raw = complete(system_prompt="You are a strict relevance grader.", user_prompt=prompt, max_tokens=10)
    match = re.search(r"[01](\.\d+)?", raw)
    return float(match.group()) if match else 0.5
