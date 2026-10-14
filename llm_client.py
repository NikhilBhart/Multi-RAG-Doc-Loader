
"""Unified LLM client for Gemini, OpenAI, Anthropic, and Groq."""

import os
import re

from dotenv import load_dotenv

load_dotenv()

PROVIDER_ENV_KEYS = {
    "Google Gemini": "GEMINI_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic Claude": "ANTHROPIC_API_KEY",
    "GroqCloud (Llama)": "GROQ_API_KEY",
}

DEFAULT_MODELS = {
    "Google Gemini": "gemini-3.5-flash-lite",
    "OpenAI": "gpt-5-mini",
    "Anthropic Claude": "claude-sonnet-4-6",
    "GroqCloud (Llama)": "llama-3.3-70b-versatile",
}


def complete(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
    *,
    provider: str = "Google Gemini",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Generate a response using the selected provider."""

    if provider not in PROVIDER_ENV_KEYS:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    key = api_key or os.getenv(PROVIDER_ENV_KEYS[provider])
    if not key:
        raise ValueError(
            f"Missing API key for {provider}. "
            "Enter your key in the app or configure your environment."
        )

    model = model or DEFAULT_MODELS[provider]

    if provider == "Google Gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    if provider == "OpenAI":
        from openai import OpenAI

        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    if provider == "Anthropic Claude":
        from anthropic import Anthropic

        client = Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return "".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

    if provider == "GroqCloud (Llama)":
        from groq import Groq

        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    raise ValueError(f"Unsupported LLM provider: {provider}")


def score_relevance(
    query_or_answer: str,
    passage: str,
    *,
    provider: str = "Google Gemini",
    model: str | None = None,
    api_key: str | None = None,
) -> float:
    """Grade passage relevance on a scale from 0.0 to 1.0."""

    prompt = f"""Rate how well the PASSAGE supports or answers the QUERY,
on a scale from 0.0 (irrelevant/unsupported) to 1.0
(directly relevant/fully supported).
Reply with ONLY a number between 0.0 and 1.0.

QUERY: {query_or_answer}

PASSAGE: {passage}"""

    raw = complete(
        system_prompt="You are a strict relevance grader.",
        user_prompt=prompt,
        max_tokens=10,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    match = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", raw)
    if not match:
        return 0.5

    return max(0.0, min(1.0, float(match.group())))



def authenticate_provider(provider: str, api_key: str) -> None:
    """Validate the API key with the selected provider."""

    if not api_key or not api_key.strip():
        raise ValueError("Please enter an API key.")

    if provider == "Google Gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
        next(iter(client.models.list()), None)

    elif provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        client.models.list()

    elif provider == "Anthropic Claude":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        client.models.list()

    elif provider == "GroqCloud (Llama)":
        from groq import Groq
        client = Groq(api_key=api_key)
        client.models.list()

    else:
        raise ValueError(f"Unsupported provider: {provider}")
