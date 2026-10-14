
import os

import streamlit as st
from dotenv import load_dotenv

from ingest import ingest_file
from vector_store import get_collection
from strategies.factory import run as run_strategy
import llm_client

load_dotenv()

st.set_page_config(
    page_title="Multi-RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-RAG Chatbot")
st.caption(
    "Chat with your documents using Gemini, OpenAI, Claude, or GroqCloud-hosted Llama."
)

# Display name -> actual provider API model ID
MODEL_OPTIONS = {
    "Google Gemini": {
        "Gemini 3.5 Flash-Lite": "gemini-3.5-flash-lite",
        "Gemini 3.8 Flash": "gemini-3.8-flash",
    },
    "OpenAI": {
        "GPT-5 mini": "gpt-5-mini",
        "GPT-5 nano": "gpt-5-nano",
    },
    "Anthropic Claude": {
        "Claude Haiku 5.5": "claude-haiku-5-5",
        "Claude Sonnet 5.5": "claude-sonnet-5-5",
    },
    "GroqCloud (Llama)": {
        "Llama 3.3 70B": "llama-3.3-70b-versatile",
    },
}

DEFAULT_MODELS = {
    "Google Gemini": "Gemini 3.5 Flash-Lite",
    "OpenAI": "GPT-5 mini",
    "Anthropic Claude": "Claude Haiku 5.5",
    "GroqCloud (Llama)": "Llama 3.3 70B",
}

PROVIDER_ENV_KEYS = {
    "Google Gemini": "GEMINI_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic Claude": "ANTHROPIC_API_KEY",
    "GroqCloud (Llama)": "GROQ_API_KEY",
}

RAG_OPTIONS = {
    "Simple RAG": "Simple RAG",
    "CRAG": "CRAG",
    "Self-RAG": "Self-RAG",
}

# Session state
if "collection" not in st.session_state:
    st.session_state.collection = get_collection()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()

if "connected_providers" not in st.session_state:
    st.session_state.connected_providers = {}

# Sidebar
with st.sidebar:
    st.header("Model Configuration")

    provider = st.selectbox(
        "AI Provider",
        options=list(MODEL_OPTIONS.keys()),
        key="ai_provider",
    )

    model_names = list(MODEL_OPTIONS[provider].keys())
    default_index = model_names.index(DEFAULT_MODELS[provider])

    model_name = st.selectbox(
        "AI Model",
        options=model_names,
        index=default_index,
        key=f"model_{provider}",
    )

    model_id = MODEL_OPTIONS[provider][model_name]

    env_key_name = PROVIDER_ENV_KEYS[provider]
    env_key = os.getenv(env_key_name, "")

    # Keep a separate key input for each provider.
    key_widget = f"api_key_{provider}"

    if key_widget not in st.session_state:
        st.session_state[key_widget] = env_key

    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        key=key_widget,
        placeholder="Paste your provider API key",
        help=(
            f"Enter your own key or configure {env_key_name} "
            "in your local .env file."
        ),
    )

    # Connection is associated with the selected provider and model.
    connection_id = f"{provider}::{model_id}"

    # Connect and Clear buttons
    connect_col, clear_col = st.columns(2)

    with connect_col:
        connect_clicked = st.button(
            "Connect",
            key=f"connect_{provider}",
            use_container_width=True,
        )

    with clear_col:
        clear_clicked = st.button(
            "Clear",
            key=f"clear_{provider}",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state[key_widget] = ""
        st.session_state.connected_providers.pop(
            connection_id, None
        )
        st.rerun()

    # Authenticate only when Connect is clicked.
    if connect_clicked:
        entered_key = api_key.strip()

        # Remove any old status for this provider/model first.
        st.session_state.connected_providers.pop(
            connection_id, None
        )

        if not entered_key:
            st.warning("Please enter your API key first.")
        else:
            try:
                with st.spinner("Authenticating API key..."):
                    llm_client.authenticate_provider(
                        provider,
                        entered_key,
                    )

                st.session_state.connected_providers[
                    connection_id
                ] = True

                st.success("Connected successfully!")

            except Exception as exc:
                st.session_state.connected_providers.pop(
                    connection_id, None
                )
                st.error("Connection failed. Check your API key and access.")

                with st.expander("Technical error details"):
                    st.code(f"{type(exc).__name__}: {exc}")

    # Display current connection status.
    if st.session_state.connected_providers.get(connection_id):
        st.success("🟢 Connected")
    else:
        st.caption("⚪ Not connected")

    st.caption(f"Model ID: `{model_id}`")

    st.divider()
    st.header("RAG Configuration")

    rag_mode = st.selectbox(
        "Retrieval Strategy",
        options=list(RAG_OPTIONS.keys()),
    )
    strategy_id = RAG_OPTIONS[rag_mode]

    st.divider()
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state.uploaded_names:
                    st.info(f"Already processed: {uploaded_file.name}")
                    continue

                try:
                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):
                        ingest_file(
                            st.session_state.collection,
                            uploaded_file,
                        )

                    st.session_state.uploaded_names.add(
                        uploaded_file.name
                    )
                    st.success(f"Processed: {uploaded_file.name}")

                except Exception as exc:
                    st.error(
                        f"Could not process {uploaded_file.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Conversation history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat

# Chat
query = st.chat_input("Ask a question about your documents...")

if query:
    connection_id = f"{provider}::{model_id}"

    if not st.session_state.connected_providers.get(connection_id):
        st.warning(
            f"Please connect your {provider} API key using the "
            "Connect button before chatting."
        )
        st.stop()

    api_key = st.session_state[key_widget].strip()

    if not api_key:
        st.warning("Your API key is empty. Please connect again.")
        st.stop()

    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner(f"Generating answer with {model_name}..."):
                result = run_strategy(
                    strategy_id,
                    st.session_state.collection,
                    query,
                    provider=provider,
                    model=model_id,
                    api_key=api_key,
                )

            # Extract only the answer from the strategy result.
            if isinstance(result, dict):
                answer = result.get("answer", "")
            else:
                answer = str(result)

            if not answer:
                answer = "I couldn't generate an answer."

            # Display only the answer in the main chat.
            st.markdown(answer)

            # Keep the stored chat history clean.
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

            # Optional: show sources separately, without metadata or scores.
            if isinstance(result, dict) and result.get("sources"):
                with st.expander("View sources"):
                    seen = set()

                    for source in result["sources"]:
                        document_name = source.get(
                            "metadata", {}
                        ).get("document_name", "Document")

                        text = source.get("text", "").strip()
                        source_key = (document_name, text)

                        # Avoid displaying duplicate chunks.
                        if text and source_key not in seen:
                            seen.add(source_key)
                            st.markdown(f"**{document_name}**")
                            st.write(text)

        except Exception as exc:
            st.error(
                "The request failed. Check your provider, API key, "
                "model access, and application logs."
            )

            with st.expander("Technical error details"):
                st.code(f"{type(exc).__name__}: {exc}")
