"""
Parses an uploaded file (PDF/DOCX/TXT), splits it into overlapping chunks,
and writes them into the vector store. Run once per upload.
"""
import uuid
from pypdf import PdfReader
from docx import Document as DocxDocument

from vector_store import add_chunks

CHUNK_SIZE = 800       # characters, not tokens - simple and dependency-free
CHUNK_OVERLAP = 120


def extract_text(uploaded_file) -> str:
    print("DEBUG object type:", type(uploaded_file))
    print("DEBUG object name:", repr(getattr(uploaded_file, "name", None)))
    """uploaded_file is a Streamlit UploadedFile."""
    name = uploaded_file.name.lower()

    # raise ValueError(
    # f"Unsupported file type. Filename={uploaded_file.name!r}"
    # )

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = DocxDocument(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith((".txt", ".md")):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple fixed-size sliding-window chunker."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def ingest_file(collection, uploaded_file) -> int:
    print("DEBUG collection type:", type(collection))
    print("DEBUG uploaded_file:", repr(getattr(uploaded_file, "name", None)))

    """Parses, chunks, embeds and stores one uploaded file. Returns chunk count."""
    text = extract_text(uploaded_file)
    chunks = chunk_text(text)

    doc_id = str(uuid.uuid4())
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"document_name": uploaded_file.name, "document_id": doc_id, "chunk_index": i}
        for i in range(len(chunks))
    ]

    add_chunks(collection, chunks, metadatas, ids)
    return len(chunks)
