"""
pdf_tool.py

Self-contained retrieval tool for the PDF Q&A agent.

Pipeline (a minimal, dependency-light RAG):
  1. Extract text from an uploaded PDF (`pypdf`).
  2. Split it into overlapping chunks.
  3. Build a TF-IDF vector for every chunk, purely with numpy -- no
     scikit-learn, no vector database, and no embeddings API call, so this
     stays free and works fully offline once the PDF is read.
  4. At query time, vectorize the question the same way and rank chunks by
     cosine similarity, returning the top few as context for the LLM.

This is intentionally simple (word-overlap-based retrieval, not semantic
embeddings) -- good enough for a single uploaded document, and easy to
explain end-to-end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError

CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# A short stopword list keeps ultra-common words from dominating similarity
# scores, which matters most for short documents where IDF alone can't tell
# "the"/"is"/"of" apart from a genuinely meaningful term.
STOPWORDS = frozenset(
    """
    a an the this that these those and or but if then else so than too very
    is are was were be been being do does did doing have has had having
    i you he she it we they me him her us them my your his its our their
    of in on at to for with from by about as into over after before under
    up down out off again further here there when where why how all any
    both each few more most other some such no nor not only own same
    can will just should now
    """.split()
)


class PdfProcessingError(Exception):
    """Raised whenever a PDF cannot be read or contains no extractable text."""


@dataclass
class DocumentIndex:
    """A searchable TF-IDF index over one document's chunks."""

    chunks: list[str]
    vocabulary: dict[str, int]
    idf: np.ndarray  # shape: (vocab_size,)
    chunk_vectors: np.ndarray  # shape: (num_chunks, vocab_size), L2-normalized


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS
    ]


def extract_text_from_pdf(file: BinaryIO) -> str:
    """Extract all text from an uploaded PDF file-like object.

    Raises:
        PdfProcessingError: if the PDF can't be parsed or has no extractable
            text (e.g. a scanned/image-only PDF with no text layer).
    """
    try:
        reader = PdfReader(file)
    except PdfReadError as exc:
        raise PdfProcessingError(f"Could not read this PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - pypdf raises varied types
            raise PdfProcessingError(
                "This PDF is password-protected and could not be opened."
            ) from exc

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a single malformed page shouldn't fail the whole doc
            continue

    text = "\n".join(pages_text).strip()
    if not text:
        raise PdfProcessingError(
            "No extractable text found -- this PDF may be scanned images "
            "without a text layer (OCR would be needed)."
        )
    return text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping fixed-size chunks, on whitespace boundaries."""
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # Keep the tail of this chunk as the start of the next one, so a
            # relevant sentence spanning a chunk boundary isn't lost.
            overlap_words: list[str] = []
            overlap_len = 0
            for word_back in reversed(current):
                overlap_len += len(word_back) + 1
                overlap_words.insert(0, word_back)
                if overlap_len >= overlap:
                    break
            current = overlap_words
            current_len = overlap_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_index(chunks: list[str]) -> DocumentIndex:
    """Build a TF-IDF index over the given chunks using plain numpy."""
    tokenized_chunks = [_tokenize(chunk) for chunk in chunks]

    vocabulary: dict[str, int] = {}
    for tokens in tokenized_chunks:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    num_chunks = len(chunks)
    vocab_size = len(vocabulary)
    term_frequency = np.zeros((num_chunks, vocab_size), dtype=np.float64)

    for row, tokens in enumerate(tokenized_chunks):
        for token in tokens:
            term_frequency[row, vocabulary[token]] += 1

    document_frequency = np.count_nonzero(term_frequency, axis=0)
    # Standard smoothed IDF: log((1 + N) / (1 + df)) + 1, always positive.
    idf = np.log((1 + num_chunks) / (1 + document_frequency)) + 1.0

    tfidf = term_frequency * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for an empty chunk
    chunk_vectors = tfidf / norms

    return DocumentIndex(
        chunks=chunks,
        vocabulary=vocabulary,
        idf=idf,
        chunk_vectors=chunk_vectors,
    )


def _vectorize_query(query: str, index: DocumentIndex) -> np.ndarray:
    """Project a query into the same TF-IDF space as the indexed chunks."""
    vector = np.zeros(len(index.vocabulary), dtype=np.float64)
    for token in _tokenize(query):
        column = index.vocabulary.get(token)
        if column is not None:
            vector[column] += 1.0
    vector *= index.idf
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def search(
    index: DocumentIndex, query: str, top_k: int = 4, min_score: float = 1e-6
) -> list[tuple[str, float]]:
    """Return the top-k chunks most relevant to `query`, as (chunk, score) pairs."""
    if not index.chunks:
        return []

    query_vector = _vectorize_query(query, index)
    scores = index.chunk_vectors @ query_vector  # cosine similarity (already normalized)

    ranked_positions = np.argsort(scores)[::-1][:top_k]
    return [
        (index.chunks[position], float(scores[position]))
        for position in ranked_positions
        if scores[position] > min_score
    ]


# Words that signal the user is asking about the document as a whole
# ("what's in this pdf?", "summarize it", "what topics does it cover?")
# rather than a specific fact. These almost never appear as content words
# inside the document's own body text, so plain keyword-overlap search
# scores them near zero and would wrongly claim the document doesn't cover
# the question -- these queries need broad coverage instead of narrow match.
BROAD_QUERY_KEYWORDS = frozenset(
    {
        "summary", "summarize", "summarise", "overview", "about", "topic",
        "topics", "contain", "contains", "content", "contents", "describe",
        "explain", "document", "pdf", "file", "whole", "entire", "everything",
        "gist", "tldr",
    }
)


def is_broad_query(query: str) -> bool:
    """True if the question is about the document as a whole, not a specific fact."""
    tokens = set(TOKEN_PATTERN.findall(query.lower()))
    return bool(tokens & BROAD_QUERY_KEYWORDS)


def sample_chunks(index: DocumentIndex, max_chunks: int = 6) -> list[str]:
    """Evenly-spaced sample of chunks spanning the whole document.

    Used for overview-style questions where similarity search isn't the
    right tool -- it gives the LLM a representative slice from the start,
    middle, and end of the document instead of just the top keyword matches.
    """
    total = len(index.chunks)
    if total <= max_chunks:
        return list(index.chunks)
    step = total / max_chunks
    positions = sorted({int(i * step) for i in range(max_chunks)})
    return [index.chunks[position] for position in positions]


def build_index_from_pdf(file: BinaryIO) -> DocumentIndex:
    """Convenience wrapper: extract, chunk, and index a PDF in one call."""
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    if not chunks:
        raise PdfProcessingError("The PDF had no usable text after processing.")
    return build_index(chunks)
