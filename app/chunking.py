"""Split document text into overlapping passages small enough to embed whole.

Passages break at paragraph, then line, then sentence boundaries where possible,
so a passage rarely starts or ends mid-sentence.
"""
import re

CHUNK_CHARS = 1500     # ~400 tokens: well inside a 2,048-token embedding window
OVERLAP_CHARS = 200    # carried into the next passage so facts spanning a break survive

_BREAKS = ("\n\n", "\n", ". ", " ")


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[tuple[int, str]]:
    """Return (start_offset, passage) pairs covering all of ``text``."""
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [(0, text)]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in _BREAKS:
                cut = window.rfind(sep)
                if cut > size // 2:          # only break where it keeps the passage reasonably full
                    end = start + cut + len(sep)
                    break
        passage = text[start:end].strip()
        if passage:
            chunks.append((start, passage))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
