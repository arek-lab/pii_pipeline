from dataclasses import dataclass

from .models import Span

# Separators in priority order — split after the separator (separator stays in the preceding chunk)
SEPARATORS = [". ", "? ", "! ", ";\n", "\n\n", "\n", ", ", " i ", " oraz ", "; "]

SOURCE_PRIORITY: dict[str, int] = {"regex": 10, "ner": 8, "teryt": 6}


@dataclass
class Chunk:
    text: str
    offset_start: int  # absolute offset in the original full_text

    # Invariant (verified in tests and optionally at DEBUG runtime):
    #   full_text[offset_start : offset_start + len(text)] == text


def find_split_point(text: str, max_chars: int) -> int:
    """Return the position at which to end a chunk so it is <= max_chars long.

    Searches for the last separator inside text[:max_chars] and returns the
    position *after* the separator (the separator stays in the current chunk).
    Falls back to max_chars if no separator is found.
    """
    window = text[:max_chars]
    for sep in SEPARATORS:
        idx = window.rfind(sep)
        if idx != -1:
            return idx + len(sep)
    return max_chars


def chunk_with_offsets(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Split text into overlapping chunks, never cutting mid-separator.

    Each returned Chunk satisfies:
        text[chunk.offset_start : chunk.offset_start + len(chunk.text)] == chunk.text

    Chunking is only used for NER (FastPDN token limit). Regex and TERYT
    stages receive the full text directly.
    """
    if len(text) <= max_chars:
        return [Chunk(text=text, offset_start=0)]

    chunks: list[Chunk] = []
    pos = 0

    while pos < len(text):
        remaining = text[pos:]
        if len(remaining) <= max_chars:
            chunks.append(Chunk(text=remaining, offset_start=pos))
            break

        split = find_split_point(remaining, max_chars)
        chunks.append(Chunk(text=remaining[:split], offset_start=pos))

        # Next chunk starts overlap_chars before the split point so entities
        # near chunk boundaries get full context.  Always advance by at least 1
        # to prevent an infinite loop on pathological input.
        next_pos = pos + max(1, split - overlap_chars)
        pos = next_pos

    return chunks


def resolve_conflicts(spans: list[Span]) -> list[Span]:
    """Merge overlapping spans, keeping the best one per conflict.

    Resolution rules (in priority order):
    1. Longer span wins — more context implies higher confidence.
    2. Equal length → higher SOURCE_PRIORITY wins (regex > ner > teryt).
    3. Identical (start, end) from different sources → keep one (dedup).
    """
    if not spans:
        return []

    # Sort by start; ties broken by length descending so the greedy pass below
    # naturally encounters the best candidate first.
    sorted_spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))

    result: list[Span] = []
    for span in sorted_spans:
        if not result:
            result.append(span)
            continue

        last = result[-1]
        if span.start >= last.end:
            # No overlap — just append
            result.append(span)
            continue

        # Overlap — keep the better span
        span_len = span.end - span.start
        last_len = last.end - last.start

        if span_len > last_len:
            result[-1] = span
        elif span_len == last_len:
            if SOURCE_PRIORITY.get(span.source, 0) > SOURCE_PRIORITY.get(last.source, 0):
                result[-1] = span
        # else: current span is shorter or lower-priority — discard it

    return result
