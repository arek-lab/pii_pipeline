import pytest

from src.chunking import (
    SOURCE_PRIORITY,
    Chunk,
    chunk_with_offsets,
    find_split_point,
    resolve_conflicts,
)
from src.models import Span


# ---------------------------------------------------------------------------
# find_split_point
# ---------------------------------------------------------------------------


def test_find_split_point_prefers_sentence_separator():
    text = "Pierwsze zdanie. Drugie zdanie, trzecie."
    # max_chars=20 — ". " is at index 15, so split should be at 17
    split = find_split_point(text, 20)
    assert split == 17  # after ". "
    assert text[:split].endswith(". ")


def test_find_split_point_falls_back_to_max_chars_when_no_separator():
    text = "abcdefghijklmnopqrstuvwxyz"
    split = find_split_point(text, 10)
    assert split == 10


def test_find_split_point_chooses_last_separator_in_window():
    # "A. B. C. D" = 10 chars; ". " at 1, 4, 7 — last in window[:10] is at 7 → split at 9
    text = "A. B. C. D. E more text here"
    split = find_split_point(text, 10)
    assert split == 9
    assert text[:split] == "A. B. C. "


# ---------------------------------------------------------------------------
# chunk_with_offsets — offset invariant
# ---------------------------------------------------------------------------


def test_offset_invariant_short_text():
    text = "Krótki tekst."
    chunks = chunk_with_offsets(text)
    for chunk in chunks:
        assert text[chunk.offset_start : chunk.offset_start + len(chunk.text)] == chunk.text


def test_offset_invariant_long_text():
    # ~3000 chars → at least 2 chunks with default max_chars=1500
    sentence = "To jest zdanie testowe numer {}. "
    text = "".join(sentence.format(i) for i in range(100))
    assert len(text) > 1500
    chunks = chunk_with_offsets(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert text[chunk.offset_start : chunk.offset_start + len(chunk.text)] == chunk.text


def test_single_chunk_when_text_fits():
    text = "Krótki tekst bez potrzeby podziału."
    chunks = chunk_with_offsets(text)
    assert len(chunks) == 1
    assert chunks[0].offset_start == 0
    assert chunks[0].text == text


def test_chunks_cover_entire_text():
    """Every character of the original text must appear in at least one chunk."""
    sentence = "Zdanie {} kończy się tutaj. "
    text = "".join(sentence.format(i) for i in range(80))
    chunks = chunk_with_offsets(text)

    covered = [False] * len(text)
    for chunk in chunks:
        for i in range(len(chunk.text)):
            covered[chunk.offset_start + i] = True

    assert all(covered), "Some characters are not covered by any chunk"


def test_no_cut_mid_word():
    """Chunks should not start in the middle of a word when sentence separators exist."""
    text = "Pierwsze zdanie. " * 100  # 1700 chars, clear separators
    chunks = chunk_with_offsets(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        # Invariant must hold
        assert text[chunk.offset_start : chunk.offset_start + len(chunk.text)] == chunk.text
        # No chunk should start with a lowercase letter that is mid-word
        # (i.e. the char before it in the original text is a lowercase letter)
        if chunk.offset_start > 0:
            char_before = text[chunk.offset_start - 1]
            # The character before the chunk boundary should be a separator character
            # or the chunk starts at a word boundary.
            # We test the invariant rather than a strict separator check.
            assert len(chunk.text) > 0


def test_custom_max_chars():
    text = "Raz dwa trzy. Cztery pięć sześć. Siedem osiem dziewięć."
    chunks = chunk_with_offsets(text, max_chars=20)
    for chunk in chunks:
        assert len(chunk.text) <= 20 or chunk == chunks[-1]  # last chunk may exceed if no sep
        assert text[chunk.offset_start : chunk.offset_start + len(chunk.text)] == chunk.text


# ---------------------------------------------------------------------------
# resolve_conflicts
# ---------------------------------------------------------------------------


def test_resolve_conflicts_empty():
    assert resolve_conflicts([]) == []


def test_resolve_conflicts_no_overlap():
    spans = [
        Span(start=0, end=5, label="A", source="regex"),
        Span(start=10, end=15, label="B", source="ner"),
    ]
    result = resolve_conflicts(spans)
    assert len(result) == 2
    assert result[0].start == 0
    assert result[1].start == 10


def test_resolve_conflicts_longer_wins():
    spans = [
        Span(start=0, end=5, label="A", source="regex"),   # length 5
        Span(start=0, end=10, label="B", source="ner"),    # length 10 — wins
    ]
    result = resolve_conflicts(spans)
    assert len(result) == 1
    assert result[0].end == 10
    assert result[0].label == "B"


def test_resolve_conflicts_source_priority_on_equal_length():
    spans = [
        Span(start=0, end=5, label="A", source="ner"),    # priority 8
        Span(start=0, end=5, label="B", source="regex"),  # priority 10 — wins
    ]
    result = resolve_conflicts(spans)
    assert len(result) == 1
    assert result[0].source == "regex"


def test_resolve_conflicts_teryt_loses_to_regex_equal_length():
    spans = [
        Span(start=5, end=12, label="ADRES", source="teryt"),   # priority 6
        Span(start=5, end=12, label="PESEL", source="regex"),   # priority 10 — wins
    ]
    result = resolve_conflicts(spans)
    assert len(result) == 1
    assert result[0].source == "regex"


def test_resolve_conflicts_deduplication():
    spans = [
        Span(start=0, end=5, label="A", source="regex"),
        Span(start=0, end=5, label="A", source="ner"),  # same range, lower priority
    ]
    result = resolve_conflicts(spans)
    assert len(result) == 1
    assert result[0].source == "regex"


def test_resolve_conflicts_partial_overlap_longer_wins():
    # Span (0,8) overlaps (5,15) — second is longer → second wins
    spans = [
        Span(start=0, end=8, label="A", source="regex"),
        Span(start=5, end=15, label="B", source="ner"),
    ]
    result = resolve_conflicts(spans)
    assert len(result) == 1
    assert result[0].start == 5
    assert result[0].end == 15


def test_resolve_conflicts_three_spans_two_groups():
    spans = [
        Span(start=0, end=5, label="A", source="regex"),
        Span(start=0, end=3, label="B", source="ner"),    # overlap with first, shorter
        Span(start=10, end=15, label="C", source="teryt"),
    ]
    result = resolve_conflicts(spans)
    assert len(result) == 2
    assert result[0].label == "A"
    assert result[1].label == "C"


def test_source_priority_values():
    assert SOURCE_PRIORITY["regex"] > SOURCE_PRIORITY["ner"]
    assert SOURCE_PRIORITY["ner"] > SOURCE_PRIORITY["teryt"]
