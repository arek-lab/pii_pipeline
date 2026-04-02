import random
import string

import pytest

from src.errors import InternalMaskingError
from src.masking import apply_masks, assert_length
from src.models import Span


# ---------------------------------------------------------------------------
# assert_length
# ---------------------------------------------------------------------------


def test_assert_length_equal_passes():
    assert_length("hello", "-----")  # must not raise


def test_assert_length_shorter_raises():
    with pytest.raises(InternalMaskingError):
        assert_length("hello", "----")


def test_assert_length_longer_raises():
    with pytest.raises(InternalMaskingError):
        assert_length("hi", "---")


def test_assert_length_empty_passes():
    assert_length("", "")


# ---------------------------------------------------------------------------
# apply_masks — basic correctness
# ---------------------------------------------------------------------------


def test_no_spans_returns_original():
    text = "Nic tu nie ma"
    assert apply_masks(text, []) == text


def test_single_span_replaces_letters_with_dash():
    text = "PESEL 85030512345 koniec"
    span = Span(start=6, end=17, label="PESEL", source="regex")
    masked = apply_masks(text, [span])
    assert masked == "PESEL ----------- koniec"
    assert len(masked) == len(text)


def test_spaces_inside_entity_preserved():
    # "Jan Kowalski" (0..12) — space at index 3 must stay
    text = "Jan Kowalski jest tutaj"
    span = Span(start=0, end=12, label="OSOBA", source="ner")
    masked = apply_masks(text, [span])
    assert masked[3] == " "
    assert masked == "--- -------- jest tutaj"
    assert len(masked) == len(text)


def test_multiple_non_overlapping_spans():
    text = "Jan Kowalski, PESEL 85030512345"
    #       0123456789012345678901234567890
    #       0         1         2         3
    spans = [
        Span(start=0, end=3, label="IMIE", source="ner"),
        Span(start=4, end=12, label="NAZWISKO", source="ner"),
        Span(start=20, end=31, label="PESEL", source="regex"),
    ]
    masked = apply_masks(text, spans)
    assert masked == "--- --------, PESEL -----------"
    assert len(masked) == len(text)


def test_span_covering_full_text():
    text = "abc"
    span = Span(start=0, end=3, label="X", source="regex")
    masked = apply_masks(text, [span])
    assert masked == "---"
    assert len(masked) == len(text)


# ---------------------------------------------------------------------------
# apply_masks — length invariant (property-based style)
# ---------------------------------------------------------------------------


def test_length_invariant_50_random_inputs():
    rng = random.Random(42)
    for _ in range(50):
        n = rng.randint(1, 500)
        text = "".join(rng.choices(string.printable, k=n))

        spans: list[Span] = []
        if n > 4:
            start = rng.randint(0, n // 2)
            end = rng.randint(start + 1, min(start + n // 3 + 1, n))
            spans = [Span(start=start, end=end, label="TEST", source="regex")]

        masked = apply_masks(text, spans)
        assert len(masked) == len(text), (
            f"Length mismatch: original={len(text)}, masked={len(masked)}"
        )


def test_length_invariant_unicode():
    text = "Zażółć gęślą jaźń"
    span = Span(start=0, end=6, label="X", source="regex")
    masked = apply_masks(text, [span])
    assert len(masked) == len(text)


def test_length_invariant_multiline():
    text = "Imie: Jan\nNazwisko: Kowalski\nPESEL: 85030512345"
    # "Imie: Jan\n"          = 10 chars → "Jan" at (6, 9)
    # "Nazwisko: Kowalski\n" = 19 chars → "Kowalski" at (20, 28)
    # "PESEL: 85030512345"   = 18 chars → "85030512345" at (36, 47)
    assert text[6:9] == "Jan"
    assert text[20:28] == "Kowalski"
    assert text[36:47] == "85030512345"
    spans = [
        Span(start=6, end=9, label="IMIE", source="ner"),
        Span(start=20, end=28, label="NAZWISKO", source="ner"),
        Span(start=36, end=47, label="PESEL", source="regex"),
    ]
    masked = apply_masks(text, spans)
    assert len(masked) == len(text)


# ---------------------------------------------------------------------------
# apply_masks — non-letter characters outside spans unchanged
# ---------------------------------------------------------------------------


def test_characters_outside_spans_unchanged():
    text = "abc XYZ def"
    span = Span(start=4, end=7, label="X", source="regex")
    masked = apply_masks(text, [span])
    assert masked[:4] == "abc "
    assert masked[7:] == " def"
