"""TERYT stage — detect Polish place and street names via Morfeusz2 lemmatisation.

Masking policy (CLAUDE.md table):
  SIMC (places):  always mask on match
  ULIC (streets): always mask on match
  TERC (gminy/powiaty/województwa): mask ONLY when preceded by an address keyword

This stage is synchronous — Morfeusz2 is a fast C++ binding and does not need
run_in_executor.  It operates on the full, unmodified original text.
"""

import re

from ..models import Span
from ..singletons import get_morfeusz, get_teryt

# Single capitalised token in Polish orthography (handles diacritics).
_UPPER_TOKEN = re.compile(r"\b[A-ZŁŻŹĆŚĄÓĘŃ][a-złżźćśąóęń]+\b")

# Address keyword that must appear at the end of the text preceding a TERC token.
# Covers abbreviated and full-word forms used in Polish administrative prose.
_ADDR_KEYWORD_SUFFIX = re.compile(
    r"(?:woj\.|pow\.|gm\.|woj\b|pow\b|gm\b"
    r"|województw[ao]?|powiatu?|gmin[ay]?)\s*$",
    re.IGNORECASE,
)


def _lemmas(morfeusz, token: str) -> set[str]:
    """Return the set of lowercase base forms Morfeusz2 proposes for *token*.

    Morfeusz2 analysis format:
        [(seg_start, seg_end, (orth, lemma, tag, ...)), ...]
    """
    analyses = morfeusz.analyse(token)
    return {a[2][1].lower() for a in analyses}


def run(text: str) -> list[Span]:
    """Detect TERYT entities in *text* and return absolute Spans.

    All three stages receive the original text — never the partially-masked one.
    """
    morfeusz = get_morfeusz()
    teryt_simc, teryt_ulic, teryt_terc = get_teryt()
    spans: list[Span] = []

    for m in _UPPER_TOKEN.finditer(text):
        token = m.group()
        start, end = m.start(), m.end()

        lemmas = _lemmas(morfeusz, token)

        # SIMC / ULIC — mask unconditionally
        if lemmas & teryt_simc or lemmas & teryt_ulic:
            spans.append(Span(start=start, end=end, label="ADRES", source="teryt"))
            continue

        # TERC — mask only when an address keyword immediately precedes the token
        if lemmas & teryt_terc:
            preceding = text[:start]
            if _ADDR_KEYWORD_SUFFIX.search(preceding):
                spans.append(Span(start=start, end=end, label="ADRES", source="teryt"))

    return spans
