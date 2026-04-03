"""TERYT stage — detect Polish place and street names via Morfeusz2 lemmatisation.

Masking policy (CLAUDE.md table):
  SIMC (places):  always mask on match
  ULIC (streets): always mask on match
  TERC (gminy/powiaty/województwa): mask ONLY when preceded by an address keyword

Role in the pipeline
---------------------
This stage is an edge-case handler for names that NER misses because they
look like common nouns or appear in atypical casing:
  - proper names written in lowercase  ("ul. mickiewicza")
  - inflected forms NER didn't tag     ("Mieszkam na Różanej")
  - administrative units in formal prose ("pow. poznański")

Ambiguous tokens without an address keyword (bare "Mickiewicza 5" at sentence
start, list items, etc.) are intentionally left to the NER stage.

Keyword matching
-----------------
_ADDR_KEYWORD_SUFFIX uses re.IGNORECASE so all orthographic variants work:
  ul. / UL. / Ul.
  ulicy / Ulicy / ULICY
  osiedle / Osiedle
  …and so on for every keyword.

This stage is synchronous — Morfeusz2 is a fast C++ binding and does not need
run_in_executor.  It operates on the full, unmodified original text.
"""

import re

from ..models import Span
from ..singletons import get_morfeusz, get_teryt

# Single capitalised token in Polish orthography (handles diacritics).
# Requires at least one lowercase letter after the capital so all-caps
# abbreviations (NIP, PESEL…) are not accidentally matched.
_UPPER_TOKEN = re.compile(r"\b[A-ZŁŻŹĆŚĄÓĘŃ][a-złżźćśąóęń]+\b")

# All-lowercase token, min 3 chars to skip short prepositions (w, na, do…).
_LOWER_TOKEN = re.compile(r"\b[a-złżźćśąóęń]{3,}\b")

# Address keyword immediately preceding the name token.
# re.IGNORECASE handles sentence-initial caps, all-caps typos, abbreviations.
# Full inflected forms are listed so Morfeusz2 doesn't need to be called on
# keywords — keeps this a fast pure-regex check.
_ADDR_KEYWORD_SUFFIX = re.compile(
    r"(?:"
    r"ul\.|al\.|pl\.|os\.|gm\.|pow\.|woj\."           # abbreviations
    r"|ulic[ay]|ulicą|ulicę|ulica"                     # ulica (all cases)
    r"|alei|aleją|aleję|aleja|aleje"                   # aleja
    r"|placu|placem|plac"                               # plac
    r"|osiedl[au]|osiedlem|osiedle|osiedli"             # osiedle
    r"|skweru?|rynku?|bulwaru?|promenady?"              # remaining street types
    r"|ronda?|obwodnicy?|traktu?|drogi?|szosy?"         # more street types
    r"|wybrzeż[ae]|wzgórz[ae]|parku?"                   # topographic
    r"|województw[ao]?|powiatu?|gmin[ay]?"              # admin (TERC)
    r")\s*$",
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

    # ------------------------------------------------------------------
    # Pass 1: capitalised tokens — matched unconditionally for SIMC/ULIC
    # ------------------------------------------------------------------
    for m in _UPPER_TOKEN.finditer(text):
        token = m.group()
        start, end = m.start(), m.end()
        lemmas = _lemmas(morfeusz, token)

        # SIMC / ULIC — mask unconditionally
        if lemmas & teryt_simc or lemmas & teryt_ulic:
            spans.append(Span(start=start, end=end, label="ADRES", source="teryt"))
            continue

        # TERC — mask only when an address keyword immediately precedes
        if lemmas & teryt_terc:
            if _ADDR_KEYWORD_SUFFIX.search(text[:start]):
                spans.append(Span(start=start, end=end, label="ADRES", source="teryt"))

    # ------------------------------------------------------------------
    # Pass 2: lowercase tokens — ONLY when preceded by an address keyword
    # ------------------------------------------------------------------
    for m in _LOWER_TOKEN.finditer(text):
        token = m.group()
        start, end = m.start(), m.end()

        if not _ADDR_KEYWORD_SUFFIX.search(text[:start]):
            continue

        lemmas = _lemmas(morfeusz, token)

        if lemmas & teryt_simc or lemmas & teryt_ulic or lemmas & teryt_terc:
            spans.append(Span(start=start, end=end, label="ADRES", source="teryt"))

    return spans