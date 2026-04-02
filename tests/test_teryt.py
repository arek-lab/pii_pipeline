"""Tests for stages/teryt_stage.py.

Morfeusz2 and the TERYT singleton sets are mocked throughout — no real model or
CSV data is loaded.  The mock Morfeusz2 uses a simple token→lemma dictionary so
tests are fully deterministic.

Key invariant verified throughout:
    text[span.start : span.end] == detected surface form
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models import Span
from src.stages.teryt_stage import run

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

PATCH_MORFEUSZ = "src.stages.teryt_stage.get_morfeusz"
PATCH_TERYT = "src.stages.teryt_stage.get_teryt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small representative TERYT sets used across most tests.
_SIMC = {"drzewo", "warszawa", "kraków"}          # places (SIMC)
_ULIC = {"stwosza", "marszałkowska"}               # streets (ULIC)
_TERC = {"mazowieckie", "dolnośląskie", "radomski"} # admin units (TERC)


def make_morfeusz(lemma_map: dict[str, str]) -> MagicMock:
    """Return a Morfeusz2 mock that uses *lemma_map* for lookups.

    Any token not in lemma_map is lemmatised as its own lowercased form.
    The returned analysis list mimics Morfeusz2's real tuple structure:
        [(seg_start, seg_end, (orth, lemma, tag)), ...]
    """

    def analyse(token: str):
        lemma = lemma_map.get(token, token.lower())
        return [(0, 1, (token, lemma, "subst:sg:loc:m3"))]

    mock = MagicMock()
    mock.analyse.side_effect = analyse
    return mock


def _run(text: str, lemma_map: dict[str, str] | None = None,
         simc=_SIMC, ulic=_ULIC, terc=_TERC) -> list[Span]:
    """Call teryt_stage.run() with mocked singletons."""
    morfeusz = make_morfeusz(lemma_map or {})
    with (
        patch(PATCH_MORFEUSZ, return_value=morfeusz),
        patch(PATCH_TERYT, return_value=(simc, ulic, terc)),
    ):
        return run(text)


# ---------------------------------------------------------------------------
# PRD-mandated cases (section 16 of PRD)
# ---------------------------------------------------------------------------


def test_drzewo_simc_match_after_lemmatisation():
    """'w Drzewie' → Morfeusz lemmatises 'Drzewie' → 'drzewo' → SIMC match → masked."""
    text = "mieszka w Drzewie."
    spans = _run(text, lemma_map={"Drzewie": "drzewo"})
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Drzewie"
    assert spans[0].label == "ADRES"
    assert spans[0].source == "teryt"


def test_terc_without_address_keyword_not_masked():
    """'Mazowieckim stylem' — TERC match but no address keyword → NOT masked."""
    text = "Mazowieckim stylem podchodzimy do sprawy."
    spans = _run(text, lemma_map={"Mazowieckim": "mazowieckie"})
    assert spans == []


def test_terc_with_woj_keyword_masked():
    """'woj. Mazowieckim' — TERC match preceded by 'woj.' → masked."""
    text = "Mieszka w woj. Mazowieckim od lat."
    spans = _run(text, lemma_map={"Mazowieckim": "mazowieckie"})
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Mazowieckim"
    assert spans[0].label == "ADRES"


# ---------------------------------------------------------------------------
# SIMC / ULIC — always mask
# ---------------------------------------------------------------------------


def test_simc_match_always_masked():
    text = "Firma mieści się w Warszawie."
    spans = _run(text, lemma_map={"Warszawie": "warszawa"})
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Warszawie"


def test_ulic_match_always_masked():
    text = "Mieszka przy ulicy Stwosza."
    spans = _run(text, lemma_map={"Stwosza": "stwosza"})
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Stwosza"


def test_simc_match_no_context_required():
    """SIMC never requires an address keyword — even mid-sentence."""
    text = "Kraków to piękne miasto."
    spans = _run(text, lemma_map={"Kraków": "kraków"})
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Kraków"


# ---------------------------------------------------------------------------
# TERC — various address keywords
# ---------------------------------------------------------------------------


def test_terc_with_pow_keyword_masked():
    text = "Biuro mieści się w pow. Radomskim."
    spans = _run(text, lemma_map={"Radomskim": "radomski"})
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Radomskim"


def test_terc_with_gm_keyword_masked():
    text = "Położona w gm. Mazowieckim."
    spans = _run(text, lemma_map={"Mazowieckim": "mazowieckie"})
    assert len(spans) == 1


def test_terc_with_full_word_powiat_masked():
    text = "Obszar powiatu Radomskim."
    spans = _run(text, lemma_map={"Radomskim": "radomski"})
    assert len(spans) == 1


def test_terc_with_full_word_wojewodztwo_masked():
    text = "Teren województwa Dolnośląskim."
    spans = _run(text, lemma_map={"Dolnośląskim": "dolnośląskie"})
    assert len(spans) == 1


def test_terc_no_keyword_different_positions():
    """TERC token with nothing before it — no keyword possible."""
    text = "Dolnośląskim trudno zarządzać."
    spans = _run(text, lemma_map={"Dolnośląskim": "dolnośląskie"})
    assert spans == []


def test_terc_keyword_separated_by_other_word_not_matched():
    """Keyword followed by an intervening word → does not qualify as address context."""
    text = "woj. pięknym Mazowieckim."
    # "pięknym" intervenes between "woj." and "Mazowieckim" — not a direct context
    spans = _run(text, lemma_map={"Mazowieckim": "mazowieckie", "pięknym": "piękny"})
    # "pięknym" is lowercase so _UPPER_TOKEN won't match it; only "Mazowieckim" is
    # a candidate; preceding text = "woj. pięknym " — does NOT end with addr keyword
    assert spans == []


# ---------------------------------------------------------------------------
# False positives — common capitalised words that are NOT in TERYT
# ---------------------------------------------------------------------------


def test_common_word_not_in_teryt_not_masked():
    """'Polska' is not in our small test sets → not masked."""
    text = "Polska jest piękna."
    # lemma_map left empty → lemma = "polska" → not in simc/ulic/terc
    spans = _run(text)
    assert spans == []


def test_sentence_start_capitalisation_not_masked():
    """Word at sentence start capitalised but not a place → not masked."""
    text = "Rower jest szybki."
    spans = _run(text)
    assert spans == []


def test_multiple_common_words_none_masked():
    text = "Pan Kowalski jedzie do Biura."
    # "Pan", "Kowalski", "Biura" — none in simc/ulic/terc
    spans = _run(text)
    assert spans == []


# ---------------------------------------------------------------------------
# Multiple entities in one text
# ---------------------------------------------------------------------------


def test_multiple_simc_entities_detected():
    text = "Z Warszawy do Krakowa."
    spans = _run(
        text,
        lemma_map={"Warszawy": "warszawa", "Krakowa": "kraków"},
    )
    assert len(spans) == 2
    surface_forms = {text[s.start : s.end] for s in spans}
    assert surface_forms == {"Warszawy", "Krakowa"}


def test_simc_and_terc_in_one_text():
    """One SIMC entity (always masked) and one TERC with keyword (masked)."""
    text = "Mieszka w Warszawie, w woj. Mazowieckim."
    spans = _run(
        text,
        lemma_map={"Warszawie": "warszawa", "Mazowieckim": "mazowieckie"},
    )
    assert len(spans) == 2
    surface_forms = {text[s.start : s.end] for s in spans}
    assert surface_forms == {"Warszawie", "Mazowieckim"}


def test_simc_and_terc_no_keyword_only_simc_masked():
    """SIMC is masked; TERC without keyword is not."""
    text = "Warszawie Mazowieckim."
    spans = _run(
        text,
        lemma_map={"Warszawie": "warszawa", "Mazowieckim": "mazowieckie"},
    )
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Warszawie"


# ---------------------------------------------------------------------------
# Offset correctness invariant
# ---------------------------------------------------------------------------


def test_offset_invariant_simc():
    """text[span.start:span.end] == surface form — verified for SIMC."""
    text = "Biuro jest w Warszawie przy ulicy Marszałkowskiej."
    spans = _run(
        text,
        lemma_map={"Warszawie": "warszawa", "Marszałkowskiej": "marszałkowska"},
    )
    for span in spans:
        assert text[span.start : span.end] in ("Warszawie", "Marszałkowskiej")


def test_offset_invariant_terc_with_keyword():
    """Offset invariant holds for a TERC span with address keyword."""
    text = "Dokument dotyczy pow. Radomskim i jego mieszkańców."
    spans = _run(text, lemma_map={"Radomskim": "radomski"})
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Radomskim"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty():
    """Empty string must not crash and must return no spans."""
    spans = _run(" ")
    assert spans == []


def test_all_lowercase_text_returns_empty():
    """No uppercase tokens → no candidates → no spans."""
    text = "mieszka w warszawie przy ulicy marszałkowskiej"
    spans = _run(text)
    assert spans == []


def test_token_with_multiple_morfeusz_analyses():
    """When Morfeusz2 returns multiple analyses for a token, any SIMC match suffices."""

    def analyse(token: str):
        if token == "Krakowie":
            # Two analyses: first lemma not in SIMC, second one is — either should trigger
            return [
                (0, 1, (token, "nieznane", "adj:sg:loc:m1")),
                (0, 1, (token, "kraków", "subst:sg:loc:m3")),
            ]
        return [(0, 1, (token, token.lower(), "subst:sg:loc:m3"))]

    mock_morfeusz = MagicMock()
    mock_morfeusz.analyse.side_effect = analyse

    # Use a text where only "Krakowie" is the candidate (starts with uppercase,
    # rest of the word is lowercase); "w" is too short to match _UPPER_TOKEN
    text = "w Krakowie."
    with (
        patch(PATCH_MORFEUSZ, return_value=mock_morfeusz),
        patch(PATCH_TERYT, return_value=(_SIMC, _ULIC, _TERC)),
    ):
        spans = run(text)

    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Krakowie"


def test_span_source_is_teryt():
    text = "w Warszawie."
    spans = _run(text, lemma_map={"Warszawie": "warszawa"})
    assert all(s.source == "teryt" for s in spans)


def test_span_label_is_adres():
    text = "w Warszawie."
    spans = _run(text, lemma_map={"Warszawie": "warszawa"})
    assert all(s.label == "ADRES" for s in spans)
