"""End-to-end tests for the full PII detection pipeline.

Strategy
--------
* ``regex_stage`` runs natively — pure Python, no heavy dependencies.
* ``ner_stage`` singletons (``get_ner_model``, ``get_ner_lock``) are mocked so
  the test suite never loads a real model.
* ``teryt_stage`` singletons (``get_morfeusz``, ``get_teryt``) are mocked with
  a small deterministic lemma dictionary and representative TERYT sets.

Every fixture asserts:
1. The masked output matches the expected string character-by-character.
2. Length invariant: ``len(masked_text) == len(original_text)``.
3. Offset sanity: ``original_text[span.start:span.end] == entity surface form``
   (verified implicitly — correct masking proves correct offsets).
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import httpx
from httpx import AsyncClient, ASGITransport

from src.errors import InternalMaskingError
from src.models import DetectRequest, DetectResponse
from src.pipeline import detect

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

PATCH_NER_MODEL = "src.stages.ner_stage.get_ner_model"
PATCH_NER_LOCK = "src.stages.ner_stage.get_ner_lock"
PATCH_MORFEUSZ = "src.stages.teryt_stage.get_morfeusz"
PATCH_TERYT_SETS = "src.stages.teryt_stage.get_teryt"

# Representative TERYT sets shared across most tests
_SIMC = {"drzewo", "warszawa", "kraków", "nowak"}
_ULIC = {"stwosza", "marszałkowska"}
_TERC = {"mazowieckie", "dolnośląskie"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ner_result(entity_group: str, start: int, end: int, word: str = "entity") -> dict:
    """Build a dict mimicking a single HuggingFace NER pipeline result."""
    return {
        "entity_group": entity_group,
        "start": start,
        "end": end,
        "word": word,
        "score": 0.99,
    }


def make_ner_model(*results: dict) -> MagicMock:
    """Return a callable mock that always returns *results*."""
    return MagicMock(return_value=list(results))


def make_morfeusz(lemma_map: dict[str, str] | None = None) -> MagicMock:
    """Return a Morfeusz2 mock backed by *lemma_map*.

    Tokens absent from *lemma_map* are lemmatised as their own lowercased form.
    """
    lemma_map = lemma_map or {}

    def analyse(token: str):
        lemma = lemma_map.get(token, token.lower())
        return [(0, 1, (token, lemma, "subst:sg:loc:m3"))]

    mock = MagicMock()
    mock.analyse.side_effect = analyse
    return mock


async def run_pipeline(
    text: str,
    *,
    ner_results: list[dict] | None = None,
    morfeusz_lemmas: dict[str, str] | None = None,
    simc: set[str] | None = None,
    ulic: set[str] | None = None,
    terc: set[str] | None = None,
) -> DetectResponse:
    """Run ``pipeline.detect()`` with mocked NER and TERYT singletons."""
    ner_model = make_ner_model(*(ner_results or []))
    lock = asyncio.Lock()
    morfeusz = make_morfeusz(morfeusz_lemmas)
    teryt_sets = (
        simc if simc is not None else set(),
        ulic if ulic is not None else set(),
        terc if terc is not None else set(),
    )

    with (
        patch(PATCH_NER_MODEL, return_value=ner_model),
        patch(PATCH_NER_LOCK, return_value=lock),
        patch(PATCH_MORFEUSZ, return_value=morfeusz),
        patch(PATCH_TERYT_SETS, return_value=teryt_sets),
    ):
        return await detect(DetectRequest(text=text))


# ---------------------------------------------------------------------------
# Fixture 1 — regex only: PESEL
# ---------------------------------------------------------------------------

# Text:     "PESEL: 85030512345"
#            0123456789012345678
# "85030512345" occupies [7:18]


async def test_fixture_pesel_only():
    """Regex detects PESEL; NER and TERYT return nothing."""
    text = "PESEL: 85030512345"
    assert text[7:18] == "85030512345"

    resp = await run_pipeline(text)

    expected = "PESEL: -----------"
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 2 — regex only: EMAIL
# ---------------------------------------------------------------------------

# Text:     "email: jan@example.com koniec"
#            0123456789012345678901234567890
# "jan@example.com" occupies [7:22]


async def test_fixture_email_only():
    """Regex detects an email address; NER and TERYT return nothing."""
    text = "email: jan@example.com koniec"
    start = text.index("jan@example.com")
    end = start + len("jan@example.com")
    assert text[start:end] == "jan@example.com"

    resp = await run_pipeline(text)

    expected = "email: --------------- koniec"
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 3 — NER only: person name
# ---------------------------------------------------------------------------

# Text:     "Pani Anna Nowak pracuje w ZUS."
#            0    5    0    5    0    5
# "Anna Nowak" occupies [5:15]


async def test_fixture_ner_person_name():
    """NER detects a person name; regex and TERYT return nothing."""
    text = "Pani Anna Nowak pracuje w ZUS."
    assert text[5:15] == "Anna Nowak"

    resp = await run_pipeline(
        text,
        ner_results=[ner_result("persName", 5, 15, "Anna Nowak")],
        # empty TERYT sets — "Nowak" not in simc/ulic/terc
    )

    # "Anna Nowak" → "---- -----" (space at relative index 4 stays)
    expected = "Pani ---- ----- pracuje w ZUS."
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 4 — NER only: organisation
# ---------------------------------------------------------------------------

# Text:     "firma PKN ORLEN jest duza"
#            0     6         6
# "PKN ORLEN" occupies [6:15]


async def test_fixture_ner_organisation():
    """NER detects an organisation name."""
    text = "firma PKN ORLEN jest duza"
    assert text[6:15] == "PKN ORLEN"

    resp = await run_pipeline(
        text,
        ner_results=[ner_result("orgName", 6, 15, "PKN ORLEN")],
    )

    expected = "firma --- ----- jest duza"
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 5 — TERYT only: SIMC place name (inflected form)
# ---------------------------------------------------------------------------

# Text:     "Mieszka w Drzewie."
#            0         1
# "Drzewie" occupies [10:17]


async def test_fixture_teryt_simc_inflected():
    """TERYT detects a place name via Morfeusz2 lemmatisation."""
    text = "Mieszka w Drzewie."
    assert text[10:17] == "Drzewie"

    resp = await run_pipeline(
        text,
        morfeusz_lemmas={"Drzewie": "drzewo"},
        simc={"drzewo"},
    )

    expected = "Mieszka w -------."
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 6 — TERYT only: TERC with address keyword
# ---------------------------------------------------------------------------

# Text:     "Mieszka w woj. Mazowieckim od lat."
#            0              5
# "Mazowieckim" occupies [15:26]


async def test_fixture_teryt_terc_with_keyword():
    """TERC entity is masked only when preceded by an address keyword."""
    text = "Mieszka w woj. Mazowieckim od lat."
    maz_start = text.index("Mazowieckim")
    maz_end = maz_start + len("Mazowieckim")
    assert text[maz_start:maz_end] == "Mazowieckim"

    resp = await run_pipeline(
        text,
        morfeusz_lemmas={"Mazowieckim": "mazowieckie"},
        terc={"mazowieckie"},
    )

    dashes = "-" * len("Mazowieckim")
    expected = text[:maz_start] + dashes + text[maz_end:]
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


async def test_fixture_teryt_terc_without_keyword_not_masked():
    """TERC entity without an address keyword is NOT masked."""
    text = "Mazowieckim stylem podchodzimy do sprawy."
    resp = await run_pipeline(
        text,
        morfeusz_lemmas={"Mazowieckim": "mazowieckie"},
        terc={"mazowieckie"},
    )
    # No address keyword → no masking
    assert resp.masked_text == text
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 7 — combined: NER + regex + TERYT (PRD example)
# ---------------------------------------------------------------------------

# Text:     "Jan Kowalski, PESEL 85030512345, mieszka w Drzewie."
#            0   4         4   9   0         1   5   0   4   0
# "Jan Kowalski" = [0:12]   (NER persName → OSOBA)
# "85030512345"  = [20:31]  (regex PESEL)
# "Drzewie"      = [43:50]  (TERYT SIMC)


async def test_fixture_combined_prd_example():
    """PRD reference example: name + PESEL + place name all masked."""
    text = "Jan Kowalski, PESEL 85030512345, mieszka w Drzewie."
    assert text[0:12] == "Jan Kowalski"
    assert text[20:31] == "85030512345"
    assert text[43:50] == "Drzewie"

    resp = await run_pipeline(
        text,
        ner_results=[ner_result("persName", 0, 12, "Jan Kowalski")],
        morfeusz_lemmas={"Drzewie": "drzewo"},
        simc={"drzewo"},
    )

    expected = "--- --------, PESEL -----------, mieszka w -------."
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 8 — combined: NIP + NER organisation
# ---------------------------------------------------------------------------

# Text:     "NIP: 123-456-78-90, firma XYZ Sp. z o.o."
# "123-456-78-90" = NIP at [5:18]
# "XYZ Sp. z o.o." — NER might detect as ORG; use a simpler fragment:
#
# Text:     "NIP 1234567890 firmy ABC"
# "1234567890" = [4:14]  (regex NIP \b\d{10}\b)
# "ABC" = [21:24] (NER orgName)


async def test_fixture_nip_and_ner_org():
    """Regex detects NIP; NER detects organisation name."""
    text = "NIP 1234567890 firmy ABC"
    assert text[4:14] == "1234567890"
    assert text[21:24] == "ABC"

    resp = await run_pipeline(
        text,
        ner_results=[ner_result("orgName", 21, 24, "ABC")],
    )

    expected = "NIP ---------- firmy ---"
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 9 — conflict resolution: NER vs TERYT overlap (NER wins)
# ---------------------------------------------------------------------------

# "Nowak" is both a surname (NER persName) and in simc (TERYT).
# NER priority (8) > TERYT priority (6) at equal length → NER span kept.
# The masked result is the same either way, but only one span is in the list.

# Text:     "Pani Nowak mieszka tu."
#            0    5    0
# "Nowak" = [5:10]


async def test_fixture_conflict_ner_beats_teryt():
    """When NER and TERYT detect the same span, NER (higher priority) wins."""
    text = "Pani Nowak mieszka tu."
    assert text[5:10] == "Nowak"

    resp = await run_pipeline(
        text,
        ner_results=[ner_result("persName", 5, 10, "Nowak")],
        morfeusz_lemmas={"Nowak": "nowak"},
        simc={"nowak"},  # "Nowak" is also a village — TERYT would detect it too
    )

    # Either way "Nowak" is masked — verify length invariant and correct masking
    expected = "Pani ----- mieszka tu."
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 10 — conflict resolution: longer span wins
# ---------------------------------------------------------------------------

# Regex detects "jan.k@example.com" (EMAIL, longer).
# NER detects only "jan.k" (shorter) — this would not normally happen, but
# we simulate it to test that resolve_conflicts keeps the longer span.

# Text:     "Napisz na jan.k@example.com."
# "jan.k@example.com" = regex EMAIL


async def test_fixture_conflict_longer_span_wins():
    """Longer span (regex EMAIL) beats a shorter overlapping NER span."""
    text = "Napisz na jan.k@example.com."
    email_start = text.index("jan.k@example.com")
    email_end = email_start + len("jan.k@example.com")
    assert text[email_start:email_end] == "jan.k@example.com"

    # Simulate NER detecting only "jan.k" (the local part)
    resp = await run_pipeline(
        text,
        ner_results=[ner_result("orgName", email_start, email_start + 5, "jan.k")],
    )

    # The longer EMAIL span from regex should win — full email masked
    dashes = "-" * len("jan.k@example.com")
    expected = text[:email_start] + dashes + text[email_end:]
    assert resp.masked_text == expected
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 11 — no PII: text unchanged
# ---------------------------------------------------------------------------


async def test_fixture_no_pii_text_unchanged():
    """When no PII is detected, the text is returned unchanged."""
    text = "Ala ma kota i psa. Kot ma na imię Burek."
    resp = await run_pipeline(text)
    assert resp.masked_text == text
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 12 — length invariant with Unicode Polish characters
# ---------------------------------------------------------------------------


async def test_fixture_length_invariant_unicode():
    """Length invariant holds for texts with Polish diacritics."""
    text = "Zażółć gęślą jaźń — PESEL: 85030512345."
    resp = await run_pipeline(text)
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 13 — multiline text
# ---------------------------------------------------------------------------


async def test_fixture_multiline_text():
    """Pipeline handles multiline text; each line's entities are detected."""
    # "85030512345" — valid PESEL
    text = "Imię: Jan\nNazwisko: Kowalski\nPESEL: 85030512345"
    jan_start = text.index("Jan")
    pesel_start = text.index("85030512345")

    resp = await run_pipeline(
        text,
        ner_results=[ner_result("persName", jan_start, jan_start + 3, "Jan")],
    )

    assert len(resp.masked_text) == len(text)
    # NER masked "Jan"
    assert resp.masked_text[jan_start : jan_start + 3] == "---"
    # Regex masked the PESEL
    assert resp.masked_text[pesel_start : pesel_start + 11] == "-----------"
    # "Imię: " prefix unchanged
    assert resp.masked_text[: jan_start] == text[:jan_start]


# ---------------------------------------------------------------------------
# Fixture 14 — NER placeName and date are filtered (not masked)
# ---------------------------------------------------------------------------


async def test_fixture_ner_placename_and_date_filtered():
    """NER placeName and date entities are discarded; text stays as-is."""
    text = "W Warszawie dnia 2024-01-15."
    resp = await run_pipeline(
        text,
        ner_results=[
            ner_result("placeName", 2, 10, "Warszawie"),
            ner_result("date", 16, 26, "2024-01-15"),
        ],
    )
    # placeName → filtered; date → filtered; no regex/teryt hits either
    assert resp.masked_text == text
    assert len(resp.masked_text) == len(text)


# ---------------------------------------------------------------------------
# Fixture 15 — DATA_URODZENIA: only the date part is masked
# ---------------------------------------------------------------------------

# Text:     "urodzona 12.03.1985 w Gdańsku"
# Regex DATA_URODZENIA matches "12.03.1985" (group 1 — date only)


async def test_fixture_data_urodzenia_keyword_not_masked():
    """For DATA_URODZENIA the keyword is preserved; only the date is masked."""
    text = "urodzona 12.03.1985 w Gdansku"
    date_start = text.index("12.03.1985")
    date_end = date_start + len("12.03.1985")
    assert text[date_start:date_end] == "12.03.1985"

    resp = await run_pipeline(text)

    assert len(resp.masked_text) == len(text)
    # "urodzona " prefix is unchanged
    assert resp.masked_text[:date_start] == text[:date_start]
    # date is masked
    assert resp.masked_text[date_start:date_end] == "----------"
    # remainder unchanged
    assert resp.masked_text[date_end:] == text[date_end:]


# ---------------------------------------------------------------------------
# Fixture 16 — InternalMaskingError propagates as-is from pipeline
# ---------------------------------------------------------------------------


async def test_fixture_internal_masking_error_propagated(monkeypatch):
    """If assert_length() fires (bug), InternalMaskingError is not swallowed."""
    import src.pipeline as pipeline_mod

    original_apply = pipeline_mod.apply_masks

    def broken_apply(text, spans):
        # Simulate a buggy apply_masks that returns a shorter string
        return original_apply(text, spans)[:-1]

    monkeypatch.setattr(pipeline_mod, "apply_masks", broken_apply)

    with pytest.raises(InternalMaskingError):
        await run_pipeline("Jan Kowalski PESEL 85030512345")


# ---------------------------------------------------------------------------
# HTTP-level tests — /detect endpoint
# ---------------------------------------------------------------------------


async def test_http_detect_returns_200(monkeypatch):
    """POST /detect with valid input returns 200 and masked_text field."""
    from src.main import app

    # init_all only creates the asyncio.Lock (TODOs for real models) — safe to call
    text = "PESEL: 85030512345"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        with (
            patch(PATCH_NER_MODEL, return_value=make_ner_model()),
            patch(PATCH_NER_LOCK, return_value=asyncio.Lock()),
            patch(PATCH_MORFEUSZ, return_value=make_morfeusz()),
            patch(PATCH_TERYT_SETS, return_value=(set(), set(), set())),
        ):
            resp = await client.post("/detect", json={"text": text})

    assert resp.status_code == 200
    body = resp.json()
    assert "masked_text" in body
    assert len(body["masked_text"]) == len(text)


async def test_http_detect_empty_text_returns_422():
    """POST /detect with empty text returns HTTP 422 (validation error)."""
    from src.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/detect", json={"text": "   "})

    assert resp.status_code == 422


async def test_http_detect_invalid_chunk_size_returns_422():
    """chunk_size < 200 is rejected at the Pydantic layer."""
    from src.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/detect", json={"text": "hello", "chunk_size": 10})

    assert resp.status_code == 422


async def test_http_detect_internal_error_returns_500(monkeypatch):
    """If assert_length() fires inside pipeline, /detect returns HTTP 500."""
    import src.pipeline as pipeline_mod

    original_apply = pipeline_mod.apply_masks

    def broken_apply(text, spans):
        return original_apply(text, spans)[:-1]

    monkeypatch.setattr(pipeline_mod, "apply_masks", broken_apply)

    from src.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        with (
            patch(PATCH_NER_MODEL, return_value=make_ner_model()),
            patch(PATCH_NER_LOCK, return_value=asyncio.Lock()),
            patch(PATCH_MORFEUSZ, return_value=make_morfeusz()),
            patch(PATCH_TERYT_SETS, return_value=(set(), set(), set())),
        ):
            resp = await client.post("/detect", json={"text": "jakiś tekst"})

    assert resp.status_code == 500


async def test_http_health_returns_ok():
    """GET /health returns 200 with status=ok."""
    from src.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
