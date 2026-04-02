"""Tests for stages/ner_stage.py.

The real FastPDN model is never loaded here.  All tests mock get_ner_model()
and get_ner_lock() at the module level inside ner_stage so that:
  - no ML dependencies are required to run the test suite
  - the logic under test is the label mapping and offset translation,
    not the model's accuracy

Key invariant verified throughout:
    full_text[span.start : span.end] == detected entity surface form
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.chunking import Chunk
from src.models import Span
from src.stages.ner_stage import NER_LABEL_MAP, run, run_chunked

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

PATCH_MODEL = "src.stages.ner_stage.get_ner_model"
PATCH_LOCK = "src.stages.ner_stage.get_ner_lock"


def ner_result(entity_group: str, start: int, end: int, word: str = "dummy") -> dict:
    """Build a dict that mimics a single HuggingFace NER pipeline result."""
    return {
        "entity_group": entity_group,
        "start": start,
        "end": end,
        "word": word,
        "score": 0.99,
    }


def make_model(*results: dict) -> MagicMock:
    """Return a callable mock that returns *results* on every call."""
    return MagicMock(return_value=list(results))


def make_lock() -> asyncio.Lock:
    return asyncio.Lock()


# ---------------------------------------------------------------------------
# NER_LABEL_MAP sanity checks (no I/O)
# ---------------------------------------------------------------------------


def test_label_map_persname_is_list():
    assert isinstance(NER_LABEL_MAP["persName"], list)


def test_label_map_orgname_is_str():
    assert NER_LABEL_MAP["orgName"] == "ORG"


def test_label_map_placename_is_none():
    assert NER_LABEL_MAP["placeName"] is None


def test_label_map_date_is_none():
    assert NER_LABEL_MAP["date"] is None


# ---------------------------------------------------------------------------
# run() — label mapping
# ---------------------------------------------------------------------------


async def test_run_persname_mapped_to_osoba():
    lock = make_lock()
    model = make_model(ner_result("persName", 0, 12, "Jan Kowalski"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run("Jan Kowalski")
    assert len(spans) == 1
    assert spans[0].label == "OSOBA"
    assert spans[0].source == "ner"


async def test_run_orgname_mapped_to_org():
    text = "firma PKN ORLEN jest duza"
    lock = make_lock()
    model = make_model(ner_result("orgName", 6, 15, "PKN ORLEN"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run(text)
    assert len(spans) == 1
    assert spans[0].label == "ORG"
    assert text[spans[0].start : spans[0].end] == "PKN ORLEN"


async def test_run_placename_filtered_out():
    lock = make_lock()
    model = make_model(ner_result("placeName", 3, 11, "Warszawa"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run("w Warszawie")
    assert spans == []


async def test_run_date_filtered_out():
    lock = make_lock()
    model = make_model(ner_result("date", 0, 10, "2023-01-01"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run("2023-01-01 jest wazna data")
    assert spans == []


async def test_run_unknown_entity_group_filtered():
    lock = make_lock()
    model = make_model(ner_result("unknownXYZ", 0, 5, "hello"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run("hello world")
    assert spans == []


async def test_run_empty_model_results():
    lock = make_lock()
    model = MagicMock(return_value=[])
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run("tekst bez zadnych encji")
    assert spans == []


async def test_run_mixed_entities_only_mapped_returned():
    """placeName and date are filtered; persName and orgName are kept."""
    lock = make_lock()
    model = make_model(
        ner_result("persName", 0, 12, "Jan Kowalski"),
        ner_result("placeName", 14, 22, "Warszawa"),
        ner_result("orgName", 24, 33, "ZUS"),
        ner_result("date", 35, 45, "2020-05-01"),
    )
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run("Jan Kowalski, Warszawa, ZUS, 2020-05-01")
    assert len(spans) == 2
    labels = {s.label for s in spans}
    assert labels == {"OSOBA", "ORG"}


# ---------------------------------------------------------------------------
# run() — offset correctness
# ---------------------------------------------------------------------------


async def test_run_offsets_match_text_slice():
    """span.start:span.end must select the entity text."""
    text = "Pani Anna Nowak pracuje w ZUS."
    anna_start = text.index("Anna Nowak")
    anna_end = anna_start + len("Anna Nowak")
    lock = make_lock()
    model = make_model(ner_result("persName", anna_start, anna_end, "Anna Nowak"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run(text)
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Anna Nowak"


async def test_run_preserves_start_and_end_from_model():
    """run() must not shift offsets — they are local to the text argument."""
    text = "X" * 20 + "Firma ABC" + "Y" * 10
    lock = make_lock()
    model = make_model(ner_result("orgName", 20, 29, "Firma ABC"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run(text)
    assert spans[0].start == 20
    assert spans[0].end == 29
    assert text[spans[0].start : spans[0].end] == "Firma ABC"


# ---------------------------------------------------------------------------
# run_chunked() — global offset translation
# ---------------------------------------------------------------------------


async def test_run_chunked_zero_offset_chunk():
    text = "Jan Kowalski pracuje."
    chunk = Chunk(text=text, offset_start=0)
    lock = make_lock()
    model = make_model(ner_result("persName", 0, 12, "Jan Kowalski"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run_chunked(text, [chunk])
    assert len(spans) == 1
    assert spans[0].start == 0
    assert spans[0].end == 12
    assert text[spans[0].start : spans[0].end] == "Jan Kowalski"


async def test_run_chunked_nonzero_offset_translates_correctly():
    """Local offset [0, 12] in a chunk starting at 100 → global [100, 112]."""
    prefix = "A" * 100
    chunk_text = "Jan Kowalski mieszka."
    full_text = prefix + chunk_text
    chunk = Chunk(text=chunk_text, offset_start=100)
    lock = make_lock()
    model = make_model(ner_result("persName", 0, 12, "Jan Kowalski"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run_chunked(full_text, [chunk])
    assert len(spans) == 1
    assert spans[0].start == 100
    assert spans[0].end == 112
    assert full_text[spans[0].start : spans[0].end] == "Jan Kowalski"


async def test_run_chunked_offset_invariant_holds():
    """full_text[s.start:s.end] must equal the entity for any non-zero offset."""
    full_text = "Przykładowy tekst z imieniem Anna Nowak w środku."
    anna_start = full_text.index("Anna Nowak")
    anna_end = anna_start + len("Anna Nowak")
    chunk = Chunk(text=full_text, offset_start=0)
    lock = make_lock()
    model = make_model(ner_result("persName", anna_start, anna_end, "Anna Nowak"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run_chunked(full_text, [chunk])
    assert full_text[spans[0].start : spans[0].end] == "Anna Nowak"


async def test_run_chunked_two_chunks_independent_offsets():
    """Each chunk contributes spans with its own offset_start applied."""
    chunk1 = Chunk(text="Jan Kowalski jest lekarzem.", offset_start=0)
    chunk2 = Chunk(text="Anna Nowak to pielęgniarka.", offset_start=50)

    call_count = 0

    def model_side_effect(text):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # chunk1: "Jan Kowalski" at local [0, 12]
            return [ner_result("persName", 0, 12, "Jan Kowalski")]
        else:
            # chunk2: "Anna Nowak" at local [0, 10]
            return [ner_result("persName", 0, 10, "Anna Nowak")]

    model = MagicMock(side_effect=model_side_effect)
    lock = make_lock()
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        full_text = chunk1.text + " " * 23 + chunk2.text
        spans = await run_chunked(full_text, [chunk1, chunk2])

    assert len(spans) == 2
    # chunk1 entity: global 0+0=0 .. 12+0=12
    assert spans[0].start == 0
    assert spans[0].end == 12
    # chunk2 entity: global 0+50=50 .. 10+50=60
    assert spans[1].start == 50
    assert spans[1].end == 60


async def test_run_chunked_empty_chunk_list_returns_empty():
    model = MagicMock(return_value=[])
    lock = make_lock()
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run_chunked("jakiś tekst", [])
    assert spans == []
    model.assert_not_called()


async def test_run_chunked_filters_carry_through_chunks():
    """placeName results in a chunk still get filtered even via run_chunked."""
    chunk = Chunk(text="w Krakowie mieszka.", offset_start=20)
    lock = make_lock()
    model = make_model(ner_result("placeName", 2, 10, "Krakowie"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run_chunked("X" * 20 + chunk.text, [chunk])
    assert spans == []


async def test_run_chunked_source_field_is_ner():
    chunk = Chunk(text="Anna Kowalska.", offset_start=5)
    lock = make_lock()
    model = make_model(ner_result("persName", 0, 13, "Anna Kowalska"))
    with patch(PATCH_MODEL, return_value=model), patch(PATCH_LOCK, return_value=lock):
        spans = await run_chunked("XXXXX" + chunk.text, [chunk])
    assert spans[0].source == "ner"
