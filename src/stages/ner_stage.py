"""NER stage — clarin-pl/FastPDN entity detection.

Singletons are assumed to already be initialised (init_all() called at startup).
Never load models here.

Public interface:
    run(text)              — async; acquires asyncio.Lock; uses run_in_executor
    run_chunked(text, chunks) — iterates chunks, translates local → global offsets
"""

import asyncio

from ..chunking import Chunk
from ..models import Span
from ..singletons import get_ner_lock, get_ner_model

# FastPDN entity groups → internal labels.
# None = filtered out (handled by a different stage).
# A list value means the entity covers multiple roles — collapsed to "OSOBA".
NER_LABEL_MAP: dict[str, str | list[str] | None] = {
    "persName":  ["IMIE", "NAZWISKO"],  # full person name — mask as one span
    "orgName":   "ORG",
    "placeName": None,   # ignored — addresses handled by TERYT stage (more precise)
    "date":      None,   # ignored — birth dates handled by regex with keyword context
}


async def run(text: str) -> list[Span]:
    """Run NER inference on *text* and return spans with local offsets.

    Offsets are relative to the start of *text* (which may be a chunk).
    The caller is responsible for translating to global offsets.

    Thread safety: acquires the singleton asyncio.Lock so that only one
    inference runs at a time — PyTorch is not safe for concurrent use on a
    single model instance.  run_in_executor is used so the FastAPI event loop
    is not blocked during the (CPU-bound) inference.
    """
    model = get_ner_model()
    lock = get_ner_lock()

    def _infer() -> list[dict]:
        return model(text)

    async with lock:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _infer)

    spans: list[Span] = []
    for r in results:
        label_val = NER_LABEL_MAP.get(r["entity_group"])
        if label_val is None:
            continue
        spans.append(Span(
            start=r["start"],
            end=r["end"],
            label=label_val if isinstance(label_val, str) else "OSOBA",
            source="ner",
        ))
    return spans


async def run_chunked(text: str, chunks: list[Chunk]) -> list[Span]:
    """Run NER over all *chunks* and translate local offsets to global offsets.

    Each chunk's ``offset_start`` is added to every span's start/end so the
    returned spans are absolute in *text*.

    Invariant that must hold for every returned span *s*:
        text[s.start : s.end] == the detected entity surface form
    """
    all_spans: list[Span] = []
    for chunk in chunks:
        local_spans = await run(chunk.text)
        for s in local_spans:
            all_spans.append(Span(
                start=s.start + chunk.offset_start,
                end=s.end + chunk.offset_start,
                label=s.label,
                source=s.source,
            ))
    return all_spans
