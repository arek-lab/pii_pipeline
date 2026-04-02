"""Orchestrates the three PII-detection stages and applies masks.

Stage execution order:
  1. regex_stage  — fast, synchronous, runs on full text
  2. ner_stage    — async, behind asyncio.Lock, runs on chunks
  3. teryt_stage  — synchronous, runs on full text (no chunking needed)

Every stage receives the *original* text, never a partially masked copy.
resolve_conflicts() is called once on the combined span list before masking.
"""

from .chunking import chunk_with_offsets, resolve_conflicts
from .errors import InternalMaskingError
from .masking import apply_masks, assert_length
from .models import DetectRequest, DetectResponse, Span
from .stages import ner_stage, regex_stage, teryt_stage


async def detect(request: DetectRequest) -> DetectResponse:
    text = request.text
    # Chunking is used exclusively for NER (FastPDN token limit).
    # Regex and TERYT run on the full text without chunking.
    chunks = chunk_with_offsets(text, max_chars=request.chunk_size)

    # Stage 1 — regex (synchronous, full text — no chunking needed)
    regex_spans: list[Span] = regex_stage.run(text)

    # Stage 2 — NER (async, chunked, behind asyncio.Lock)
    ner_spans: list[Span] = await ner_stage.run_chunked(text, chunks)

    # Stage 3 — TERYT (synchronous, full text — no chunking needed)
    teryt_spans: list[Span] = teryt_stage.run(text)

    all_spans = resolve_conflicts(regex_spans + ner_spans + teryt_spans)

    masked = apply_masks(text, all_spans)
    assert_length(text, masked)  # raises InternalMaskingError on length violation

    return DetectResponse(masked_text=masked)
