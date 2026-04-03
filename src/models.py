from pydantic import BaseModel, field_validator
from typing import Literal


class DetectRequest(BaseModel):
    text: str
    language: Literal["pl", "en"] = "pl"
    chunk_size: int = 1500  # per-request override; short texts get 1 chunk automatically

    @field_validator("text")
    @classmethod
    def text_must_be_nonempty(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("text must be a string")
        if len(v.strip()) == 0:
            raise ValueError("text must not be empty")
        return v

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_in_range(cls, v: int) -> int:
        if v < 200:
            raise ValueError("chunk_size must be >= 200")
        if v > 10_000:
            raise ValueError("chunk_size must be <= 10000")
        return v


class DetectResponse(BaseModel):
    masked_text: str
    detected: bool


class Span(BaseModel):
    """Absolute offset in full_text. Used internally between pipeline stages."""

    start: int  # inclusive
    end: int  # exclusive — like Python slice: text[start:end]
    label: str  # PESEL | NIP | EMAIL | TELEFON | IMIE | NAZWISKO | ORG | ADRES |
    #            DATA_URODZENIA | KOD_POCZTOWY | NR_REJESTRACYJNY | NR_KARTY
    source: str  # "regex" | "ner" | "teryt" — for debugging and resolve_conflicts
