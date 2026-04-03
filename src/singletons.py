"""Singleton resources loaded once at startup via FastAPI lifespan.

All heavy resources (NER model, Morfeusz2, TERYT sets) are initialised in
init_all() and accessed through the getters below.  Nothing outside this
module should assign to the module-level variables directly.

Never load a model or read a CSV inside a request handler.
"""

import asyncio

_ner_model = None
_ner_lock: asyncio.Lock | None = None
_morfeusz = None
_teryt_simc: set[str] = set()
_teryt_ulic: set[str] = set()
_teryt_terc: set[str] = set()


def get_ner_model():
    return _ner_model


def get_ner_lock() -> asyncio.Lock | None:
    return _ner_lock


def get_morfeusz():
    return _morfeusz


def get_teryt() -> tuple[set[str], set[str], set[str]]:
    """Returns (simc, ulic, terc) sets of lowercase lemmatised place names."""
    return _teryt_simc, _teryt_ulic, _teryt_terc


async def init_all() -> None:
    """Load all heavy resources.  Called once from the FastAPI lifespan.

    /health returns 503 until this function completes.
    """
    global _ner_model, _ner_lock, _morfeusz, _teryt_simc, _teryt_ulic, _teryt_terc

    from transformers import pipeline as hf_pipeline
    _ner_model = hf_pipeline(
        "ner",
        model="clarin-pl/FastPDN",
        aggregation_strategy="simple",
        device=0,
    )

    # asyncio.Lock must be created inside the running event loop
    _ner_lock = asyncio.Lock()

    import morfeusz2
    _morfeusz = morfeusz2.Morfeusz()

    import pandas as pd
    simc = pd.read_csv("data/teryt/SIMC.csv", sep=";", encoding="utf-8")
    _teryt_simc = set(simc["NAZWA"].str.lower().str.strip())
    ulic = pd.read_csv("data/teryt/ULIC.csv", sep=";", encoding="utf-8")
    _teryt_ulic = set(ulic["NAZWA_1"].str.lower().str.strip())
    terc = pd.read_csv("data/teryt/TERC.csv", sep=";", encoding="utf-8")
    _teryt_terc = set(terc["NAZWA"].str.lower().str.strip())
