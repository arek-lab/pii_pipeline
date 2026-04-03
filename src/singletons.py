"""Singleton resources loaded once at startup via FastAPI lifespan.

All heavy resources (NER model, Morfeusz2, TERYT sets) are initialised in
init_all() and accessed through the getters below.  Nothing outside this
module should assign to the module-level variables directly.

Never load a model or read a CSV inside a request handler.

TERYT pickle cache
------------------
Lemmatising ~200k street names via Morfeusz2 takes 10–30 s, which is
painful during development with uvicorn --reload.  To avoid this, the
computed sets are serialised to data/teryt/.teryt_cache.pkl together with
a manifest of source-file mtimes.  On subsequent startups the cache is
loaded in <1 s unless any of the three CSV files has been modified.

To force a full rebuild (e.g. after updating TERYT data) either:
  - touch / replace the CSV files, or
  - delete data/teryt/.teryt_cache.pkl manually.
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


def _lemmatise_series(morf, series) -> set[str]:
    """Return a set of lowercase lemmas (+ original forms) for every name in *series*.

    For each name Morfeusz2 may propose multiple lemma candidates — all are
    added to the set.  The original lowercased form is always included as a
    fallback for proper nouns that Morfeusz2 does not recognise.

    This ensures that both the nominative lemma ("mickiewicz") and the
    genitive form stored in TERYT ("mickiewicza") end up in the lookup set,
    so the matcher in teryt.py — which lemmatises tokens from the input text —
    finds a hit regardless of which direction the mismatch goes.
    """
    result: set[str] = set()
    for name in series.dropna().str.strip():
        if not name:
            continue
        # Always keep the original form (covers unknown proper nouns)
        result.add(name.lower())
        # Add every lemma candidate Morfeusz2 proposes
        for analysis in morf.analyse(name):
            result.add(analysis[2][1].lower())
    return result


_TERYT_CSV_PATHS = {
    "simc": "data/teryt/simc.csv",
    "ulic": "data/teryt/ulic.csv",
    "terc": "data/teryt/terc.csv",
}
_TERYT_CACHE_PATH = "data/teryt/.teryt_cache.pkl"


def _csv_mtime_manifest() -> dict[str, float]:
    """Return a dict of {csv_key: mtime} for all TERYT source files."""
    import os
    return {key: os.path.getmtime(path) for key, path in _TERYT_CSV_PATHS.items()}


def _load_teryt_cache() -> tuple[set[str], set[str], set[str]] | None:
    """Return cached (simc, ulic, terc) sets if the cache is fresh, else None."""
    import os
    import pickle

    if not os.path.exists(_TERYT_CACHE_PATH):
        return None

    try:
        with open(_TERYT_CACHE_PATH, "rb") as fh:
            cached = pickle.load(fh)
    except Exception:
        # Corrupted cache — rebuild silently.
        return None

    if cached.get("manifest") != _csv_mtime_manifest():
        return None

    return cached["simc"], cached["ulic"], cached["terc"]


def _save_teryt_cache(simc: set[str], ulic: set[str], terc: set[str]) -> None:
    """Persist sets to pickle so subsequent startups skip lemmatisation."""
    import pickle

    payload = {
        "manifest": _csv_mtime_manifest(),
        "simc": simc,
        "ulic": ulic,
        "terc": terc,
    }
    # Write to a temp file first to avoid a half-written cache on crash.
    tmp = _TERYT_CACHE_PATH + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    import os
    os.replace(tmp, _TERYT_CACHE_PATH)


def _build_teryt_sets(morf) -> tuple[set[str], set[str], set[str]]:
    """Read CSVs and lemmatise all TERYT names.  Called only on cache miss."""
    import pandas as pd

    # --- SIMC (miejscowości) ---
    simc_df = pd.read_csv(_TERYT_CSV_PATHS["simc"], sep=";", encoding="utf-8")
    simc = _lemmatise_series(morf, simc_df["NAZWA"])

    # --- ULIC (ulice) ---
    # NAZWA_1 — człon główny, np. "Mickiewicza", "Trzydziestolecia"
    # NAZWA_2 — człon dodatkowy, np. "Adama" (imię patrona); często NaN
    ulic_df = pd.read_csv(_TERYT_CSV_PATHS["ulic"], sep=";", encoding="utf-8")
    ulic = (
        _lemmatise_series(morf, ulic_df["NAZWA_1"])
        | _lemmatise_series(morf, ulic_df["NAZWA_2"])
        # Pełna nazwa złożona ("mickiewicza adama") na wypadek gdy NER zwróci
        # cały dwuczłonowy span jako jeden token w przyszłych rozszerzeniach.
        | _lemmatise_series(
            morf,
            (ulic_df["NAZWA_1"].fillna("") + " " + ulic_df["NAZWA_2"].fillna("")).str.strip(),
        )
    )

    # --- TERC (gminy, powiaty, województwa) ---
    terc_df = pd.read_csv(_TERYT_CSV_PATHS["terc"], sep=";", encoding="utf-8")
    terc = _lemmatise_series(morf, terc_df["NAZWA"])

    return simc, ulic, terc


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

    # Try the pickle cache first; fall back to full lemmatisation on miss.
    cached = _load_teryt_cache()
    if cached is not None:
        _teryt_simc, _teryt_ulic, _teryt_terc = cached
    else:
        _teryt_simc, _teryt_ulic, _teryt_terc = _build_teryt_sets(_morfeusz)
        _save_teryt_cache(_teryt_simc, _teryt_ulic, _teryt_terc)