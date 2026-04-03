# PRD — `pii-pipeline`

**Wersja:** 1.1  
**Status:** Draft  
**Scope:** Tylko serwis `pii-pipeline` — detekcja i maskowanie PII w plain text.

---

## 1. Cel i zakres

`pii-pipeline` przyjmuje plain text (UTF-8), wykrywa dane osobowe (PII) i zwraca string o **identycznej długości** jak wejście, gdzie każda litera zamaskowanej encji zastąpiona jest myślnikiem `-`. Spacje wewnątrz encji pozostają niezmienione.

Serwis jest wywoływany wyłącznie przez `doc-masking-service/gateway` przez wewnętrzną sieć Docker. Nie jest dostępny publicznie.

---

## 2. Kontrakt API

### `POST /detect`

**Request:**
```json
{
  "text": "Jan Kowalski, PESEL 85030512345, mieszka w Drzewie.",
  "language": "pl"
}
```

**Response:**
```json
{
  "masked_text": "--- --------, PESEL -----------, mieszka w ------."
}
```

**Niezmienniki (weryfikowane przed zwróceniem odpowiedzi):**
- `len(masked_text) == len(text)` — zawsze, bez wyjątku
- Znaki niebędące literą encji są identyczne jak w oryginale
- Spacje wewnątrz wielowyrazowych encji pozostają spacjami

Jeśli warunek długości nie jest spełniony po przetworzeniu — serwis rzuca `InternalMaskingError` i zwraca HTTP 500. Nie zwraca częściowo zamaskowanego tekstu.

### `GET /health`

Zwraca `{"status": "ok"}` gdy wszystkie singletony są załadowane. Używany przez `depends_on` w docker-compose.

---

## 3. Architektura modułów

```
pii-pipeline/
│
├── src/
│   ├── main.py               # FastAPI app, lifespan, /detect, /health
│   ├── pipeline.py           # orkiestracja: etap 1 → 2 → 3 → apply_masks()
│   ├── chunking.py           # chunk_with_offsets(), merge_spans(), resolve_conflicts()
│   ├── masking.py            # apply_masks(), assert_length()
│   │
│   ├── stages/
│   │   ├── regex_stage.py    # etap 1: regex patterns
│   │   ├── ner_stage.py      # etap 2: clarin-pl/FastPDN
│   │   └── teryt_stage.py    # etap 3: Morfeusz2 + lookup TERYT
│   │
│   ├── singletons.py         # inicjalizacja i dostęp do wszystkich singletonów
│   ├── models.py             # Pydantic: DetectRequest, DetectResponse, Span
│   └── errors.py             # InternalMaskingError i inne
│
├── data/
│   └── teryt/
│       ├── SIMC.csv          # miejscowości GUS
│       ├── ULIC.csv          # ulice GUS
│       └── TERC.csv          # gminy, powiaty, województwa GUS
│
├── tests/
│   ├── test_regex.py
│   ├── test_chunking.py
│   ├── test_ner.py
│   ├── test_teryt.py
│   ├── test_masking.py       # test niezmiennika długości
│   └── fixtures/
│
├── Dockerfile
├── requirements.txt
├── openapi.yaml
└── CLAUDE.md
```

---

## 4. Modele Pydantic

```python
# models.py

from pydantic import BaseModel, field_validator
from typing import Literal

class DetectRequest(BaseModel):
    text: str
    language: Literal["pl", "en"] = "pl"
    chunk_size: int = 1500  # nadpisywalny per-request; None = auto (len(text) < 1500 → 1 chunk)

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

class Span(BaseModel):
    """Absolutny offset w full_text. Używany wewnętrznie między etapami."""
    start: int   # inclusive
    end: int     # exclusive — jak Python slice: text[start:end]
    label: str   # PESEL | NIP | EMAIL | TELEFON | IMIE | NAZWISKO | ORG | ADRES | DATA_UR | KOD_POCZTOWY | NR_REJESTRACYJNY | NR_KARTY
    source: str  # "regex" | "ner" | "teryt" — do debugowania i resolve_conflicts
```

---

## 5. Singletony

Wszystkie ciężkie zasoby inicjowane **raz przy starcie aplikacji** przez FastAPI `lifespan`. Żaden etap nie ładuje zasobów per-request.

```python
# singletons.py

import asyncio
from transformers import pipeline as hf_pipeline
import morfeusz2
import pandas as pd
from functools import lru_cache

_ner_model = None
_ner_lock: asyncio.Lock = None   # asyncio.Lock — współdzielony z event loop
_morfeusz = None
_teryt_simc: set[str] = set()
_teryt_ulic: set[str] = set()
_teryt_terc: set[str] = set()   # gminy, powiaty, województwa

def get_ner_model():
    return _ner_model

def get_ner_lock() -> asyncio.Lock:
    return _ner_lock

def get_morfeusz():
    return _morfeusz

def get_teryt() -> tuple[set[str], set[str], set[str]]:
    return _teryt_simc, _teryt_ulic, _teryt_terc

async def init_all():
    global _ner_model, _ner_lock, _morfeusz, _teryt_simc, _teryt_ulic, _teryt_terc

    # NER — ładowanie synchroniczne, raz
    _ner_model = hf_pipeline(
        "ner",
        model="clarin-pl/FastPDN",
        aggregation_strategy="first",  # scala B-/I- tokeny w jeden span
        device=0,        # GPU jeśli dostępne, fallback na CPU
    )

    # Lock dla NER — jeden request na raz przez inferencję
    # (asyncio.Lock, nie threading.Lock — działa z run_in_executor)
    _ner_lock = asyncio.Lock()

    # Morfeusz2
    _morfeusz = morfeusz2.Morfeusz()

    # TERYT — ładowanie CSV do setów (lowercase, strip)
    simc = pd.read_csv("data/teryt/SIMC.csv", sep=";", encoding="utf-8")
    _teryt_simc = set(simc["NAZWA"].str.lower().str.strip())

    ulic = pd.read_csv("data/teryt/ULIC.csv", sep=";", encoding="utf-8")
    _teryt_ulic = set(ulic["NAZWA_1"].str.lower().str.strip())

    # TERC: nazwy jednostek administracyjnych (gminy, powiaty, wojewodztwa)
    # Uzywane jako dodatkowy sygnal w teryt_stage gdy token nie pasuje do SIMC/ULIC
    terc = pd.read_csv("data/teryt/TERC.csv", sep=";", encoding="utf-8")
    _teryt_terc = set(terc["NAZWA"].str.lower().str.strip())
```

```python
# main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from .singletons import init_all

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_all()
    yield
    # cleanup jeśli potrzebny

app = FastAPI(lifespan=lifespan)
```

---

## 6. Chunking

### Parametry domyślne

| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| `max_chars` | 1500 | ~400 tokenów dla polskiego tekstu z marginesem dla FastPDN (limit 512) |
| `overlap_chars` | 200 | Zapewnia kontekst dla encji na granicy chunku |

### Zasada cięcia

Cięcie zawsze na separatorze — nigdy w środku słowa ani liczby. Separatory w kolejności priorytetu:

```python
SEPARATORS = [". ", "? ", "! ", ";\n", "\n\n", "\n", ", ", " i ", " oraz ", "; "]
```

`find_split_point(text, max_chars)` szuka ostatniego wystąpienia separatora w oknie `text[:max_chars]`. Cięcie następuje **po** separatorze (separator trafia do poprzedniego chunku).

### Niezmiennik chunkingu

```python
for chunk in chunks:
    assert text[chunk.offset_start : chunk.offset_start + len(chunk.text)] == chunk.text
```

Ten assert wykonywany jest w testach jednostkowych i opcjonalnie w trybie `DEBUG` przy każdym żądaniu.

### Struktura

```python
@dataclass
class Chunk:
    text: str
    offset_start: int   # absolutny offset w oryginalnym full_text
```

### Re-assembly offsetów

Po przetworzeniu chunku każdy zwrócony `Span` jest tłumaczony na przestrzeń globalną:

```python
global_span = Span(
    start=local_span.start + chunk.offset_start,
    end=local_span.end + chunk.offset_start,
    label=local_span.label,
    source=local_span.source,
)
```

---

## 7. Etap 1 — Regex

### Wzorce (skompilowane na poziomie modułu)

```python
# stages/regex_stage.py

import re
from ..models import Span

# Kompilacja raz przy imporcie modułu
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("PESEL",            re.compile(r'\b\d{2}[0-3]\d[0-5]\d{4}\d\b')),
    ("NIP",              re.compile(r'\b\d{3}-\d{3}-\d{2}-\d{2}\b|\b\d{10}\b')),
    ("EMAIL",            re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.I)),
    ("TELEFON",          re.compile(r'(?<!\d)(\+48[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)')),
    ("KOD_POCZTOWY",     re.compile(r'\b\d{2}-\d{3}\b')),
    ("NR_KARTY",         re.compile(r'\b(?:\d{4}[\s-]?){3}\d{4}\b')),
    ("NR_REJESTRACYJNY", re.compile(r'\b[A-Z]{2,3}[\s]?\d{4,5}\b|\b[A-Z]{2,3}[\s]?[A-Z0-9]{4,5}\b')),
    ("DATA_URODZENIA", re.compile(
        r'(?:'
        # -- prefiksy słowne --
        r'urodzon[aey]m?\s+dnia\b'   # "urodzonym", "urodzoną", "urodzonej"
        r'|urodzon[ay]\b'
        r'|data\s+ur\.?\s*:?'
        r'|data\s+urodzenia\s*:?'
        r'|ur\.?\s*:?'
        r'|dob\s*:?'
        r'|date\s+of\s+birth\s*:?'
        r'|rocznik\b'
        r'|ur\.\s+w\b'
        r')\s*'

        # -- data --
        r'('
        # DD.MM.YYYY  DD-MM-YYYY  DD/MM/YYYY
        r'\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}'
        # YYYY-MM-DD  YYYY.MM.DD
        r'|\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}'
        # DD miesiąc_słownie YYYY  np. "3 maja 1990"
        r'|\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|'
        r'lipca|sierpnia|września|października|listopada|grudnia)\s+\d{4}'
        r'(?:\s*r\.?)?'
        # DD cyfra_rzymska YYYY  np. "3 V 1990"
        r'|\d{1,2}\.?\s*(?:I{1,3}|IV|VI{0,3}|IX|XI{0,2}|XII)\.?\s+\d{4}'
        # sam rok
        r'|\d{4}(?:\s*r\.?)?'
        r')'

        # opcjonalne " r." po dacie (np. "15.06.1985 r.")
        r'(?:\s*r\.)?',

        re.IGNORECASE,
    )),
]

def run(text: str) -> list[Span]:
    """
    Zwraca listę Span z absolutnymi offsetami w przekazanym tekście.
    Dla DATA_URODZENIA — span obejmuje tylko datę, nie słowo kluczowe.
    """
    spans = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            # Dla DATA_URODZENIA używamy grupy 1 (sama data)
            if label == "DATA_URODZENIA" and m.lastindex:
                start, end = m.start(1), m.end(1)
            else:
                start, end = m.start(), m.end()
            spans.append(Span(start=start, end=end, label=label, source="regex"))
    return spans
```

**Rozszerzalność:** dodanie nowego wzorca = jeden wpis w `_PATTERNS`. Żadne inne miejsce w kodzie nie wymaga zmiany.

---

## 8. Etap 2 — NER (clarin-pl/FastPDN)

### Mapowanie labelów

FastPDN używa tagów BIO. Z `aggregation_strategy="first"` model zwraca już scalone spany. Mapowanie do wewnętrznych labelów:

```python
NER_LABEL_MAP = {
    "persName":  ["IMIE", "NAZWISKO"],  # osoby — maskuj oba
    "orgName":   "ORG",
    "placeName": None,   # ignoruj — adresy obsługuje etap 3 (TERYT)
    "date":      None,   # ignoruj — daty urodzenia obsługuje regex z kontekstem
}
```

`placeName` jest ignorowany przez NER — TERYT z kontekstem jest bardziej precyzyjny dla polskich nazw miejscowości.

### Asynchroniczna inferencja (thread safety)

PyTorch nie jest thread-safe przy równoległej inferencji na tym samym modelu. Rozwiązanie: **jeden `asyncio.Lock`** + `run_in_executor` żeby nie blokować event loop FastAPI.

```python
# stages/ner_stage.py

import asyncio
from ..singletons import get_ner_model, get_ner_lock

async def run(text: str) -> list[Span]:
    model = get_ner_model()
    lock = get_ner_lock()

    def _infer():
        return model(text)

    async with lock:
        # run_in_executor przenosi inferencję do thread pool
        # event loop FastAPI nie blokuje się podczas czekania
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _infer)

    spans = []
    for r in results:
        label = NER_LABEL_MAP.get(r["entity_group"])
        if label is None:
            continue
        spans.append(Span(
            start=r["start"],
            end=r["end"],
            label=label if isinstance(label, str) else "OSOBA",
            source="ner",
        ))
    return spans
```

**Uzasadnienie wyboru lock zamiast puli modeli:** Przy 1–4 workerach uvicorn narzut kolejkowania jest akceptowalny. RAM dla jednej kopii FastPDN to ~1.5–2 GB — pula 2–4x byłaby zbyt droga. Lock można zastąpić pulą w przyszłości bez zmiany interfejsu `run()`.

### Chunking dla NER

NER działa na chunkach (limit 512 tokenów FastPDN). Chunki przetwarzane sekwencyjnie (jeden lock):

```python
async def run_chunked(text: str, chunks: list[Chunk]) -> list[Span]:
    all_spans = []
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
```

---

## 9. Etap 3 — Morfeusz2 + TERYT

### Cel

Wykrywanie nazw miejscowości i ulic pisanych wielką literą w różnych formach gramatycznych (np. „w Drzewie" → forma mianownikowa „Drzewo" → match w SIMC).

### Algorytm

```python
# stages/teryt_stage.py

import re
from ..singletons import get_morfeusz, get_teryt
from ..models import Span

_UPPER_TOKEN = re.compile(r'\b[A-ZŁŻŹĆŚĄÓĘŃ][a-złżźćśąóęń]+\b')

def run(text: str) -> list[Span]:
    morfeusz = get_morfeusz()
    teryt_simc, teryt_ulic = get_teryt()
    spans = []

    for m in _UPPER_TOKEN.finditer(text):
        token = m.group()
        start, end = m.start(), m.end()

        # Morfeusz2 zwraca listę analiz: (start_seg, end_seg, (orth, base, tag, ...))
        analyses = morfeusz.analyse(token)
        lemmas = {a[2][1].lower() for a in analyses}  # forma mianownikowa (base form)

        # Match w TERYT — maskuj zawsze gdy znaleziono
        if lemmas & teryt_simc or lemmas & teryt_ulic:
            spans.append(Span(start=start, end=end, label="ADRES", source="teryt"))

    return spans
```

**Uwaga:** Morfeusz2 jest synchroniczny i szybki (C++ binding) — nie wymaga `run_in_executor`. Działa bezpośrednio w async endpoincie.

---

## 10. Łączenie wyników — `resolve_conflicts`

Wszystkie trzy etapy działają na oryginalnym tekście. Wyniki łączone są przed nałożeniem masek.

### Polityka konfliktów

Gdy dwa spany nakładają się:

1. **Dłuższy span wygrywa** — więcej kontekstu = wyższa pewność
2. **Przy równej długości — wyższy priorytet źródła:**
   ```python
   SOURCE_PRIORITY = {"regex": 10, "ner": 8, "teryt": 6}
   ```
3. **Identyczny `(start, end, label)` z różnych źródeł** → zachowaj jeden (deduplikacja)

```python
# chunking.py

def resolve_conflicts(spans: list[Span]) -> list[Span]:
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    result: list[Span] = []
    for span in spans:
        if not result:
            result.append(span)
            continue
        last = result[-1]
        if span.start < last.end:  # nakładanie
            span_len = span.end - span.start
            last_len = last.end - last.start
            if span_len > last_len:
                result[-1] = span
            elif span_len == last_len:
                if SOURCE_PRIORITY.get(span.source, 0) > SOURCE_PRIORITY.get(last.source, 0):
                    result[-1] = span
            # else: krótszy lub niższy priorytet — ignoruj
        else:
            result.append(span)
    return result
```

---

## 11. Nakładanie masek — `apply_masks`

```python
# masking.py

def apply_masks(text: str, spans: list[Span]) -> str:
    chars = list(text)
    for span in spans:
        for i in range(span.start, span.end):
            if chars[i] != " ":   # spacje wewnątrz encji pozostają spacjami
                chars[i] = "-"
    return "".join(chars)

def assert_length(original: str, masked: str) -> None:
    if len(original) != len(masked):
        raise InternalMaskingError(
            f"Length mismatch: original={len(original)}, masked={len(masked)}"
        )
```

Przykład:
```
input:    "Jan Kowalski, PESEL 85030512345"
spans:    [(0,3,"IMIE"), (4,12,"NAZWISKO"), (21,32,"PESEL")]
output:   "--- --------, PESEL -----------"
len in:   31
len out:  31  ✓
```

---

## 12. Orkiestracja — `pipeline.py`

```python
# pipeline.py

async def detect(request: DetectRequest) -> DetectResponse:
    text = request.text
    chunks = chunk_with_offsets(text)  # zawsze, nawet dla krótkich tekstów

    # Etap 1 — regex (synchroniczny, szybki, na pełnym tekście)
    regex_spans: list[Span] = []
    for chunk in chunks:
        local = regex_stage.run(chunk.text)
        regex_spans += [globalize(s, chunk) for s in local]

    # Etap 2 — NER (async, przez lock, na chunkach)
    ner_spans = await ner_stage.run_chunked(text, chunks)

    # Etap 3 — TERYT (synchroniczny, na pełnym tekście bez chunkingu)
    teryt_spans = teryt_stage.run(text)

    # Łączenie i rozstrzyganie konfliktów
    all_spans = resolve_conflicts(regex_spans + ner_spans + teryt_spans)

    # Maskowanie
    masked = apply_masks(text, all_spans)
    assert_length(text, masked)

    return DetectResponse(masked_text=masked)
```

**Uwaga:** Regex i TERYT działają na pełnym tekście bez chunkingu (są szybkie i nie mają limitu tokenów). Chunking dotyczy wyłącznie NER.

---

## 13. Endpoint

```python
# main.py

@app.post("/detect", response_model=DetectResponse)
async def detect_endpoint(request: DetectRequest):
    try:
        return await pipeline.detect(request)
    except InternalMaskingError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    # Sprawdź czy singletony są załadowane
    if get_ner_model() is None or get_morfeusz() is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok"}
```

---

## 14. Requirements

```
# requirements.txt

fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0

# NER
transformers>=4.40.0
torch>=2.3.0

# Morfeusz2 — wymaga libmorfeusz2 zainstalowanego w systemie (Dockerfile)
morfeusz2>=1.9.0

# TERYT
pandas>=2.2.0

# testy
httpx>=0.27.0
pytest>=8.2.0
pytest-asyncio>=0.23.0
```

---

## 15. Dockerfile (fragmenty krytyczne)

```dockerfile
FROM python:3.11-slim

# Morfeusz2 — natywna biblioteka C++
RUN apt-get update && apt-get install -y \
    libmorfeusz2-dev \
    morfeusz2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dane TERYT
COPY data/ ./data/

COPY src/ ./src/
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8083"]
```

---

## 16. Testy — wymagania minimalne

| Test | Co sprawdza |
|---|---|
| `test_masking.py::test_length_invariant` | `len(masked) == len(original)` dla 50 losowych inputów |
| `test_masking.py::test_spaces_preserved` | Spacje wewnątrz encji pozostają spacjami |
| `test_regex.py::test_pesel` | Wykrycie PESEL, NIP, EMAIL, TELEFON, KOD, NR_KARTY, NR_REJ, DATA_UR |
| `test_chunking.py::test_offset_invariant` | `text[chunk.offset_start:][:len(chunk.text)] == chunk.text` |
| `test_chunking.py::test_no_cut_mid_entity` | Encja nie jest rozcięta między chunkami |
| `test_teryt.py::test_drzewo` | „w Drzewie" → match SIMC → zamaskowane |
| `test_teryt.py::test_false_positive` | Pospolite słowa pisane wielką literą NIE trafiają do TERYT |
| `test_ner.py::test_integration` | FastPDN wykrywa imię i nazwisko w prostym zdaniu |

---

## 17. Otwarte decyzje (poza scope tego PRD)

| Kwestia | Opcje |
|---|---|
| Obsługa języka `"en"` | Osobny model angielski, stub zwracający błąd, regex-only |
| Warmup modelu | Pusty forward pass po załadowaniu żeby JIT skompilował grafy |
| Metryki precision/recall | Osobny notebook z polskim test setem |
| Cache wyników | Redis z kluczem `hash(text)` — dla identycznych dokumentów |
| Upgrade do puli modeli | Gdy lock stanie się wąskim gardłem przy >4 workerach |
