# pii-pipeline — CLAUDE.md

## Co to jest

Mikroserwis detekcji i maskowania PII w języku polskim. Wejście: plain text UTF-8.
Wyjście: string o **identycznej długości** jak wejście, gdzie każda litera encji
zastąpiona jest myślnikiem `-`. Spacje wewnątrz encji pozostają spacjami.

Serwis działa wyłącznie wewnątrz sieci Docker — nie jest dostępny publicznie.
Jedynym klientem jest `doc-masking-service/gateway`.

---

## Stack

- Python 3.11
- FastAPI + uvicorn
- clarin-pl/FastPDN (HuggingFace transformers) — NER dla języka polskiego
- Morfeusz2 (C++ binding) — lematyzacja dla TERYT lookup
- pandas — ładowanie CSV TERYT przy starcie
- Docker — pełna kontrola nad obrazem (apt-get dostępny)

---

## Struktura projektu

```
src/
├── main.py           # FastAPI app, lifespan, /detect, /health
├── pipeline.py       # orkiestracja 3 etapów + apply_masks
├── chunking.py       # chunk_with_offsets, resolve_conflicts
├── masking.py        # apply_masks, assert_length
├── singletons.py     # init_all(), gettery — ładowane RAZ przy starcie
├── models.py         # DetectRequest, DetectResponse, Span (Pydantic)
├── errors.py         # InternalMaskingError
└── stages/
    ├── regex_stage.py   # etap 1: wzorce skompilowane na poziomie modułu
    ├── ner_stage.py     # etap 2: FastPDN przez asyncio.Lock + run_in_executor
    └── teryt_stage.py   # etap 3: Morfeusz2 + lookup SIMC/ULIC/TERC
data/
└── teryt/
    ├── SIMC.csv      # miejscowości GUS
    ├── ULIC.csv      # ulice GUS
    └── TERC.csv      # gminy, powiaty, województwa GUS
```

---

## Zasady — przeczytaj przed każdą zmianą

### Niezmiennik długości — najważniejsza zasada

```python
assert len(masked_text) == len(original_text)  # ZAWSZE, bez wyjątku
```

`apply_masks()` w `masking.py` zawiera `assert_length()` który rzuca
`InternalMaskingError` jeśli warunek nie jest spełniony. Serwis zwraca HTTP 500
zamiast zwracać błędny wynik. **Nigdy nie wyłączaj tego asserta.**

### Singletony — ładowane raz, nie per-request

Wszystkie ciężkie zasoby (model NER, Morfeusz2, sety TERYT) inicjowane są
w `singletons.init_all()` wywołanym przez FastAPI `lifespan`. Dostęp przez
gettery: `get_ner_model()`, `get_morfeusz()`, `get_teryt()`.

**Nigdy nie ładuj modelu ani CSV wewnątrz funkcji obsługującej request.**

### NER jest za lockiem

`ner_stage.run()` jest `async` i używa `asyncio.Lock` + `run_in_executor`.
PyTorch nie jest thread-safe — lock zapewnia że tylko jeden request na raz
przechodzi przez inferencję. Event loop FastAPI nie blokuje się podczas czekania.

**Nie usuwaj locka. Nie wywołuj modelu bezpośrednio poza `ner_stage.run()`.**

### Wszystkie etapy działają na oryginalnym tekście

Regex, NER i TERYT dostają **oryginalny tekst** — nie modyfikowany przez
poprzedni etap. Maski nakładane są dopiero w `apply_masks()` na końcu.
`resolve_conflicts()` rozstrzyga nakładające się spany zanim cokolwiek zostanie
zamaskowane.

### Offsety są absolutne w full_text

Każdy `Span.start` i `Span.end` to offset w oryginalnym tekście przekazanym
do `/detect`, nie w chunku. `chunk_with_offsets()` zwraca `Chunk.offset_start`
który jest dodawany do lokalnych offsetów z każdego etapu.

Niezmiennik który musi przechodzić zawsze:
```python
assert text[span.start:span.end] == wykryta_encja
```

### Chunking dotyczy tylko NER

Regex i TERYT działają na pełnym tekście bez chunkingu — są szybkie i nie mają
limitu tokenów. Chunking (`max_chars=1500`, `overlap_chars=200`) dotyczy
wyłącznie wywołań FastPDN.

Cięcie zawsze na separatorze — nigdy w środku słowa ani liczby. Separatory
w kolejności priorytetu: `. `, `? `, `! `, `;\n`, `\n\n`, `\n`, `, `, ` i `, ` oraz `.

---

## Kontrakt API

```
POST /detect
  Request:  { text: str, language: "pl"|"en", chunk_size?: int }
  Response: { masked_text: str }

GET /health
  Response: { status: "ok" }  — tylko gdy wszystkie singletony załadowane
```

Kontrakt jest opisany w `openapi.yaml`. **Nie zmieniaj formatu response bez
aktualizacji openapi.yaml i koordynacji z doc-masking-service.**

---

## Labele encji

```
PESEL | NIP | EMAIL | TELEFON | KOD_POCZTOWY | NR_KARTY | NR_REJESTRACYJNY
DATA_URODZENIA | IMIE | NAZWISKO | ORG | ADRES
```

Dodanie nowego labela = zmiana w `regex_stage._PATTERNS` lub `ner_stage.NER_LABEL_MAP`
+ aktualizacja `openapi.yaml`.

---

## TERYT — trzy pliki, różna polityka maskowania

| Plik | Zawartość | Polityka |
|---|---|---|
| SIMC.csv | Miejscowości | Maskuj zawsze przy match |
| ULIC.csv | Ulice | Maskuj zawsze przy match |
| TERC.csv | Gminy, powiaty, województwa | Maskuj tylko gdy poprzedza słowo adresowe (pow., gm., woj., ...) |

Morfeusz2 lematyzuje token przed lookupem — obsługuje odmiany przez przypadki
(„w Drzewie" → „drzewo" → match SIMC).

---

## Testy

```bash
pytest tests/ -v
```

Krytyczne testy których nie wolno pomijać:
- `test_masking.py::test_length_invariant` — niezmiennik długości
- `test_chunking.py::test_offset_invariant` — poprawność offsetów
- `test_teryt.py::test_drzewo` — edge case TERYT

---

## Srodowisko lokalne -- uv

Projekt uzywa `uv`. Nie uzywaj `pip` bezposrednio.

```bash
# Pierwsza konfiguracja
uv venv                  # tworzy .venv/
uv sync --extra dev      # instaluje wszystko z uv.lock

# Aktywacja
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate.bat      # Windows

# Uruchomienie
uvicorn src.main:app --reload --port 8083

# Testy
pytest tests/ -v

# Dodanie zaleznosci
uv add nazwa-pakietu          # produkcyjna -- commituj pyproject.toml i uv.lock razem
uv add --dev nazwa-pakietu    # deweloperska
```

morfeusz2 instaluje sie przez `uv sync` bez problemow na Windows 10 i Linuksie.
W Dockerze wymagany jest `libmorfeusz2-dev` z apt jako zaleznosc systemowa.

Model FastPDN pobiera sie przy pierwszym uruchomieniu (~500 MB).
Przy starcie kontenera ładowanie modelu zajmuje ~20-40 sekund --
`/health` zwroci 503 do momentu zakonczenia `init_all()`.

## Z docker-compose

```bash
docker-compose up pii-service
```

---

## Czego tu nie ma (poza scope)

- Parsowanie DOCX/PDF — to robi `doc-masking-service/docx-service`
- Autoryzacja JWT — to robi `doc-masking-service/gateway`
- Przechowywanie plików — to robi MinIO
- Mapowanie offsetów na runs POI — to robi Java

`pii-pipeline` widzi tylko string, zwraca tylko zamaskowany string.
