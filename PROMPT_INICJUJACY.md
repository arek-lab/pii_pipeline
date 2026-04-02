# Prompt inicjujący pracę — `pii-pipeline`

Wklej ten prompt na początku każdej nowej sesji z Claude Code w repo `pii-pipeline`.

---

Pracujemy nad serwisem `pii-pipeline`. Przeczytaj najpierw `CLAUDE.md` w całości,
a następnie `PRD/pii-pipeline-prd.md`.

Kontekst architektoniczny: ten serwis jest jednym z czterech komponentów większego
systemu (`doc-masking-service`). Jego jedynym klientem jest `gateway` który wywołuje
`POST /detect` z plain textem wyekstrahowanym z dokumentu DOCX lub PDF przez Java.
Serwis zwraca zamaskowany string o identycznej długości — Java używa tego do
podmiany runs w dokumencie. Długość musi być identyczna bo Java mapuje offsety.

Zanim zaczniesz cokolwiek implementować:
1. Potwierdź że rozumiesz niezmiennik długości i dlaczego jest krytyczny
2. Potwierdź że rozumiesz dlaczego wszystkie etapy działają na oryginalnym tekście
   a nie na częściowo zamaskowanym
3. Potwierdź że rozumiesz dlaczego NER jest za asyncio.Lock

---

## Zadanie na tę sesję

Opcja E — integracja i testy end-to-end - wykonaj tylko tę opcję.

### Opcja A — bootstrap projektu (pierwsza sesja)
Zbuduj szkielet projektu zgodnie z PRD:
- Struktura katalogów i puste pliki
- `models.py` z Pydantic models
- `errors.py`
- `singletons.py` z `init_all()` (bez faktycznego ładowania modeli — użyj TODO)
- `masking.py` z `apply_masks()` i `assert_length()`
- `chunking.py` z `chunk_with_offsets()` i `resolve_conflicts()`
- `main.py` z endpointami `/detect` i `/health`
- Testy jednostkowe dla `masking.py` i `chunking.py`

Nie implementuj jeszcze etapów (`stages/`) — to osobna sesja.

### Opcja B — regex stage
Zaimplementuj `stages/regex_stage.py` zgodnie z PRD sekcja 7.
Napisz `tests/test_regex.py` pokrywający wszystkie wzorce.
Wzorce muszą być skompilowane na poziomie modułu.
Dodaj do każdego wzorca co najmniej 3 przypadki testowe (true positive)
i 2 przypadki gdzie NIE powinno matchować (true negative).

### Opcja C — NER stage
Zaimplementuj `stages/ner_stage.py` zgodnie z PRD sekcja 8.
Zakładaj że singletony są już zainicjowane.
Napisz `tests/test_ner.py` z mockiem modelu (nie ładuj prawdziwego modelu w testach).
Szczególna uwaga na poprawność tłumaczenia lokalnych offsetów FastPDN
na absolutne offsety w oryginalnym tekście.

### Opcja D — TERYT stage
Zaimplementuj `stages/teryt_stage.py` zgodnie z PRD sekcja 9.
Napisz `tests/test_teryt.py` z mockiem Morfeusz2 i małym zestawem TERYT.
Obowiązkowo przetestuj:
- „w Drzewie" → zamaskowane (SIMC match po lematyzacji)
- „Mazowieckim stylem" → NIE zamaskowane (TERC bez kontekstu adresowego)
- „woj. Mazowieckim" → zamaskowane (TERC z kontekstem adresowym)

### Opcja E — integracja i testy end-to-end
Połącz wszystkie etapy w `pipeline.py`.
Napisz `tests/test_pipeline.py` z pełnym przepływem na fiksturach tekstowych.
Każda fikstra musi zawierać:
- tekst wejściowy
- oczekiwany zamaskowany tekst
- weryfikację niezmiennika długości

---

## Srodowisko

Projekt uzywa `uv`. Jesli .venv nie istnieje:
```bash
uv venv && uv sync --extra dev
source .venv/bin/activate   # lub .venv\Scripts\activate.bat na Windows
```
Nie uzywaj `pip install` bezposrednio. Nowe zaleznosci: `uv add nazwa`.
Commituj `pyproject.toml` i `uv.lock` razem.

---

## Zasady pracy w tej sesji

- Nie zmieniaj interfejsu `DetectRequest`/`DetectResponse` bez konsultacji
- Nie zmieniaj `openapi.yaml` bez konsultacji
- Każda zmiana w `masking.py` wymaga uruchomienia `test_masking.py`
- Jeśli dodajesz nowy label encji — zaktualizuj `openapi.yaml` i `CLAUDE.md`
- Commit po każdym działającym etapie, nie na końcu całej sesji

## Czego nie rób

- Nie ładuj modeli ani CSV poza `singletons.init_all()`
- Nie wyłączaj `assert_length()` nawet tymczasowo
- Nie przekazuj zamaskowanego tekstu do kolejnego etapu — każdy etap dostaje oryginał
- Nie używaj `threading.Lock` — tylko `asyncio.Lock`
- Nie modyfikuj `chunk.offset_start` po stworzeniu chunka
