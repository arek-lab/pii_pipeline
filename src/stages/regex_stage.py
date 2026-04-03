import re

from ..models import Span

# Patterns compiled once at module import — never inside request handlers.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # PESEL: YY(2) + month-tens[0-3](1) + month-units(1) + day-tens[0-5](1)
    #        + day-units(1) + serial(4) + checksum(1) = 11 digits total.
    # PRD had \d{4}\d after [0-5] which is 10 digits — missing day-units \d.
    ("PESEL", re.compile(r"\b\d{2}[0-3]\d[0-5]\d\d{4}\d\b")),
    ("NIP", re.compile(r"\b\d{3}-\d{3}-\d{2}-\d{2}\b|\b\d{10}\b")),
    # EMAIL: (?:[\w-]+\.)+ handles subdomains (e.g. test@sub.domain.org).
    ("EMAIL", re.compile(r"\b[\w.+-]+@(?:[\w-]+\.)+[a-z]{2,}\b", re.I)),
    ("TELEFON", re.compile(r"(?<!\d)(\+48[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)")),
    ("KOD_POCZTOWY", re.compile(r"\b\d{2}-\d{3}\b")),
    ("NR_KARTY", re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")),
    (
        "NR_REJESTRACYJNY",
        re.compile(r"\b[A-Z]{2,3}[\s]?\d{4,5}\b|\b[A-Z]{2,3}[\s]?[A-Z0-9]{4,5}\b"),
    ),
    (
        "DATA_URODZENIA",
        re.compile(
            r"(?:urodzon(?:y|a|ym)|ur\.?|data\s+urodzenia)\s*:?\s*"
            r"(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})",
            re.I,
        ),
    ),
]


def run(text: str) -> list[Span]:
    """Return Spans with absolute offsets into *text*.

    For DATA_URODZENIA the span covers only the date (group 1), not the
    preceding keyword — so the keyword remains unmasked in the output.
    """
    spans: list[Span] = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if label == "DATA_URODZENIA" and m.lastindex:
                start, end = m.start(1), m.end(1)
            else:
                start, end = m.start(), m.end()
            spans.append(Span(start=start, end=end, label=label, source="regex"))
    return spans
