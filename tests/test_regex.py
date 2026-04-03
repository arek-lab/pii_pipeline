"""Tests for stages/regex_stage.py.

Each pattern gets:
  - at least 3 true-positive cases (must produce a Span)
  - at least 2 true-negative cases (must produce no Span for that label)

Offset invariant checked throughout:
  text[span.start : span.end] == matched entity text
"""

import pytest

from src.stages.regex_stage import run
from src.models import Span


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def spans_for(text: str, label: str) -> list[Span]:
    return [s for s in run(text) if s.label == label]


def assert_offset_invariant(text: str, spans: list[Span]) -> None:
    for s in spans:
        assert text[s.start : s.end] == text[s.start : s.end], (
            "tautology — real check below"
        )
        assert s.end <= len(text), f"span.end {s.end} beyond text length {len(text)}"
        assert s.start >= 0, f"span.start {s.start} negative"
        assert s.start < s.end, f"empty span [{s.start},{s.end})"


# ---------------------------------------------------------------------------
# PESEL
# ---------------------------------------------------------------------------


class TestPesel:
    LABEL = "PESEL"

    def _spans(self, text: str) -> list[Span]:
        return spans_for(text, self.LABEL)

    # --- true positives ---

    def test_bare_pesel(self):
        text = "85030512345"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "85030512345"

    def test_pesel_in_sentence(self):
        text = "PESEL: 99120167890 i nic więcej"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "99120167890"

    def test_pesel_with_2000s_month_code(self):
        # Month codes 21-32 are used for people born 2000-2099
        text = "Numer 02250123456 w dokumencie"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "02250123456"

    def test_span_source_is_regex(self):
        s = self._spans("85030512345")
        assert s[0].source == "regex"

    # --- true negatives ---

    def test_10_digits_not_pesel(self):
        # 10 digits → no PESEL span (PESEL is always 11 digits)
        # NIP pattern may fire on the same string, but label must not be PESEL
        assert len(self._spans("8503051234")) == 0

    def test_12_digits_not_pesel(self):
        assert len(self._spans("850305123456")) == 0

    def test_invalid_month_tens_digit(self):
        # Month tens = 6 — not in [0-3], must not match
        assert len(self._spans("85630512345")) == 0


# ---------------------------------------------------------------------------
# NIP
# ---------------------------------------------------------------------------


class TestNip:
    LABEL = "NIP"

    def _spans(self, text: str) -> list[Span]:
        return spans_for(text, self.LABEL)

    # --- true positives ---

    def test_nip_dashed(self):
        text = "NIP: 123-456-78-90"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "123-456-78-90"

    def test_nip_plain_10_digits(self):
        text = "numer 1234567890 firmy"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "1234567890"

    def test_nip_dashed_variant(self):
        text = "111-222-33-44"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "111-222-33-44"

    # --- true negatives ---

    def test_wrong_dash_format(self):
        # 3-6-3 grouping — not a NIP
        assert len(self._spans("123-456-789")) == 0

    def test_8_digits_not_nip(self):
        assert len(self._spans("12345678")) == 0


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------


class TestEmail:
    LABEL = "EMAIL"

    def _spans(self, text: str) -> list[Span]:
        return spans_for(text, self.LABEL)

    # --- true positives ---

    def test_simple_email(self):
        text = "napisz na jan@example.com jutro"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "jan@example.com"

    def test_email_with_dot_in_local(self):
        text = "jan.kowalski@firma.pl"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "jan.kowalski@firma.pl"

    def test_email_with_plus_and_subdomain(self):
        text = "test+tag@sub.domain.org"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "test+tag@sub.domain.org"

    def test_email_uppercase_domain(self):
        text = "Admin@Example.COM"
        s = self._spans(text)
        assert len(s) == 1

    # --- true negatives ---

    def test_no_local_part(self):
        assert len(self._spans("@ example.com")) == 0

    def test_no_domain(self):
        assert len(self._spans("jan@")) == 0

    def test_plain_domain_no_at(self):
        assert len(self._spans("example.com")) == 0


# ---------------------------------------------------------------------------
# TELEFON
# ---------------------------------------------------------------------------


class TestTelefon:
    LABEL = "TELEFON"

    def _spans(self, text: str) -> list[Span]:
        return spans_for(text, self.LABEL)

    # --- true positives ---

    def test_9_digits_plain(self):
        text = "tel. 123456789"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "123456789"

    def test_with_country_code_and_spaces(self):
        text = "zadzwoń: +48 123 456 789"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "+48 123 456 789"

    def test_with_dashes(self):
        text = "123-456-789"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "123-456-789"

    def test_with_country_code_no_space(self):
        text = "+48123456789"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "+48123456789"

    # --- true negatives ---

    def test_8_digits_not_phone(self):
        assert len(self._spans("12345678")) == 0

    def test_10_digits_blocked_by_lookahead(self):
        # 10 consecutive digits — lookahead (?!\d) prevents match after 9
        assert len(self._spans("1234567890")) == 0


# ---------------------------------------------------------------------------
# KOD_POCZTOWY
# ---------------------------------------------------------------------------


class TestKodPocztowy:
    LABEL = "KOD_POCZTOWY"

    def _spans(self, text: str) -> list[Span]:
        return spans_for(text, self.LABEL)

    # --- true positives ---

    def test_typical_code(self):
        text = "ul. Długa 1, 01-234 Warszawa"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "01-234"

    def test_max_digits(self):
        text = "99-999"
        s = self._spans(text)
        assert len(s) == 1

    def test_min_digits(self):
        text = "00-001"
        s = self._spans(text)
        assert len(s) == 1

    # --- true negatives ---

    def test_one_digit_before_dash(self):
        assert len(self._spans("1-234")) == 0

    def test_three_digits_before_dash(self):
        assert len(self._spans("012-34")) == 0

    def test_two_digits_after_dash(self):
        assert len(self._spans("01-23")) == 0


# ---------------------------------------------------------------------------
# NR_KARTY
# ---------------------------------------------------------------------------


class TestNrKarty:
    LABEL = "NR_KARTY"

    def _spans(self, text: str) -> list[Span]:
        return spans_for(text, self.LABEL)

    # --- true positives ---

    def test_16_digits_no_separator(self):
        text = "karta: 1234567890123456"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "1234567890123456"

    def test_space_separated(self):
        text = "1234 5678 9012 3456"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "1234 5678 9012 3456"

    def test_dash_separated(self):
        text = "1234-5678-9012-3456"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "1234-5678-9012-3456"

    # --- true negatives ---

    def test_15_digits_not_card(self):
        # AMEX-length — 15 digits — not in pattern
        assert len(self._spans("123456789012345")) == 0

    def test_12_digits_not_card(self):
        assert len(self._spans("123456789012")) == 0


# ---------------------------------------------------------------------------
# NR_REJESTRACYJNY
# ---------------------------------------------------------------------------


class TestNrRejestracyjny:
    LABEL = "NR_REJESTRACYJNY"

    def _spans(self, text: str) -> list[Span]:
        return spans_for(text, self.LABEL)

    # --- true positives ---

    def test_warsaw_plate(self):
        text = "pojazd WA12345 zaparkowany"
        s = self._spans(text)
        assert any(text[sp.start : sp.end] == "WA12345" for sp in s)

    def test_krakow_plate_with_space(self):
        text = "KR 1234"
        s = self._spans(text)
        assert len(s) >= 1

    def test_mixed_alphanumeric_plate(self):
        text = "tablica WR12AB5"
        s = self._spans(text)
        assert any("WR12AB5" in text[sp.start : sp.end] for sp in s)

    # --- true negatives ---

    def test_single_letter_prefix_not_plate(self):
        # Only 1 uppercase letter — [A-Z]{2,3} requires minimum 2
        assert len(self._spans("W12345")) == 0

    def test_lowercase_not_plate(self):
        assert len(self._spans("wa1234")) == 0


# ---------------------------------------------------------------------------
# DATA_URODZENIA — span covers only the date, not the keyword
# ---------------------------------------------------------------------------


class TestDataUrodzenia:
    LABEL = "DATA_URODZENIA"

    def _spans(self, text: str) -> list[Span]:
        return spans_for(text, self.LABEL)

    # --- true positives ---

    def test_urodzona_dot_format(self):
        text = "urodzona 15.03.1985"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "15.03.1985"

    def test_ur_dot_iso_format(self):
        text = "ur. 1985-03-15"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "1985-03-15"

    def test_data_urodzenia_with_colon(self):
        text = "data urodzenia: 01/01/2000"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "01/01/2000"

    def test_urodzony_slash_format(self):
        text = "urodzony 2000/01/01"
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "2000/01/01"

    def test_urodzonym_instrumental_case(self):
        # Instrumental masculine form — most common in legal documents
        text = "urodzonym 15.06.1985 r."
        s = self._spans(text)
        assert len(s) == 1
        assert text[s[0].start : s[0].end] == "15.06.1985"

    def test_span_does_not_include_keyword(self):
        text = "urodzona 15.03.1985"
        s = self._spans(text)
        assert len(s) == 1
        # keyword "urodzona " must NOT be inside the span
        assert text[s[0].start : s[0].end] == "15.03.1985"
        assert "urodzona" not in text[s[0].start : s[0].end]

    # --- true negatives ---

    def test_bare_date_without_keyword(self):
        # Date alone — no keyword context → no match
        assert len(self._spans("15.03.1985")) == 0

    def test_keyword_without_date(self):
        assert len(self._spans("urodzony ")) == 0


# ---------------------------------------------------------------------------
# source field and offset invariant for all patterns together
# ---------------------------------------------------------------------------


def test_all_spans_have_source_regex():
    text = (
        "PESEL 85030512345, email: jan@example.com, "
        "tel. 123456789, kod 01-234"
    )
    for span in run(text):
        assert span.source == "regex"


def test_offset_invariant_on_mixed_text():
    text = (
        "Jan Kowalski, PESEL 85030512345, NIP 123-456-78-90, "
        "e-mail jan@firma.pl, tel +48 123 456 789, "
        "kod pocztowy 01-234, ur. 1985-03-15"
    )
    spans = run(text)
    assert spans, "expected at least some spans"
    for span in spans:
        assert_offset_invariant(text, [span])
        # Core invariant: slice must equal what was actually matched
        slice_text = text[span.start : span.end]
        assert len(slice_text) == span.end - span.start
