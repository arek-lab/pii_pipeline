from .errors import InternalMaskingError
from .models import Span


def apply_masks(text: str, spans: list[Span]) -> str:
    """Replace every non-space character inside each span with '-'.

    Spaces inside multi-word entities are preserved as spaces.
    The returned string is always the same length as the input.
    """
    chars = list(text)
    for span in spans:
        for i in range(span.start, span.end):
            if chars[i] != " ":
                chars[i] = "-"
    return "".join(chars)


def assert_length(original: str, masked: str) -> None:
    """Raise InternalMaskingError if lengths differ.

    Never disable this check — the Java gateway maps offsets assuming
    len(masked_text) == len(original_text) always holds.
    """
    if len(original) != len(masked):
        raise InternalMaskingError(
            f"Length mismatch: original={len(original)}, masked={len(masked)}"
        )
