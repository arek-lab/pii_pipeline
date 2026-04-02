class InternalMaskingError(Exception):
    """Raised when the length invariant len(masked) == len(original) is violated.

    The service must return HTTP 500 rather than a wrong-length string, because
    the Java gateway uses character-level offsets to substitute runs in DOCX/PDF.
    A length mismatch would corrupt the document.
    """
