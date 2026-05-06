"""Text normalisation for EOM source text.

All offset-comparing code (validators, compilers) must apply this
before computing or checking offsets, so byte-for-byte agreement is
guaranteed across implementations.

Cross-implementation note: NFC follows the Unicode database shipped
with the running Python interpreter. For full cross-language
conformance, the EOM standard will pin a Unicode version (tracked as
an open question in v0.1; not blocking within Python).
"""

import unicodedata


def normalise(text: str) -> str:
    """Return text in canonical EOM form.

    Order of operations:
    1. Strip a leading U+FEFF (BOM) if present. Mid-text U+FEFF is preserved.
    2. Normalise CRLF and lone CR to LF.
    3. Strip trailing whitespace from each line, preserving line breaks.
       "Whitespace" here is Python's ``str.rstrip()`` default, i.e. every
       character where ``str.isspace()`` is true. That includes ASCII space,
       tab, vertical tab, form feed, plus Unicode whitespace such as
       U+00A0 (NBSP), U+2028, U+3000. Line content callers care about is
       never stripped — only trailing whitespace is removed.
    4. Apply NFC unicode normalisation.

    The function is idempotent: ``normalise(normalise(t)) == normalise(t)``.
    """
    if text.startswith("﻿"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = unicodedata.normalize("NFC", text)
    return text
