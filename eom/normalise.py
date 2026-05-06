"""Text normalisation for EOM source text.

All offset-comparing code (validators, compilers) must apply this
before computing or checking offsets, so byte-for-byte agreement is
guaranteed across implementations.
"""

import unicodedata


def normalise(text: str) -> str:
    """Return text in canonical EOM form.

    - Strips UTF-8 BOM if present.
    - Normalises CRLF and lone CR to LF.
    - Strips trailing whitespace from each line (preserving line breaks).
    - Applies NFC unicode normalisation.
    """
    if text.startswith("﻿"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = unicodedata.normalize("NFC", text)
    return text
