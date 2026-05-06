"""Test fixture loaders.

Reads a markdown source file, normalises it, and pairs it with the
expected EOM JSON (with checksum / chars filled in).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eom.normalise import normalise
from eom.schema import EOMDocument

FIXTURE_DIR = Path(__file__).parent


def load_pair(slug: str) -> tuple[str, EOMDocument]:
    """Return (normalised_source_text, expected_eom) for the named fixture."""
    md_path = FIXTURE_DIR / f"{slug}.md"
    eom_path = FIXTURE_DIR / f"{slug}.eom.json"
    raw = md_path.read_text(encoding="utf-8")
    source = normalise(raw)
    eom_dict = json.loads(eom_path.read_text(encoding="utf-8"))
    eom_dict["source"]["checksum"] = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    eom_dict["source"]["chars"] = len(source)
    return source, EOMDocument.model_validate(eom_dict)
