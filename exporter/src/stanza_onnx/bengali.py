"""Metadata for the native downstream Bengali rule tokenizer.

No upstream Bengali tokenizer model exists in the pinned Stanza catalog. This
module packages the native rule implementation; it does not impersonate an
upstream Stanza reference or require character-language-model checkpoints.
"""

import json
from pathlib import Path
from . import UNICODE_VERSION

LANGUAGE = "bn"
PACKAGE = "bengali-rule-v1"
RULES_VERSION = 1


def metadata():
    return {
        "kind": "bengali-rule",
        "language": LANGUAGE,
        "package": PACKAGE,
        "rules_version": RULES_VERSION,
        "unicode_version": UNICODE_VERSION,
    }


def validate_metadata(value):
    expected = metadata()
    if not isinstance(value, dict):
        raise ValueError("Bengali rule tokenizer metadata must be an object")
    if not set(expected).issubset(value) or not set(value).issubset(
        set(expected) | {"source", "license"}
    ):
        raise ValueError(
            "Bengali rule tokenizer metadata has missing or unknown fields"
        )
    for name, required in expected.items():
        if type(value[name]) is not type(required) or value[name] != required:
            raise ValueError(f"Invalid Bengali rule tokenizer metadata: {name}")
    for name in ("source", "license"):
        if name in value and (
            not isinstance(value[name], str) or not value[name].strip()
        ):
            raise ValueError(f"Invalid Bengali rule tokenizer metadata: {name}")
    return value


def write_artifacts(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer.json").write_text(
        json.dumps(metadata(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
