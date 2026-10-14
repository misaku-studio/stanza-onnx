"""An offline, normalized Stanza reference for exact runtime parity checks."""

import json
from pathlib import Path
import unicodedata

from . import STANZA_VERSION, UNICODE_VERSION
from .unicode import provenance
from .unicode_stanza import adapted_stanza, analyze
from .catalog import language_profile


def normalize(document):
    """Keep only the public annotations compared with the native runtime."""
    sentences = []
    for sentence in document.sentences:
        tokens = []
        for token in sentence.tokens:
            words = []
            for word in token.words:
                words.append(
                    {
                        "text": word.text,
                        "upos": word.upos,
                        "xpos": word.xpos,
                        "feats": word.feats,
                        "lemma": word.lemma,
                    }
                )
            tokens.append(
                {
                    "text": token.text,
                    "start": token.start_char,
                    "end": token.end_char,
                    "words": words,
                }
            )
        sentences.append({"tokens": tokens})
    return sentences


def _read_inputs(input_path):
    inputs = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(inputs, dict) or not isinstance(inputs.get("texts"), list):
        raise ValueError("Reference input must contain a texts array")
    for entry in inputs["texts"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not isinstance(entry.get("text"), str)
        ):
            raise ValueError("Each reference input must have string id and text fields")
    return inputs["texts"]


def _analyze_inputs(pipeline, inputs):
    cases = []
    for entry in inputs:
        document = analyze(pipeline, entry["text"])
        cases.append(
            {
                "id": entry["id"],
                "text": entry["text"],
                "sentences": normalize(document),
            }
        )
    return cases


def reference(language, cache_dir, input_path, output):
    import stanza
    import torch

    if stanza.__version__ != STANZA_VERSION:
        raise ValueError(
            f"Reference requires Stanza {STANZA_VERSION}, found {stanza.__version__}"
        )
    profile = language_profile(language)
    inputs = _read_inputs(input_path)
    torch.set_num_threads(1)
    pipeline = stanza.Pipeline(
        language,
        dir=str(cache_dir),
        package=None,
        processors=profile["processors"],
        use_gpu=False,
        download_method=None,
        verbose=False,
    )
    with adapted_stanza():
        result = {
            "language": language,
            "stanzaVersion": stanza.__version__,
            "unicodeVersion": UNICODE_VERSION,
            "referenceMode": "stanza-unicode18-preprocessing",
            "referenceHostUnicodeVersion": unicodedata.unidata_version,
            "unicodeData": provenance(),
            "processors": profile["processors"],
            "parity": _analyze_inputs(pipeline, inputs),
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "language": language,
        "output": str(output.resolve()),
        "fixtures": len(result["parity"]),
    }
