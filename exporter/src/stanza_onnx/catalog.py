"""The pinned resource catalog is separate from usable pipeline support."""

from functools import lru_cache
from importlib.resources import files
import json
import re

from . import RESOURCE_VERSION


@lru_cache(maxsize=1)
def catalog():
    return json.loads(
        files("stanza_onnx").joinpath("data/selection.json").read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def resources():
    return json.loads(
        files("stanza_onnx")
        .joinpath(f"data/resources_{RESOURCE_VERSION}.json")
        .read_text(encoding="utf-8")
    )


def language_profile(language, *, require_pipeline=True):
    # Accept only canonical catalog keys, never path fragments or implicit aliases.
    if language not in catalog():
        raise ValueError(
            f"Unknown language {language!r}; use 'stanza-onnx list' for canonical identifiers"
        )
    profile = catalog()[language]
    if require_pipeline and not profile["pipeline_available"]:
        raise ValueError(
            f"{profile['language_name']} ({language}) has partial upstream resources but no selected tokenizer pipeline"
        )
    return profile


def selected_sources(language):
    """Return task checkpoints plus recursively required auxiliary checkpoints."""
    profile = language_profile(language)
    result = {}

    def add(processor, package):
        if package == "identity" or (processor, package) in result:
            return
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", processor) or not re.fullmatch(
            r"[a-zA-Z0-9_-]+", package
        ):
            raise ValueError("Unsafe upstream checkpoint name")
        meta = resources()[language][processor][package]
        url = resources()["url"].format(
            resources_version=RESOURCE_VERSION,
            lang=language,
            filename=f"{processor}/{package}.pt",
        )
        result[(processor, package)] = {
            "processor": processor,
            "package": package,
            "url": url,
            "md5": meta["md5"],
            "license": "NOASSERTION",
        }
        for dependency in meta.get("dependencies", []):
            add(dependency["model"], dependency["package"])

    for processor, package in profile["processors"].items():
        add(processor, package)
    return list(result.values())


def capabilities(language):
    processors = language_profile(language)["processors"]
    lemma = processors.get("lemma")
    if lemma == "identity":
        lemma_capability = "identity"
    elif lemma:
        lemma_capability = "trained"
    else:
        lemma_capability = "none"
    return {
        "tokenize": True,
        "sentenceSplit": True,
        "multiwordExpansion": "mwt" in processors,
        "pos": "pos" in processors,
        "morphology": "pos" in processors,
        "lemma": lemma_capability,
    }
