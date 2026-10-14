"""Assemble flat runtime assets and record their integrity and provenance."""

import json
from pathlib import Path
import shutil
import tempfile

from . import __version__, RESOURCE_VERSION, STANZA_VERSION, UNICODE_VERSION
from .cache import digest
from .catalog import capabilities, language_profile, selected_sources

RUNTIME_FILES = {
    "tokenizer.onnx",
    "tokenizer.json",
    "pos-char.onnx",
    "pos-tagger.onnx",
    "pos-charlm-forward.onnx",
    "pos-charlm-backward.onnx",
    "pos-vocab.json",
    "pretrained.f32",
    "mwt.json",
    "mwt.onnx",
    "mwt_encoder.onnx",
    "mwt_step.onnx",
    "lemma.json",
    "lemma_encoder.onnx",
    "lemma_step.onnx",
}
# Configs may embed upstream developers' absolute training paths. Ship only
# fields read by the runtime, leaving vocabularies and dictionaries untouched.
TOKENIZER_CONFIG = {
    "feat_funcs",
    "feat_dim",
    "use_dictionary",
    "num_dict_feat",
    "skip_newline",
    "shorthand",
    "lang",
}
DECODER_CONFIG = {
    "copy",
    "vocab_size",
    "pos",
    "beam_size",
    "max_dec_len",
    "batch_size",
    "ensemble_dict",
    "caseless",
    "edit",
}


def sanitize_metadata(filename, data):
    if filename == "tokenizer.json":
        data["config"] = {
            key: value
            for key, value in data["config"].items()
            if key in TOKENIZER_CONFIG
        }
    elif filename in {"lemma.json", "mwt.json"} and "args" in data:
        data["args"] = {
            key: value for key, value in data["args"].items() if key in DECODER_CONFIG
        }
    return data


def _decoder_kind_matches_selection(kind, selected):
    if selected is None:
        return kind == "none"
    if selected == "identity":
        return kind == "identity"
    return kind not in {"none", "identity"}


def _required_decoder_files(processor, metadata, selected, language):
    if processor == "mwt":
        allowed_kinds = {"none", "dictionary", "classifier", "seq2seq"}
    else:
        allowed_kinds = {"none", "identity", "dictionary", "seq2seq"}

    kind = metadata["kind"]
    if kind not in allowed_kinds:
        raise ValueError(f"Unknown {processor} implementation: {kind}")
    if not _decoder_kind_matches_selection(kind, selected):
        raise ValueError(
            f"{processor} artifact does not match selected {language} profile"
        )
    if kind == "classifier":
        return {f"{processor}.onnx"}
    if kind == "seq2seq":
        return {f"{processor}_encoder.onnx", f"{processor}_step.onnx"}
    return set()


def required_files(language, artifacts):
    profile = language_profile(language)
    processors = profile["processors"]
    required = {"tokenizer.json", "tokenizer.onnx", "mwt.json", "lemma.json"}
    tokenizer = json.loads((artifacts / "tokenizer.json").read_text(encoding="utf-8"))
    if (
        tokenizer.get("language") != language
        or tokenizer.get("package") != processors["tokenize"]
    ):
        raise ValueError(
            f"Tokenizer artifact does not match selected {language} profile"
        )
    if "pos" in processors:
        required.update({"pos-vocab.json", "pos-tagger.onnx"})
        meta = json.loads((artifacts / "pos-vocab.json").read_text(encoding="utf-8"))
        if meta.get("language") != language or meta.get("model") != processors["pos"]:
            raise ValueError(f"POS artifact does not match selected {language} profile")
        if meta["has_char"]:
            required.add("pos-char.onnx")
        if meta["has_pretrain"]:
            required.add("pretrained.f32")
        for context in meta.get("contexts", []):
            filename = context["filename"]
            if filename not in {"pos-charlm-forward.onnx", "pos-charlm-backward.onnx"}:
                raise ValueError(f"Unsafe or unknown contextual graph name: {filename}")
            required.add(filename)
    for processor in ["mwt", "lemma"]:
        meta = json.loads((artifacts / f"{processor}.json").read_text(encoding="utf-8"))
        selected = processors.get(processor)
        required.update(_required_decoder_files(processor, meta, selected, language))
    return required


def _copy_runtime_asset(name, source, destination):
    if name.endswith(".json"):
        metadata = json.loads(source.read_text(encoding="utf-8"))
        metadata = sanitize_metadata(name, metadata)
        # Runtime metadata is generated output: preserve compact serialization.
        content = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        destination.write_text(content + "\n", encoding="utf-8")
    else:
        shutil.copyfile(source, destination)


def bundle(language, artifacts, output):
    artifacts = Path(artifacts).resolve()
    output = Path(output).resolve()
    profile = language_profile(language)
    if artifacts == output:
        raise ValueError(
            "Bundle output must differ from the source artifacts directory"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be absent or empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in required_files(language, artifacts):
        path = artifacts / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe runtime asset: {path}")
    # Build beside the destination, then atomically rename the complete bundle.
    with tempfile.TemporaryDirectory(
        prefix=".stanza-onnx-bundle-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        inventory = []
        for name in sorted(required_files(language, artifacts)):
            source = artifacts / name
            destination = staging / name
            _copy_runtime_asset(name, source, destination)
            inventory.append(
                {
                    "path": name,
                    "size": destination.stat().st_size,
                    "sha256": digest(destination),
                }
            )
        manifest = {
            "schemaVersion": 1,
            "language": language,
            "languageName": profile["language_name"],
            "profile": profile["profile"],
            "stanzaVersion": STANZA_VERSION,
            "resourceVersion": RESOURCE_VERSION,
            "runtimeVersion": __version__,
            "unicodeVersion": UNICODE_VERSION,
            "offsetUnit": "unicode-code-point",
            "processors": profile["processors"],
            "capabilities": capabilities(language),
            "files": inventory,
            "sources": selected_sources(language),
        }
        (staging / "bundle.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            output.rmdir()  # An empty directory is explicitly accepted above.
        staging.replace(output)
    return manifest
