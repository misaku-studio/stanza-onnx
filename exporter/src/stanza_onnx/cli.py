"""CLI commands import training dependencies only when inference is required."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from . import __version__
from .bundle import bundle
from .cache import download, verify_cache, write_resources
from .catalog import capabilities, catalog, language_profile, tokenizer_kind
from .bengali import write_artifacts as write_bengali_artifacts


def parser():
    root = argparse.ArgumentParser(
        description="Export pinned Stanza pipelines for native ONNX inference"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "list", help="List selected pipelines and partial resource entries"
    )
    listing.add_argument("--json", action="store_true")
    for name in ["download", "export", "bundle", "reference"]:
        command = commands.add_parser(name)
        command.add_argument(
            "--lang", required=True, help="Canonical language identifier (see list)"
        )
        command.add_argument(
            "--cache-dir",
            type=Path,
            help="Stanza model directory (required for neural pipelines; unused for Bengali rules)",
        )
        if name in {"export", "bundle", "reference"}:
            command.add_argument("--output", required=True, type=Path)
        if name == "bundle":
            command.add_argument("--artifacts", required=True, type=Path)
        if name == "reference":
            command.add_argument("--input", required=True, type=Path)
    return root


def export(language, cache_dir, output):
    profile = language_profile(language)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be absent or empty: {output}")
    if tokenizer_kind(language) == "bengali-rule":
        with tempfile.TemporaryDirectory(prefix="stanza-onnx-bengali-") as temporary:
            artifacts = Path(temporary)
            write_bengali_artifacts(artifacts)
            return bundle(language, artifacts, output)
    if cache_dir is None:
        raise ValueError("--cache-dir is required for Stanza neural pipelines")
    verify_cache(language, cache_dir)
    # Separate subprocesses keep the converter's Torch thread initialization and
    # peak memory isolated. There are no absolute checkout/cache assumptions.
    with tempfile.TemporaryDirectory(prefix="stanza-onnx-export-") as temporary:
        artifacts = Path(temporary)
        converters = ["export_tokenizer"]
        if "pos" in profile["processors"]:
            converters.append("export_pos")
        converters.append("export_lemma_mwt")
        for converter in converters:
            command = [
                sys.executable,
                "-m",
                f"stanza_onnx.converters.{converter}",
                "--cache-dir",
                str(cache_dir.resolve()),
                "--lang",
                language,
                "--out",
                str(artifacts),
            ]
            subprocess.run(command, check=True, stdout=sys.stderr)
        return bundle(language, artifacts, output)


def _list_entries():
    entries = []
    for language, profile in sorted(catalog().items()):
        available = profile["pipeline_available"]
        entries.append(
            {
                "language": language,
                "languageName": profile["language_name"],
                "profile": profile["profile"],
                "pipelineAvailable": available,
                "processors": profile["processors"],
                "tokenizerKind": tokenizer_kind(language) if available else None,
                "capabilities": capabilities(language) if available else None,
            }
        )
    return entries


def _print_listing(as_json):
    entries = _list_entries()
    if as_json:
        print(json.dumps(catalog(), ensure_ascii=False, indent=2))
        return
    for entry in entries:
        if entry["pipelineAvailable"]:
            state = ", ".join(entry["processors"])
            if entry["tokenizerKind"] == "bengali-rule":
                state += " (downstream Bengali rules; no Stanza model)"
        else:
            state = "partial resources; no tokenizer pipeline"
        print(f"{entry['language']:9} {entry['languageName']:24} {state}")


def _run_command(args):
    language_profile(args.lang)
    if tokenizer_kind(args.lang) == "bengali-rule":
        if args.command == "download":
            return {"language": args.lang, "files": 0, "tokenizerKind": "bengali-rule"}
        if args.command == "reference":
            from .reference import reference

            return reference(args.lang, args.cache_dir, args.input, args.output)
    elif args.cache_dir is None:
        raise ValueError("--cache-dir is required for Stanza neural pipelines")
    if args.command == "download":
        return download(args.lang, args.cache_dir)
    if args.command == "export":
        return export(args.lang, args.cache_dir, args.output)
    if args.command == "bundle":
        return bundle(args.lang, args.artifacts, args.output)

    verify_cache(args.lang, args.cache_dir)
    write_resources(args.cache_dir)
    from .reference import reference

    return reference(args.lang, args.cache_dir, args.input, args.output)


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.command == "list":
            _print_listing(args.json)
            return
        result = _run_command(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        command_parser.exit(1, f"stanza-onnx: {error}\n")


if __name__ == "__main__":
    main()
