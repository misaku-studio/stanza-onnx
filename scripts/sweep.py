"""Build language bundles and compare the native CLI with pinned Stanza.

Run from an installed checkout. Directories are supplied explicitly so this also
works with a pre-existing local Stanza model cache and converted ONNX artifacts.
No corpus is downloaded by this script. Fixture input can be supplied privately.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ["en", "fr", "nl", "zh-hans", "th", "sd", "my"]


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _validate_parity_cases(cases: list, label: str) -> None:
    ids = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not isinstance(case.get("text"), str)
            or not isinstance(case.get("sentences"), list)
        ):
            raise ValueError(f"{label} contains an invalid parity case")
        if case["id"] in ids:
            raise ValueError(f"{label} contains duplicate case ID {case['id']}")
        ids.add(case["id"])


def compare_native_output(reference: dict, native: dict) -> dict:
    """Require exact public output parity, including selected model metadata."""
    if (
        not isinstance(reference.get("unicodeVersion"), str)
        or not reference["unicodeVersion"]
    ):
        raise ValueError("Reference must identify its preprocessing unicodeVersion")
    for name in ("language", "stanzaVersion", "processors", "unicodeVersion"):
        if native.get(name) != reference.get(name):
            raise ValueError(f"Native output and reference {name} disagree")
    expected = reference.get("parity")
    actual = native.get("parity")
    if not isinstance(expected, list) or not isinstance(actual, list):
        raise ValueError("Native output and reference must contain parity arrays")
    if not expected or len(expected) != len(actual):
        raise ValueError(
            "Native output and reference fixture counts differ or are empty"
        )
    for label, cases in (("Reference", expected), ("Native", actual)):
        _validate_parity_cases(cases, label)
    failures = [
        {"id": left["id"], "expected": left, "actual": right}
        for left, right in zip(expected, actual)
        if left != right
    ]
    return {
        "language": reference["language"],
        "runtime": "native",
        "offsetUnit": "unicode-code-point",
        "fixtures": len(expected),
        "exactFixtures": len(expected) - len(failures),
        "referenceWords": sum(
            len(token["words"])
            for case in expected
            for sentence in case["sentences"]
            for token in sentence["tokens"]
        ),
        "failureCount": len(failures),
        "failures": failures,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Stanza model cache root, with en/tokenize/... below it",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Root containing one flat converted-artifact directory per language",
    )
    parser.add_argument(
        "--bundles-dir",
        type=Path,
        required=True,
        help="Generated bundle root (one directory per language)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Logs, references and parity results; no files are written into the checkout",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=ROOT / "fixtures",
        help="Optional private fixture directory; missing languages use synthetic smoke input",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--languages",
        nargs="+",
        help="Space-separated or comma-separated language codes (default: representative matrix)",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="All selected Stanza pipelines with an upstream reference; expensive",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing selected models before export (network required)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help=(
            "Reuse existing converted artifacts; write fresh bundles and references "
            "(bundle destinations must be absent/empty)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=1,
        help="Concurrent language pipelines; each worker can use substantial memory",
    )
    parser.add_argument(
        "--timeout",
        type=positive_integer,
        default=1800,
        help="Per-subprocess timeout in seconds",
    )
    parser.add_argument(
        "--native-binary",
        type=Path,
        required=True,
        help="Native Rust CLI used for all inference",
    )
    parser.add_argument(
        "--ort-library",
        type=Path,
        help="Explicit ONNX Runtime shared library for the native CLI",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable with stanza-onnx-export installed",
    )
    return parser


def _select_languages(args, catalog):
    eligible = [
        code
        for code, entry in catalog.items()
        if entry.get("pipeline_available")
        and entry.get("tokenizer_kind", "stanza") == "stanza"
    ]
    if args.all:
        return sorted(eligible)
    codes = (
        code
        for group in (args.languages or MATRIX)
        for code in re.split(r"[\s,]+", group.strip())
        if code
    )
    return list(dict.fromkeys(codes))


def _read_fixture(source):
    fixture = json.loads(source.read_text(encoding="utf-8"))
    texts = fixture.get("texts")
    if not isinstance(texts, list) or not texts:
        raise ValueError(f"Fixture must contain nonempty texts array: {source}")

    # Metadata stays in the private source; the portable CLI consumes id/text.
    normalized = []
    for index, item in enumerate(texts):
        if isinstance(item, str):
            item = {"id": f"input_{index}", "text": item}
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("text"), str)
            or not isinstance(item.get("id"), str)
        ):
            raise ValueError(f"Invalid fixture {source}, case {index}")
        normalized.append({"id": item["id"], "text": item["text"]})
    return normalized


def _native_batch_command(args, bundle, input_path, output_path):
    command = [str(args.native_binary), "--model-dir", str(bundle)]
    if args.ort_library is not None:
        command.extend(["--ort-library", str(args.ort_library)])
    command.extend(["batch", "--input", str(input_path), "--output", str(output_path)])
    return command


def _sweep_report(languages, all_eligible, results):
    return {
        "languages": languages,
        "allEligible": bool(all_eligible),
        "runtime": "native",
        "results": results,
        "passed": sum(entry["status"] == "passed" for entry in results.values()),
        "failed": sum(
            entry["status"] in {"failed", "no_pipeline"} for entry in results.values()
        ),
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    for name in (
        "cache_dir",
        "artifacts_dir",
        "bundles_dir",
        "output_dir",
        "fixtures_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if not args.cache_dir.is_dir() and not args.download:
        parser.error("--cache-dir does not exist; provide a cache or add --download")
    if not args.fixtures_dir.is_dir():
        parser.error("--fixtures-dir must be an existing directory")
    args.native_binary = args.native_binary.resolve()
    if not args.native_binary.is_file():
        parser.error(f"Native executable not found: {args.native_binary}")
    if args.ort_library is not None:
        args.ort_library = args.ort_library.resolve()
        if not args.ort_library.is_file():
            parser.error(f"ONNX Runtime shared library not found: {args.ort_library}")

    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
    )
    cli = [args.python, "-m", "stanza_onnx"]
    catalog_process = subprocess.run(
        cli + ["list", "--json"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    if catalog_process.returncode:
        raise RuntimeError(
            f"Cannot read installed exporter catalog: {catalog_process.stderr}"
        )
    catalog = json.loads(catalog_process.stdout)
    if not isinstance(catalog, dict):
        raise RuntimeError("Expected list --json to return a language catalog mapping")
    languages = _select_languages(args, catalog)
    if not languages:
        parser.error("No language codes selected")
    unknown = [language for language in languages if language not in catalog]
    if unknown:
        parser.error(f"Unknown languages: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results: dict[str, dict] = {}
    report_path = args.output_dir / "sweep.json"

    def save(language: str, result: dict) -> None:
        with lock:
            results[language] = result
            report = _sweep_report(languages, args.all, results)
            temporary = report_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(report_path)

    def run_step(language: str, label: str, command: list[str]) -> dict:
        log_path = args.output_dir / language / f"{label}.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        if completed.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(
                f"{label} exited {completed.returncode}; see {log_path}\n{tail}"
            )
        return {"seconds": time.monotonic() - started, "log": str(log_path)}

    def process(language: str) -> None:
        entry = catalog[language]
        result = {
            "status": "running",
            "languageName": entry.get("language_name"),
            "steps": {},
        }
        save(language, result)
        started = time.monotonic()
        try:
            if not entry.get("pipeline_available"):
                result.update(
                    status="no_pipeline",
                    error="Catalog has partial resources but no selected tokenizer pipeline",
                )
                return
            if entry.get("tokenizer_kind", "stanza") != "stanza":
                raise ValueError(
                    "Downstream rule tokenizers have no stock Stanza reference; "
                    "use scripts/test_bengali.py for authored Bengali checks"
                )
            work = args.output_dir / language
            work.mkdir(parents=True, exist_ok=True)
            source = args.fixtures_dir / f"{language}.json"
            if not source.exists():
                source = ROOT / "fixtures" / "smoke.json"
                result["fixtureKind"] = "synthetic_smoke"
            else:
                result["fixtureKind"] = "language_fixture"
            normalized = _read_fixture(source)
            input_path = work / "input.json"
            input_path.write_text(
                json.dumps({"texts": normalized}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            artifacts = args.artifacts_dir / language
            bundle = args.bundles_dir / language
            common = ["--lang", language, "--cache-dir", str(args.cache_dir)]
            if args.download:
                result["steps"]["download"] = run_step(
                    language, "download", cli + ["download"] + common
                )
            if not args.skip_export:
                result["steps"]["export"] = run_step(
                    language,
                    "export",
                    cli + ["export"] + common + ["--output", str(artifacts)],
                )
            elif not artifacts.is_dir():
                raise ValueError(f"Converted artifacts do not exist: {artifacts}")
            result["steps"]["bundle"] = run_step(
                language,
                "bundle",
                cli
                + ["bundle"]
                + common
                + ["--artifacts", str(artifacts), "--output", str(bundle)],
            )
            reference = work / "reference.json"
            result["steps"]["reference"] = run_step(
                language,
                "reference",
                cli
                + ["reference"]
                + common
                + ["--input", str(input_path), "--output", str(reference)],
            )
            parity = work / "parity.json"
            native_output = work / "native.json"
            command = _native_batch_command(args, bundle, input_path, native_output)
            result["steps"]["parity"] = run_step(language, "parity", command)
            comparison = compare_native_output(
                json.loads(reference.read_text(encoding="utf-8")),
                json.loads(native_output.read_text(encoding="utf-8")),
            )
            parity.write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if comparison["failureCount"]:
                raise ValueError(
                    f"Native parity failed for {comparison['failureCount']} fixture(s); see {parity}"
                )
            result.update(
                status="passed",
                fixtures=comparison["fixtures"],
                exactFixtures=comparison["exactFixtures"],
                referenceWords=comparison["referenceWords"],
                bundle=str(bundle),
            )
        except Exception as error:
            result.update(status="failed", error=str(error))
        finally:
            result["seconds"] = time.monotonic() - started
            save(language, result)
            print(
                json.dumps(
                    {
                        "language": language,
                        "status": result["status"],
                        "seconds": round(result["seconds"], 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, language) for language in languages]
        for future in as_completed(futures):
            future.result()
    failed = sum(entry["status"] != "passed" for entry in results.values())
    print(
        json.dumps(
            {
                "report": str(report_path),
                "passed": len(results) - failed,
                "failed": failed,
            },
            ensure_ascii=False,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
