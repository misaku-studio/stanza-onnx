"""Check the native CLI against saved Stanza references, without a JS runtime."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess


def _reference_inputs(reference):
    """Send only protocol fields; retain reference annotations for comparison."""
    return {
        "texts": [
            {"id": entry["id"], "text": entry["text"]} for entry in reference["parity"]
        ]
    }


def _batch_command(args, language, input_path, output_path):
    return [
        str(args.binary),
        "--model-dir",
        str(args.bundles_dir / language),
        "--ort-library",
        str(args.ort_library),
        "batch",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]


def _compare_language(language, reference, native):
    if (
        native["language"] != reference["language"]
        or native["stanzaVersion"] != reference["stanzaVersion"]
        or native["processors"] != reference["processors"]
        or native.get("unicodeVersion") != reference.get("unicodeVersion")
        or not reference.get("unicodeVersion")
    ):
        raise ValueError("Reference and native bundle metadata disagree")
    expected = reference["parity"]
    actual = native["parity"]
    if len(expected) != len(actual):
        raise ValueError("Fixture counts differ")
    exact = sum(left == right for left, right in zip(expected, actual))
    failures = [
        {"expected": left, "actual": right}
        for left, right in zip(expected, actual)
        if left != right
    ]
    words = sum(
        len(token["words"])
        for entry in expected
        for sentence in entry["sentences"]
        for token in sentence["tokens"]
    )
    return {
        "language": language,
        "status": "passed" if not failures else "failed",
        "fixtures": len(expected),
        "exactFixtures": exact,
        "words": words,
        "failures": failures,
    }


def _summarize(results):
    failures = {}
    for language, result in results.items():
        if result["status"] != "passed":
            # Full per-case differences are already saved in LANG/parity.json.
            failures[language] = {
                key: value for key, value in result.items() if key != "failures"
            }
    return {
        "pipelines": len(results),
        "passed": sum(item["status"] == "passed" for item in results.values()),
        "fixtures": sum(item.get("fixtures", 0) for item in results.values()),
        "exactFixtures": sum(item.get("exactFixtures", 0) for item in results.values()),
        "words": sum(item.get("words", 0) for item in results.values()),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--ort-library", required=True, type=Path)
    parser.add_argument("--bundles-dir", required=True, type=Path)
    parser.add_argument(
        "--references-dir",
        required=True,
        type=Path,
        help="LANG/reference.json below this directory",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in (
        "binary",
        "ort_library",
        "bundles_dir",
        "references_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    languages = args.languages or sorted(
        directory.name
        for directory in args.references_dir.iterdir()
        if (directory / "reference.json").is_file()
    )
    if not languages:
        parser.error("No reference files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def check(language):
        try:
            reference = json.loads(
                (args.references_dir / language / "reference.json").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(reference.get("parity"), list) or not reference["parity"]:
                raise ValueError("Reference must contain nonempty parity fixtures")
            work = args.output_dir / language
            work.mkdir(exist_ok=True)
            input_path = work / "input.json"
            input_path.write_text(
                json.dumps(_reference_inputs(reference), ensure_ascii=False),
                encoding="utf-8",
            )
            output = work / "native.json"
            process = subprocess.run(
                _batch_command(args, language, input_path, output),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
            )
            (work / "stderr.log").write_text(process.stderr, encoding="utf-8")
            if process.returncode:
                raise ValueError(
                    f"Native runner exited {process.returncode}: {process.stderr[-3000:]}"
                )
            native = json.loads(output.read_text(encoding="utf-8"))
            report = _compare_language(language, reference, native)
            (work / "parity.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return language, report
        except Exception as error:
            return language, {
                "language": language,
                "status": "failed",
                "error": str(error),
            }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = dict(pool.map(check, languages))
    summary = _summarize(results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
