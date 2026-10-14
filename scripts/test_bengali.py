"""Check the model-free Bengali bundle against authored rules and source spans.

This is a rule contract test, not a comparison with pretrained Stanza models.
The exporter runs with Python -S and inference runs without ONNX Runtime.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OFFSET_UNITS = ("unicode-code-point", "utf16", "utf8")
EXPECTED_CAPABILITIES = {
    "tokenize": True,
    "sentenceSplit": True,
    "multiwordExpansion": False,
    "pos": False,
    "morphology": False,
    "lemma": "none",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_fixtures(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = document.get("texts")
    require(isinstance(entries, list) and entries, "Expected authored Bengali cases")
    ids = set()
    for entry in entries:
        require(isinstance(entry, dict), "Fixture case must be an object")
        require(isinstance(entry.get("id"), str), "Fixture ID must be a string")
        require(entry["id"] not in ids, "Fixture IDs must be unique")
        ids.add(entry["id"])
        require(isinstance(entry.get("text"), str), "Fixture text must be a string")
        expected = entry.get("expectedTokens")
        require(isinstance(expected, list), "Fixture must have expectedTokens")
        for sentence in expected:
            require(
                isinstance(sentence, list) and sentence, "Expected nonempty sentence"
            )
            require(
                all(isinstance(token, str) and token for token in sentence),
                "Expected nonempty token strings",
            )
    return entries


def offset(prefix, unit):
    if unit == "utf16":
        return len(prefix.encode("utf-16-le")) // 2
    if unit == "utf8":
        return len(prefix.encode("utf-8"))
    return len(prefix)


def source_slice(text, start, end, unit):
    if unit == "utf16":
        return text.encode("utf-16-le")[start * 2 : end * 2].decode("utf-16-le")
    if unit == "utf8":
        return text.encode("utf-8")[start:end].decode("utf-8")
    return text[start:end]


def check_sentences(entry, sentences, unit):
    label = f"{entry['id']} ({unit})"
    require(isinstance(sentences, list), f"{label}: expected sentence array")
    actual = [[token["text"] for token in item["tokens"]] for item in sentences]
    require(
        actual == entry["expectedTokens"],
        f"{label}: token/sentence difference\nexpected={entry['expectedTokens']!r}\nactual={actual!r}",
    )
    text = entry["text"]
    cursor = 0
    for sentence in sentences:
        for token in sentence["tokens"]:
            token_text = token["text"]
            start = text.find(token_text, cursor)
            require(start >= cursor, f"{label}: token must occur in original input")
            end = start + len(token_text)
            expected_start = offset(text[:start], unit)
            expected_end = offset(text[:end], unit)
            require(
                type(token["start"]) is int and type(token["end"]) is int,
                f"{label}: offsets must be integers",
            )
            require(
                (token["start"], token["end"]) == (expected_start, expected_end),
                f"{label}: offsets differ for {token_text!r}",
            )
            require(
                source_slice(text, token["start"], token["end"], unit) == token_text,
                f"{label}: token must slice original input exactly",
            )
            require(
                token["words"]
                == [
                    {
                        "text": token_text,
                        "upos": None,
                        "xpos": None,
                        "feats": None,
                        "lemma": None,
                    }
                ],
                f"{label}: rule tokenizer must not invent word annotations",
            )
            cursor = end


def run(command, environment, timeout, input_text=None):
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=timeout,
    )


def successful_json(command, environment, timeout, input_text=None):
    completed = run(command, environment, timeout, input_text)
    require(
        completed.returncode == 0,
        f"Command failed: {command!r}\n{completed.stderr}",
    )
    return json.loads(completed.stdout)


def check_bundle(args, bundle, workspace):
    fixtures = read_fixtures(args.fixtures)
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "exporter" / "src")
    environment["ORT_DYLIB_PATH"] = str(workspace / "missing-onnxruntime-library")
    exporter = [args.python, "-S", "-m", "stanza_onnx"]
    successful_json(
        exporter + ["export", "--lang", "bn", "--output", str(bundle)],
        environment,
        args.timeout,
    )
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    require(manifest["language"] == "bn", "Expected Bengali bundle")
    require(manifest["tokenizerKind"] == "bengali-rule", "Expected rule tokenizer")
    require(manifest["unicodeVersion"] == "18.0.0", "Expected Unicode18 bundle")
    require(
        manifest["capabilities"] == EXPECTED_CAPABILITIES,
        "Bengali capabilities must advertise tokenization and sentence splitting only",
    )
    require(set(manifest["processors"]) == {"tokenize"}, "No neural processors")
    require(
        {item["path"] for item in manifest["files"]} == {"tokenizer.json"},
        "Rule bundle must have only tokenizer metadata, without ONNX/embedding assets",
    )
    require(
        {path.name for path in bundle.iterdir()} == {"bundle.json", "tokenizer.json"},
        "Rule bundle must contain only the two metadata files",
    )
    native = [str(args.binary), "--model-dir", str(bundle)]
    info = successful_json(native + ["info"], environment, args.timeout)
    require(info["language"] == "bn", "Expected Bengali info")
    require(info["tokenizerKind"] == "bengali-rule", "Expected native rule tokenizer")
    require(info["unicodeVersion"] == "18.0.0", "Expected Unicode18 info")
    require(
        info["capabilities"] == EXPECTED_CAPABILITIES, "Unexpected native capabilities"
    )

    for unit in OFFSET_UNITS:
        for entry in fixtures:
            result = successful_json(
                native + ["analyze", "--text", entry["text"], "--offset-unit", unit],
                environment,
                args.timeout,
            )
            require(result["language"] == "bn", "Unexpected analysis language")
            require(result["offsetUnit"] == unit, "Unexpected analysis offset unit")
            check_sentences(entry, result["sentences"], unit)

    input_path = workspace / "input.json"
    input_path.write_text(
        json.dumps({"texts": fixtures}, ensure_ascii=False), encoding="utf-8"
    )
    batch = successful_json(
        native + ["batch", "--input", str(input_path)], environment, args.timeout
    )
    require(batch["language"] == "bn", "Unexpected batch language")
    require(
        batch["processors"] == manifest["processors"], "Unexpected batch processors"
    )
    require(len(batch["parity"]) == len(fixtures), "Batch omitted fixture cases")
    for entry, result in zip(fixtures, batch["parity"]):
        require(result["id"] == entry["id"], "Batch changed fixture order/IDs")
        require(result["text"] == entry["text"], "Batch changed source text")
        check_sentences(entry, result["sentences"], "unicode-code-point")

    requests = [{"id": 0, "method": "info"}]
    checks = {}
    next_id = 1
    for unit in OFFSET_UNITS:
        for entry in fixtures:
            requests.append(
                {
                    "id": next_id,
                    "method": "analyze",
                    "params": {"text": entry["text"], "offsetUnit": unit},
                }
            )
            checks[next_id] = (entry, unit)
            next_id += 1
    invalid_id = next_id
    requests.append({"id": invalid_id, "method": "analyze", "params": {"text": False}})
    repeat_id = next_id + 1
    requests.append(
        {
            "id": repeat_id,
            "method": "analyze",
            "params": {"text": fixtures[0]["text"]},
        }
    )
    checks[repeat_id] = (fixtures[0], "unicode-code-point")
    shutdown_id = next_id + 2
    requests.append({"id": shutdown_id, "method": "shutdown"})
    lines = "\n".join(json.dumps(item, ensure_ascii=False) for item in requests) + "\n"
    # No response is allowed after an acknowledged shutdown.
    lines += json.dumps({"id": shutdown_id + 1, "method": "info"}) + "\n"
    completed = run(native + ["serve"], environment, args.timeout, lines)
    require(completed.returncode == 0, f"Persistent process failed: {completed.stderr}")
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    require(len(responses) == len(requests), "Unexpected persistent response count")
    require(
        [item["id"] for item in responses] == [item["id"] for item in requests],
        "Persistent responses changed request order/IDs",
    )
    for response in responses:
        request_id = response["id"]
        if request_id in checks:
            require("result" in response, f"Persistent analysis failed: {response!r}")
            entry, unit = checks[request_id]
            result = response["result"]
            require(result["offsetUnit"] == unit, "Persistent offset unit differs")
            check_sentences(entry, result["sentences"], unit)
        elif request_id == invalid_id:
            require(
                "error" in response,
                "Invalid text must fail without closing the process",
            )
        elif request_id == shutdown_id:
            require(
                response["result"]["closed"] is True, "Shutdown was not acknowledged"
            )
        else:
            require(
                response["result"]["capabilities"] == EXPECTED_CAPABILITIES,
                "Persistent info differs",
            )

    reference_output = workspace / "rejected-reference.json"
    rejected = run(
        exporter
        + [
            "reference",
            "--lang",
            "bn",
            "--input",
            str(input_path),
            "--output",
            str(reference_output),
        ],
        environment,
        args.timeout,
    )
    require(
        rejected.returncode != 0, "Bengali must not claim a pretrained Stanza reference"
    )
    require(
        "Stanza" in rejected.stderr
        and (
            "rule" in rejected.stderr.lower() or "downstream" in rejected.stderr.lower()
        ),
        f"Reference rejection must explain downstream rules: {rejected.stderr}",
    )
    require(not reference_output.exists(), "Rejected reference must not write output")
    print(
        json.dumps(
            {
                "language": "bn",
                "fixtureCases": len(fixtures),
                "offsetUnits": list(OFFSET_UNITS),
                "exportWithoutSitePackages": True,
                "onnxRuntimeRequired": False,
                "persistentRequests": len(requests),
                "errorRecovery": True,
                "shutdown": True,
                "stanzaReferenceRejected": True,
                "bundle": str(bundle),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Absent or empty output directory; retained for adapter checks",
    )
    parser.add_argument("--fixtures", type=Path, default=ROOT / "fixtures" / "bn.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    require(args.timeout > 0, "Timeout must be positive")
    args.binary = args.binary.resolve()
    args.fixtures = args.fixtures.resolve()
    with tempfile.TemporaryDirectory(prefix="stanza-onnx-bengali-check-") as temporary:
        workspace = Path(temporary)
        bundle = args.bundle.resolve() if args.bundle else workspace / "bundle"
        check_bundle(args, bundle, workspace)


if __name__ == "__main__":
    main()
