"""Exercise persistent native requests, error recovery, offsets and shutdown."""

import argparse
import json
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "ort-library", "bundle"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    requests = [
        {"id": 1, "method": "info"},
        {
            "id": 2,
            "method": "analyze",
            "params": {"text": "😊 The children read books.", "offsetUnit": "utf16"},
        },
        {"id": 3, "method": "analyze", "params": {"text": False}},
        {
            "id": 4,
            "method": "analyze",
            "params": {"text": "", "offsetUnit": "unicode-code-point"},
        },
        {"id": 5, "method": "shutdown"},
    ]
    lines = (
        "\n".join(json.dumps(request, ensure_ascii=False) for request in requests)
        + "\n"
    )
    process = subprocess.run(
        [
            args.binary,
            "--model-dir",
            args.bundle,
            "--ort-library",
            args.ort_library,
            "serve",
        ],
        input=lines,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=60,
    )
    if process.returncode:
        raise RuntimeError(process.stderr)
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(responses) == 5, responses
    assert [response["id"] for response in responses] == [1, 2, 3, 4, 5]
    assert responses[0]["result"]["capabilities"]["tokenize"]
    assert responses[1]["result"]["sentences"][0]["tokens"][0]["end"] == 2
    assert "error" in responses[2]
    assert responses[3]["result"]["sentences"] == []
    assert responses[4]["result"]["closed"]
    assert (
        responses[1]["result"]["sentences"][0]["tokens"][2]["words"][0]["lemma"]
        == "child"
    )
    print(
        json.dumps(
            {
                "requests": 5,
                "persistentProcess": True,
                "errorRecovery": True,
                "utf16Offsets": True,
                "shutdown": True,
            }
        )
    )


if __name__ == "__main__":
    main()
