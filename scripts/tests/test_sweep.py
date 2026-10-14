"""Model-free verification of exact parity and native-only sweep routing."""

import argparse
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "sweep.py"
spec = importlib.util.spec_from_file_location("native_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def reference():
    return {
        "language": "en",
        "stanzaVersion": "1.11.0",
        "unicodeVersion": "18.0.0",
        "processors": {"tokenize": "combined", "lemma": "identity"},
        "parity": [
            {
                "id": "underscore",
                "text": "_",
                "sentences": [
                    {
                        "tokens": [
                            {
                                "text": "_",
                                "start": 0,
                                "end": 1,
                                "words": [
                                    {
                                        "text": "_",
                                        "upos": None,
                                        "xpos": None,
                                        "feats": None,
                                        "lemma": "_",
                                    }
                                ],
                            }
                        ]
                    }
                ],
            },
            {"id": "empty", "text": "", "sentences": []},
        ],
    }


class ComparisonTests(unittest.TestCase):
    def test_exact_preserves_empty_text_and_underscore_lemma(self):
        result = sweep.compare_native_output(reference(), deepcopy(reference()))
        self.assertEqual(
            (
                result["fixtures"],
                result["exactFixtures"],
                result["referenceWords"],
                result["failureCount"],
            ),
            (2, 2, 1, 0),
        )
        self.assertEqual(result["runtime"], "native")

    def test_changed_offset_and_annotation_are_reported(self):
        native = reference()
        token = native["parity"][0]["sentences"][0]["tokens"][0]
        token["start"], token["words"][0]["lemma"] = 1, None
        result = sweep.compare_native_output(reference(), native)
        self.assertEqual(result["failureCount"], 1)
        self.assertEqual(result["failures"][0]["id"], "underscore")

    def test_changed_metadata_is_rejected(self):
        for key, value in [
            ("language", "fr"),
            ("stanzaVersion", "1.12.0"),
            ("unicodeVersion", "15.0.0"),
            ("processors", {"tokenize": "other"}),
        ]:
            native = reference()
            native[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                sweep.compare_native_output(reference(), native)

    def test_unversioned_reference_is_rejected_even_when_native_matches(self):
        for value in (None, "", 18):
            expected = reference()
            expected["unicodeVersion"] = value
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "unicodeVersion"),
            ):
                sweep.compare_native_output(expected, deepcopy(expected))

    def test_duplicate_empty_and_invalid_cases_are_rejected(self):
        for mutate in [
            lambda cases: cases.append(deepcopy(cases[0])),
            lambda cases: cases.clear(),
            lambda cases: cases[0].update(id=None),
            lambda cases: cases[0].update(sentences="invalid"),
        ]:
            native = reference()
            mutate(native["parity"])
            with self.assertRaises(ValueError):
                sweep.compare_native_output(reference(), native)


class RoutingTests(unittest.TestCase):
    def test_all_parity_languages_exclude_downstream_rule_tokenizers(self):
        args = argparse.Namespace(all=True)
        catalog = {
            "en": {"pipeline_available": True},
            "bn": {"pipeline_available": True, "tokenizer_kind": "bengali-rule"},
            "ml": {"pipeline_available": False},
        }
        self.assertEqual(sweep._select_languages(args, catalog), ["en"])

    def test_native_binary_required_before_catalog_or_download(self):
        argv = [
            str(SCRIPT),
            "--cache-dir",
            "cache",
            "--artifacts-dir",
            "artifacts",
            "--bundles-dir",
            "bundles",
            "--output-dir",
            "results",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(sweep.subprocess, "run") as run,
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            with self.assertRaises(SystemExit) as raised:
                sweep.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--native-binary", error.getvalue())
        run.assert_not_called()

    def test_skip_export_routes_to_native_batch_and_writes_exact_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, artifacts, bundles, output, fixtures = [
                root / name
                for name in ("cache", "artifacts", "bundles", "output", "fixtures")
            ]
            for directory in (cache, artifacts / "en", fixtures):
                directory.mkdir(parents=True)
            binary, library = root / "native", root / "ort-library"
            binary.write_text("fixture executable path")
            library.write_text("fixture shared library path")
            (fixtures / "en.json").write_text(
                json.dumps(
                    {
                        "texts": [
                            {"id": "underscore", "text": "_", "private": True},
                            {"id": "empty", "text": ""},
                        ]
                    }
                )
            )
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                if command[-2:] == ["list", "--json"]:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        json.dumps(
                            {
                                "en": {
                                    "language_name": "English",
                                    "pipeline_available": True,
                                }
                            }
                        ),
                        "",
                    )
                if command[0] == str(binary.resolve()):
                    self.assertIn("batch", command)
                    self.assertIn(str(library.resolve()), command)
                    Path(command[command.index("--output") + 1]).write_text(
                        json.dumps(reference())
                    )
                elif "bundle" in command:
                    Path(command[command.index("--output") + 1]).mkdir(parents=True)
                elif "reference" in command:
                    Path(command[command.index("--output") + 1]).write_text(
                        json.dumps(reference())
                    )
                else:
                    self.fail(f"Unexpected subprocess: {command}")
                return subprocess.CompletedProcess(command, 0)

            argv = [
                str(SCRIPT),
                "--cache-dir",
                str(cache),
                "--artifacts-dir",
                str(artifacts),
                "--bundles-dir",
                str(bundles),
                "--output-dir",
                str(output),
                "--fixtures-dir",
                str(fixtures),
                "--native-binary",
                str(binary),
                "--ort-library",
                str(library),
                "--skip-export",
                "--languages",
                "en",
                "--python",
                "fixture-python",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sweep.subprocess, "run", side_effect=run),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(sweep.main(), 0)
            self.assertFalse(
                any(
                    "export" in command or "download" in command for command in commands
                )
            )
            self.assertEqual(
                sum(command[0] == str(binary.resolve()) for command in commands), 1
            )
            self.assertEqual(
                json.loads((output / "sweep.json").read_text())["runtime"], "native"
            )
            self.assertEqual(
                json.loads((output / "en/parity.json").read_text())["failureCount"], 0
            )
            self.assertNotIn("private", (output / "en/input.json").read_text())


if __name__ == "__main__":
    unittest.main()
