"""The rule pack is downstream functionality, not Stanza model parity."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from stanza_onnx.bengali import metadata, write_artifacts
from stanza_onnx.bundle import bundle
from stanza_onnx.catalog import (
    capabilities,
    catalog,
    language_profile,
    selected_sources,
    tokenizer_kind,
)
from stanza_onnx.cli import export
from stanza_onnx.reference import reference


class BengaliPackTests(unittest.TestCase):
    def test_catalog_is_explicit_about_downstream_support(self):
        profile = language_profile("bn")
        self.assertTrue(profile["pipeline_available"])
        self.assertEqual(profile["tokenizer_kind"], "bengali-rule")
        self.assertEqual(profile["processors"], {"tokenize": "bengali-rule-v1"})
        self.assertTrue(profile["selection_changes"])
        self.assertEqual(selected_sources("bn"), [])
        self.assertEqual(
            capabilities("bn"),
            {
                "tokenize": True,
                "sentenceSplit": True,
                "multiwordExpansion": False,
                "pos": False,
                "morphology": False,
                "lemma": "none",
            },
        )
        self.assertEqual(tokenizer_kind("en"), "stanza")
        self.assertTrue(selected_sources("en"))

    def test_only_the_exact_synthetic_selection_skips_upstream_sources(self):
        with patch.dict(catalog()["bn"], {"processors": {"tokenize": "unknown"}}):
            with self.assertRaisesRegex(ValueError, "Invalid downstream Bengali"):
                selected_sources("bn")
        with patch.dict(catalog()["en"], {"tokenizer_kind": "bengali-rule"}):
            with self.assertRaisesRegex(ValueError, "Invalid downstream Bengali"):
                selected_sources("en")

    def test_export_is_offline_and_requires_only_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("stanza_onnx.cli.verify_cache") as verify_cache:
                manifest = export("bn", None, root / "bundle")
            verify_cache.assert_not_called()
            self.assertEqual(manifest["tokenizerKind"], "bengali-rule")
            self.assertEqual(manifest["sources"], [])
            self.assertEqual(manifest["unicodeVersion"], "18.0.0")
            self.assertEqual(manifest["processors"], {"tokenize": "bengali-rule-v1"})
            self.assertEqual(
                [item["path"] for item in manifest["files"]], ["tokenizer.json"]
            )
            self.assertEqual(
                {item.name for item in (root / "bundle").iterdir()},
                {"bundle.json", "tokenizer.json"},
            )
            self.assertEqual(
                json.loads(
                    (root / "bundle/tokenizer.json").read_text(encoding="utf-8")
                ),
                metadata(),
            )

    def test_bundling_never_adds_model_or_placeholder_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifacts(root / "artifacts")
            (root / "artifacts/tokenizer.onnx").write_bytes(b"unused")
            (root / "artifacts/mwt.json").write_text(
                '{"kind":"none"}', encoding="utf-8"
            )
            (root / "artifacts/lemma.json").write_text(
                '{"kind":"none"}', encoding="utf-8"
            )
            manifest = bundle("bn", root / "artifacts", root / "bundle")
            self.assertEqual(
                [item["path"] for item in manifest["files"]], ["tokenizer.json"]
            )
            self.assertEqual(manifest["sources"], [])

    def test_malformed_rule_metadata_is_rejected_atomically(self):
        invalid = [
            {"kind": "stanza"},
            {"language": "en"},
            {"package": "wrong"},
            {"rules_version": 2},
            {"rules_version": True},
            {"unicode_version": "15.0.0"},
            {"config": {}},
            {"source": []},
            {"license": ""},
        ]
        for update in invalid:
            with (
                self.subTest(update=update),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                artifacts = root / "artifacts"
                artifacts.mkdir()
                (artifacts / "tokenizer.json").write_text(
                    json.dumps({**metadata(), **update}), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    ValueError, "Bengali rule tokenizer metadata"
                ):
                    bundle("bn", artifacts, root / "bundle")
                self.assertFalse((root / "bundle").exists())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifacts(root / "artifacts")
            (root / "artifacts/tokenizer.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                bundle("bn", root / "artifacts", root / "bundle")

    def test_rule_pack_has_no_stock_stanza_reference(self):
        with self.assertRaisesRegex(ValueError, "no stock Stanza Bengali reference"):
            reference("bn", None, "unused-input", "unused-output")

    def test_neural_export_still_requires_a_model_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "cache-dir is required"):
                export("en", None, Path(temporary) / "bundle")

    def test_cli_is_standard_library_only_for_all_rule_preparation(self):
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
        command = [sys.executable, "-S", "-m", "stanza_onnx"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifacts(root / "artifacts")
            for arguments in [
                ["download", "--lang", "bn"],
                ["export", "--lang", "bn", "--output", str(root / "exported")],
                [
                    "bundle",
                    "--lang",
                    "bn",
                    "--artifacts",
                    str(root / "artifacts"),
                    "--output",
                    str(root / "bundled"),
                ],
            ]:
                result = subprocess.run(
                    command + arguments,
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertEqual(data["tokenizerKind"], "bengali-rule")
            for arguments, message in [
                (
                    ["export", "--lang", "en", "--output", str(root / "english")],
                    "cache-dir is required",
                ),
                (
                    [
                        "reference",
                        "--lang",
                        "bn",
                        "--input",
                        "unused",
                        "--output",
                        "unused",
                    ],
                    "no stock Stanza Bengali reference",
                ),
            ]:
                result = subprocess.run(
                    command + arguments,
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
