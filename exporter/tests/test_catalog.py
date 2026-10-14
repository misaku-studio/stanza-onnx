import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

from stanza_onnx.catalog import (
    capabilities,
    catalog,
    language_profile,
    resources,
    selected_sources,
    tokenizer_kind,
)


class CatalogTests(unittest.TestCase):
    def test_pinned_coverage_and_partial_resources(self):
        self.assertEqual(len(catalog()), 95)
        eligible = [
            lang for lang, profile in catalog().items() if profile["pipeline_available"]
        ]
        self.assertEqual(len(eligible), 92)
        self.assertEqual(
            {lang for lang in catalog() if lang not in eligible},
            {"ml", "si", "or"},
        )
        self.assertEqual(
            sum(capabilities(lang)["lemma"] == "identity" for lang in eligible), 5
        )
        self.assertEqual(capabilities("my")["lemma"], "none")
        self.assertFalse(capabilities("my")["pos"])

    def test_selected_packages_and_recursive_dependencies_are_pinned(self):
        for language, profile in catalog().items():
            if not profile["pipeline_available"]:
                continue
            sources = selected_sources(language)
            if tokenizer_kind(language) == "bengali-rule":
                self.assertEqual(language, "bn")
                self.assertEqual(sources, [])
                continue
            keys = {(source["processor"], source["package"]) for source in sources}
            self.assertEqual(len(keys), len(sources))
            for processor, package in profile["processors"].items():
                if package != "identity":
                    self.assertIn((processor, package), keys)
            for source in sources:
                meta = resources()[language][source["processor"]][source["package"]]
                self.assertRegex(source["md5"], r"^[a-f0-9]{32}$")
                self.assertEqual(source["md5"], meta["md5"])
                self.assertIn("/resolve/v1.11.0/models/", source["url"])
                self.assertEqual(source["license"], "NOASSERTION")
                for dependency in meta.get("dependencies", []):
                    self.assertIn((dependency["model"], dependency["package"]), keys)

    def test_catalog_entry_does_not_imply_pipeline(self):
        self.assertEqual(
            language_profile("ml", require_pipeline=False)["language_name"], "Malayalam"
        )
        with self.assertRaisesRegex(ValueError, "no selected tokenizer"):
            language_profile("ml")
        for invalid in ["../en", "/en", "English", "en/../bn", "does-not-exist"]:
            with self.assertRaisesRegex(ValueError, "Unknown language"):
                language_profile(invalid)

    def test_help_and_listing_need_only_standard_library(self):
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
        help_result = subprocess.run(
            [sys.executable, "-S", "-m", "stanza_onnx", "--help"],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        listing = subprocess.run(
            [sys.executable, "-S", "-m", "stanza_onnx", "list", "--json"],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertEqual(len(json.loads(listing.stdout)), 95)


if __name__ == "__main__":
    unittest.main()
