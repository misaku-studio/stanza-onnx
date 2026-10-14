import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from stanza_onnx.bundle import bundle, sanitize_metadata
from stanza_onnx.cache import digest


def fake_english_assets(directory):
    directory.mkdir()
    metadata = {
        "tokenizer.json": {
            "language": "en",
            "package": "combined",
            "config": {
                "lang": "en",
                "feat_dim": 5,
                "feat_funcs": ["numeric"],
                "train_file": "/private/developer/train.txt",
                "save_name": "private.pt",
            },
            "vocab": {"_id2unit": ["<PAD>", "a"]},
            "lexicon": ["secret/path"],
        },
        "pos-vocab.json": {
            "language": "en",
            "model": "combined_nocharlm",
            "has_char": False,
            "has_pretrain": False,
            "contexts": [],
        },
        "mwt.json": {
            "kind": "classifier",
            "args": {"vocab_size": 10, "batch_size": 50, "train_file": "/private/mwt"},
            "dict": {"can't": "can not"},
        },
        "lemma.json": {
            "kind": "seq2seq",
            "args": {"copy": True, "beam_size": 1, "save_dir": "/private/lemma"},
            "word_dict": {"路径": "路径"},
        },
    }
    for name, value in metadata.items():
        (directory / name).write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )
    for name in [
        "tokenizer.onnx",
        "pos-tagger.onnx",
        "mwt.onnx",
        "lemma_encoder.onnx",
        "lemma_step.onnx",
    ]:
        (directory / name).write_bytes(b"unit-test-graph")
    (directory / "tokenizer-validation.json").write_text("[]", encoding="utf-8")
    (directory / "unrelated-secret.json").write_text("{}", encoding="utf-8")


class BundleTests(unittest.TestCase):
    def test_flat_complete_inventory_and_sanitized_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_english_assets(root / "artifacts")
            manifest = bundle("en", root / "artifacts", root / "output")
            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(manifest["unicodeVersion"], "18.0.0")
            self.assertEqual(manifest["runtimeVersion"], "0.2.0")
            self.assertEqual(manifest["offsetUnit"], "unicode-code-point")
            self.assertEqual(manifest["capabilities"]["lemma"], "trained")
            self.assertNotIn(
                "tokenizer-validation.json",
                {item["path"] for item in manifest["files"]},
            )
            self.assertNotIn(
                "unrelated-secret.json", {item["path"] for item in manifest["files"]}
            )
            for item in manifest["files"]:
                file = root / "output" / item["path"]
                self.assertEqual(item["size"], file.stat().st_size)
                self.assertEqual(item["sha256"], digest(file))
            self.assertEqual(
                {path.name for path in (root / "output").iterdir()},
                {item["path"] for item in manifest["files"]} | {"bundle.json"},
            )
            tokenizer = json.loads(
                (root / "output/tokenizer.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("train_file", tokenizer["config"])
            self.assertNotIn("save_name", tokenizer["config"])
            # User lexicon/vocabulary content must not be globally path-filtered.
            self.assertEqual(tokenizer["lexicon"], ["secret/path"])
            self.assertEqual(
                json.loads((root / "output/lemma.json").read_text(encoding="utf-8"))[
                    "word_dict"
                ],
                {"路径": "路径"},
            )
            self.assertEqual(manifest["sources"][0]["license"], "NOASSERTION")

    def test_missing_processors_require_none_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "tokenizer.json").write_text(
                json.dumps(
                    {"language": "my", "package": "alt", "config": {}, "vocab": {}}
                ),
                encoding="utf-8",
            )
            (artifacts / "tokenizer.onnx").write_bytes(b"unit-test-graph")
            for processor in ("mwt", "lemma"):
                (artifacts / f"{processor}.json").write_text(
                    '{"kind":"none"}', encoding="utf-8"
                )
            manifest = bundle("my", artifacts, root / "output")
            self.assertEqual(manifest["capabilities"]["lemma"], "none")
            (artifacts / "lemma.json").write_text(
                '{"kind":"identity"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "lemma artifact does not match"):
                bundle("my", artifacts, root / "invalid-output")
            self.assertFalse((root / "invalid-output").exists())

    def test_missing_asset_failure_does_not_create_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_english_assets(root / "artifacts")
            (root / "artifacts/mwt.onnx").unlink()
            with self.assertRaisesRegex(ValueError, "Missing or unsafe"):
                bundle("en", root / "artifacts", root / "output")
            self.assertFalse((root / "output").exists())

    def test_symlink_graph_and_context_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_english_assets(root / "artifacts")
            (root / "artifacts/tokenizer.onnx").unlink()
            (root / "outside.onnx").write_bytes(b"outside")
            (root / "artifacts/tokenizer.onnx").symlink_to(root / "outside.onnx")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                bundle("en", root / "artifacts", root / "output")
            (root / "artifacts/tokenizer.onnx").unlink()
            (root / "artifacts/tokenizer.onnx").write_bytes(b"graph")
            meta = json.loads(
                (root / "artifacts/pos-vocab.json").read_text(encoding="utf-8")
            )
            meta["contexts"] = [{"filename": "../outside.onnx"}]
            (root / "artifacts/pos-vocab.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Unsafe or unknown"):
                bundle("en", root / "artifacts", root / "output")

    def test_foreign_tokenizer_metadata_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_english_assets(root / "artifacts")
            meta = json.loads(
                (root / "artifacts/tokenizer.json").read_text(encoding="utf-8")
            )
            meta["language"] = "fr"
            (root / "artifacts/tokenizer.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "Tokenizer artifact does not match"
            ):
                bundle("en", root / "artifacts", root / "output")
            self.assertFalse((root / "output").exists())

    def test_profile_mismatch_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_english_assets(root / "artifacts")
            (root / "output").mkdir()
            (root / "output/keep").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                bundle("en", root / "artifacts", root / "output")
            self.assertEqual((root / "output/keep").read_text(encoding="utf-8"), "keep")
            (root / "artifacts/mwt.json").write_text(
                '{"kind":"none"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                bundle("en", root / "artifacts", root / "other-output")


if __name__ == "__main__":
    unittest.main()
