import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stanza_onnx.cache import download_checkpoint, verify_cache, write_resources


class CacheTests(unittest.TestCase):
    def test_verified_download_replaces_corrupt_file_atomically(self):
        content = b"verified checkpoint"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "models.pt"
            target.write_bytes(b"corrupt checkpoint")
            with patch("stanza_onnx.cache.urlopen", return_value=io.BytesIO(content)):
                download_checkpoint(
                    "https://example.invalid/model.pt",
                    target,
                    hashlib.md5(content).hexdigest(),
                )
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_bad_download_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "models.pt"
            target.write_bytes(b"existing checkpoint")
            with patch(
                "stanza_onnx.cache.urlopen", return_value=io.BytesIO(b"invalid")
            ):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download_checkpoint(
                        "https://example.invalid/model.pt", target, "0" * 32
                    )
            self.assertEqual(target.read_bytes(), b"existing checkpoint")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_pinned_catalog_written_and_cache_missing_is_actionable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_resources(root)
            self.assertTrue((root / "resources.json").is_file())
            with self.assertRaisesRegex(ValueError, "run.*download"):
                verify_cache("en", root)
            with self.assertRaisesRegex(ValueError, "no selected tokenizer"):
                verify_cache("ml", root)


if __name__ == "__main__":
    unittest.main()
