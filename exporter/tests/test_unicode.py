import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from stanza_onnx import UNICODE_VERSION
from stanza_onnx.unicode import (
    UnicodeText,
    compile_regex,
    has_property,
    provenance,
    tables,
    verify_host_whitespace,
)


class Unicode18Tests(unittest.TestCase):
    def test_explicit_version_and_source_provenance(self):
        data = provenance()
        self.assertEqual(UNICODE_VERSION, "18.0.0")
        self.assertEqual(data["version"], UNICODE_VERSION)
        self.assertRegex(data["sha256"], r"^[a-f0-9]{64}$")
        self.assertTrue(data["sources"])
        self.assertTrue(all("/18.0.0/" in source["url"] for source in data["sources"]))
        self.assertEqual(tables()["unicodeVersion"], "18.0.0")

    def test_new_casing_uses_tables_instead_of_host_unicode(self):
        garay = UnicodeText("\U00010d50\U00010d51")
        self.assertTrue(garay.isupper())
        self.assertEqual(garay.lower(), "\U00010d70\U00010d71")
        self.assertEqual(garay.lower().upper(), garay)
        self.assertTrue(UnicodeText("\U00010d50\U00010d71").istitle())
        self.assertTrue(UnicodeText("\U00010d70").islower())
        self.assertEqual(UnicodeText("\ua7dd").lower(), "\u0277")
        self.assertEqual(UnicodeText("\u0277").upper(), "\ua7dd")

    def test_final_sigma_uses_new_case_ignorable_and_cased_neighbors(self):
        mark = "\U00011df0"  # Bengali combining anusvara above, assigned in 18.
        self.assertTrue(has_property(mark, "caseIgnorable"))
        self.assertEqual(UnicodeText("A" + mark + "Σ").lower(), "a" + mark + "ς")
        self.assertEqual(UnicodeText("AΣ" + mark).lower(), "aς" + mark)
        self.assertEqual(
            UnicodeText("AΣ" + mark + "\ua7dd").lower(), "aσ" + mark + "\u0277"
        )
        # Cased+Case_Ignorable intersection is skipped before testing cased.
        self.assertEqual(UnicodeText("AΣ\u0558").lower(), "aς\u0558")

    def test_new_numeric_and_python_property_distinctions(self):
        for digit in ("\U00010d40", "\U00011de0"):
            self.assertTrue(UnicodeText(digit).isdecimal())
            self.assertTrue(UnicodeText(digit).isdigit())
            self.assertTrue(UnicodeText(digit).isnumeric())
        number = UnicodeText("\U0001246f")  # New Unicode18 cuneiform numeric sign.
        self.assertTrue(number.isnumeric())
        self.assertTrue(number.isalnum())
        self.assertFalse(number.isdigit())
        self.assertFalse(number.isdecimal())
        self.assertFalse(number.isalpha())
        self.assertFalse(UnicodeText("Ⅸ").isalpha())
        self.assertTrue(UnicodeText("Ⅸ").isalnum())
        self.assertTrue(UnicodeText("²").isdigit())
        self.assertFalse(UnicodeText("²").isdecimal())

    def test_carriers_survive_word_operations(self):
        word = UnicodeText(" \U00010d50Σ ")
        for value in (
            word[1],
            word[1:-1],
            word.strip(),
            word.replace(" ", ""),
            word.lower(),
            word.upper(),
            word.split()[0],
            "x" + word,
        ):
            self.assertIsInstance(value, UnicodeText)
        self.assertTrue(UnicodeText("\u001c\u2007").isspace())
        verify_host_whitespace()

    def test_known_characters_match_stock_string_semantics(self):
        methods = [
            "lower",
            "upper",
            "isupper",
            "islower",
            "isalpha",
            "isalnum",
            "isdecimal",
            "isdigit",
            "isnumeric",
            "isspace",
            "istitle",
            "capitalize",
            "title",
        ]
        for sample in (
            "",
            "Hello world",
            "ΟΣ ΟΣΑ Σ",
            "AΣ\u0301",
            "İstanbul",
            "straße ﬃ",
            "ǅuro",
            "Ⅸ",
            "²",
            "١٢",
            " \t\u001c",
        ):
            for name in methods:
                self.assertEqual(
                    getattr(UnicodeText(sample), name)(),
                    getattr(sample, name)(),
                    (sample, name),
                )

    def test_regex_classes_use_latest_tables_inside_and_outside_sets(self):
        garay = "\U00010d40"
        self.assertIsNotNone(compile_regex(r"^[\d]+$").match(garay))
        self.assertIsNotNone(compile_regex(r"^\d+$").match(garay))
        self.assertIsNotNone(compile_regex(r"^\w+$").match("\U0001246f_\U00010d50"))
        self.assertIsNotNone(compile_regex(r"^[^\d\W]+$").match("\U00010d50"))
        self.assertIsNone(compile_regex(r"^[^\d\W]+$").match(garay))
        self.assertIsNotNone(compile_regex(r"^\b\w+\b$").match("\U00010d50"))
        self.assertIsNotNone(compile_regex(r"^\S+$").match("\U00010d50"))
        self.assertIsNone(compile_regex(r"^\S+$").match(" "))
        self.assertIsNotNone(compile_regex(r"^\\d$").match(r"\d"))


@unittest.skipUnless(
    importlib.util.find_spec("stanza"),
    "Stanza is optional for model-free exporter tests",
)
class StanzaUnicodeHooksTests(unittest.TestCase):
    def test_checkpoint_loaded_cjk_vocab_normalization_is_preserved(self):
        from stanza.models.tokenization.vocab import Vocab
        from stanza_onnx.unicode_stanza import adapted_stanza

        # Stanza loads this vocabulary by constructing Vocab() with empty lang,
        # then restoring state. The derived flag is not in state_attrs and must
        # not be reconstructed from the restored language during adaptation.
        constructed = Vocab(lang="zh-hans")
        loaded = Vocab.load_state_dict(constructed.state_dict())
        self.assertTrue(constructed.lang_replaces_spaces)
        self.assertFalse(loaded.lang_replaces_spaces)
        token = UnicodeText(" \ua7dd \u0277")
        with adapted_stanza():
            self.assertEqual(loaded.normalize_token(token), "\ua7dd \u0277")
            self.assertEqual(constructed.normalize_token(token), "\ua7dd\u0277")
        self.assertFalse(loaded.lang_replaces_spaces)

    def test_actual_tokenizer_features_change_for_new_characters_and_hooks_restore(
        self,
    ):
        from stanza.models.tokenization import data
        from stanza_onnx.unicode_stanza import adapted_stanza

        args = {
            "feat_funcs": ["capitalized", "numeric"],
            "use_dictionary": False,
            "max_seqlen": 1000,
            "skip_newline": False,
        }
        vocab = SimpleNamespace(unit2id=lambda character: 1)
        text = "\U00010d50\U00010d40\U00011de0"
        original_init, original_regex = (
            data.TokenizationDataset.__init__,
            data.NUMERIC_RE,
        )
        before = data.TokenizationDataset(
            args, input_text=text, vocab=vocab, evaluation=True
        )
        before_features = before.para_to_sentences(before.data[0])[0][2].tolist()
        with adapted_stanza():
            dataset = data.TokenizationDataset(
                args, input_text=text, vocab=vocab, evaluation=True
            )
            features = dataset.para_to_sentences(dataset.data[0])[0][2].tolist()
            self.assertEqual(features, [[1, 0], [0, 1], [0, 1]])
            self.assertTrue(
                all(
                    isinstance(character, UnicodeText)
                    for character, _ in dataset.data[0]
                )
            )
        self.assertIs(data.TokenizationDataset.__init__, original_init)
        self.assertIs(data.NUMERIC_RE, original_regex)
        # Hosts already implementing these properties need not differ; the
        # expected table-driven feature values are still asserted above.
        if not text[0].isupper():
            self.assertNotEqual(features, before_features)

    def test_mwt_dictionary_outputs_and_lemma_edits_use_latest_casing(self):
        from stanza.models.mwt.trainer import Trainer
        from stanza.models.lemma.edit import edit_word
        from stanza_onnx.unicode_stanza import analyze

        upper, lower = "\U00010d50", "\U00010d70"
        original_dict = {lower + "foo": lower + " bar"}
        trainer = SimpleNamespace(expansion_dict=original_dict)
        word = SimpleNamespace(text=upper + "FOO")
        token = SimpleNamespace(text=word.text, words=[word])
        document = SimpleNamespace(
            text=word.text, sentences=[SimpleNamespace(tokens=[token])]
        )

        def expand(doc):
            word.text = Trainer.dict_expansion(trainer, word.text)
            return doc

        pipeline = SimpleNamespace(
            processors={
                "tokenize": SimpleNamespace(process=lambda text: document),
                "mwt": SimpleNamespace(trainer=trainer, process=expand),
            }
        )
        result = analyze(pipeline, word.text)
        self.assertEqual(result.sentences[0].tokens[0].words[0].text, upper + " BAR")
        self.assertIs(trainer.expansion_dict, original_dict)
        self.assertEqual(
            edit_word(UnicodeText("A\U00011df0Σ"), "ignored", 2), "a\U00011df0ς"
        )

    def test_scoped_hooks_restore_on_exception(self):
        from stanza.models.tokenization import data, utils
        from stanza.models.common import doc
        from stanza_onnx.unicode_stanza import adapted_stanza

        original = data.TokenizationDataset.__init__, utils.re, doc.re
        with self.assertRaisesRegex(RuntimeError, "example"):
            with adapted_stanza():
                raise RuntimeError("example")
        self.assertEqual(
            (data.TokenizationDataset.__init__, utils.re, doc.re), original
        )


@unittest.skipUnless(
    os.environ.get("STANZA_ONNX_TEST_MODEL_CACHE"),
    "Requires separately prepared English models",
)
class StanzaUnicodeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import stanza
        import torch
        from stanza_onnx.catalog import language_profile

        torch.set_num_threads(1)
        cls.pipeline = stanza.Pipeline(
            "en",
            dir=os.environ["STANZA_ONNX_TEST_MODEL_CACHE"],
            package=None,
            processors=language_profile("en")["processors"],
            use_gpu=False,
            download_method=None,
            verbose=False,
        )

    def test_existing_authored_inputs_match_unmodified_stanza(self):
        from stanza_onnx.reference import normalize
        from stanza_onnx.unicode_stanza import adapted_stanza, analyze

        fixture = Path(__file__).resolve().parents[2] / "fixtures" / "en.json"
        for case in json.loads(fixture.read_text(encoding="utf-8"))["texts"]:
            stock = normalize(self.pipeline(case["text"]))
            with adapted_stanza():
                adapted = normalize(analyze(self.pipeline, case["text"]))
            self.assertEqual(adapted, stock, case["id"])

    def test_latest_unicode_inputs_keep_raw_source_offsets_through_all_models(self):
        from stanza_onnx.reference import normalize
        from stanza_onnx.unicode_stanza import adapted_stanza, analyze

        for text in (
            "\U00010d50\U00010d51 reads \U00010d40 books.",
            "A\U00011df0Σ AΣ\U00011df0\ua7dd.",
            "The count is \U00011de0 and \U0001246f.",
        ):
            with adapted_stanza():
                document = normalize(analyze(self.pipeline, text))
            self.assertTrue(document)
            for sentence in document:
                for token in sentence["tokens"]:
                    self.assertEqual(text[token["start"] : token["end"]], token["text"])


if __name__ == "__main__":
    unittest.main()
