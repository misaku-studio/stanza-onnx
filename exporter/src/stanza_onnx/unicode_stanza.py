"""Scoped, explicit Unicode18 adaptation of the selected Stanza 1.11 processors.

No neural model/trainer implementation is replaced. The adaptation owns only
character normalization boundaries, Unicode-sensitive regexes and string input
carriers. It restores the few patched module attributes after each reference.
"""

from contextlib import contextmanager
from functools import wraps
import re
import threading

from . import STANZA_VERSION
from .unicode import UnicodeText, compile_regex, verify_host_whitespace

_LOCK = threading.RLock()


class _UnicodeRegex:
    """Module-local regex proxy; the interpreter's global re module is untouched."""

    def compile(self, pattern, flags=0):
        return compile_regex(pattern, flags)

    def __getattr__(self, name):
        return getattr(re, name)


def _wrap_document(document):
    """Restore carriers after stock joins and dictionary lookups create strings."""
    if document.text is not None:
        document.text = UnicodeText(document.text)
    for sentence in document.sentences:
        for token in sentence.tokens:
            token.text = UnicodeText(token.text)
            for word in token.words:
                word.text = UnicodeText(word.text)
    return document


@contextmanager
def adapted_stanza():
    """Apply the audited tokenizer Unicode hooks for one reference operation."""
    import stanza
    from stanza.models.tokenization import data, utils
    from stanza.models.common import doc

    if stanza.__version__ != STANZA_VERSION:
        raise ValueError(f"Unicode18 adaptation is audited for Stanza {STANZA_VERSION}")
    with _LOCK:
        verify_host_whitespace()
        original_init = data.TokenizationDataset.__init__

        @wraps(original_init)
        def initialize(dataset, *args, **kwargs):
            original_init(dataset, *args, **kwargs)
            # Stanza builds features later in para_to_sentences. Thus normalized
            # units can carry new Unicode semantics without copying that method.
            paragraphs = []
            for paragraph in dataset.data:
                characters = [
                    (UnicodeText(character), label) for character, label in paragraph
                ]
                paragraphs.append(characters)
            dataset.data = paragraphs

        replacements = [(data.TokenizationDataset, "__init__", initialize)]
        for module, names in (
            (data, ("NEWLINE_WHITESPACE_RE", "NUMERIC_RE", "WHITESPACE_RE")),
            (utils, ("SPACE_RE", "MASK_RE")),
        ):
            for name in names:
                replacements.append(
                    (module, name, compile_regex(getattr(module, name)))
                )
        # These modules compile whitespace alignment expressions dynamically.
        replacements.extend(
            [(utils, "re", _UnicodeRegex()), (doc, "re", _UnicodeRegex())]
        )
        originals = [
            (owner, name, getattr(owner, name)) for owner, name, _ in replacements
        ]
        try:
            # Only these audited module attributes change; nested operations are
            # serialized and every attribute is restored even after an error.
            for owner, name, replacement in replacements:
                setattr(owner, name, replacement)
            yield
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)


def _process_mwt(processor, document):
    # predict_dict uppercases dictionary expansion values, not just the input
    # word. Keys retain exact checkpoint lookup semantics.
    trainer = processor.trainer
    original = trainer.expansion_dict
    trainer.expansion_dict = {
        key: UnicodeText(value) for key, value in original.items()
    }
    try:
        return processor.process(document)
    finally:
        trainer.expansion_dict = original


def analyze(pipeline, text):
    """Use stock model processors in their normal order with Unicode18 text.

    The selected pipeline contains tokenize/mwt/pos/lemma only. Document fields
    are wrapped at each boundary because str.join and vocabulary/dictionary
    returns may otherwise create host-Unicode strings.
    """
    document = UnicodeText(text)
    for name in ("tokenize", "mwt", "pos", "lemma"):
        processor = pipeline.processors.get(name)
        if processor is None:
            continue
        if name != "tokenize":
            _wrap_document(document)
        if name == "mwt":
            document = _process_mwt(processor, document)
        else:
            document = processor.process(document)
    return document
