"""Explicit Unicode 18 string operations for Stanza reference preprocessing.

The generated UCD tables are shared with Rust. These operations deliberately do
not depend on the interpreter's Unicode database or replace Python's builtins.
This module supplies the operations audited in the selected Stanza processors;
it is not a replacement for the complete unicodedata module.
"""

from bisect import bisect_right
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import re

from . import UNICODE_VERSION


@lru_cache(maxsize=1)
def tables():
    data = json.loads(
        files("stanza_onnx").joinpath("data/unicode18.json").read_text(encoding="utf-8")
    )
    if data["unicodeVersion"] != UNICODE_VERSION:
        raise ValueError("Unicode table version does not match exporter policy")
    if data["conditionalLower"] != [
        {"codepoint": 931, "condition": "Final_Sigma", "lower": "ς"}
    ]:
        raise ValueError("Unsupported language-independent conditional casing rule")
    return data


def provenance():
    raw = files("stanza_onnx").joinpath("data/unicode18.json").read_bytes()
    return {
        "version": UNICODE_VERSION,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sources": tables()["sources"],
    }


@lru_cache(maxsize=None)
def _ranges(property_name):
    ranges = tables()[property_name]
    return ranges, [start for start, _ in ranges]


def has_property(character, property_name):
    ranges, starts = _ranges(property_name)
    code = ord(character)
    index = bisect_right(starts, code) - 1
    return index >= 0 and code <= ranges[index][1]


def _all(text, property_name):
    return bool(text) and all(
        has_property(character, property_name) for character in text
    )


def _cased_all(text, property_name):
    cased = [character for character in text if has_property(character, "cased")]
    return bool(cased) and all(
        has_property(character, property_name) for character in cased
    )


def _first_non_ignorable(characters):
    """Final_Sigma ignores marks before asking whether a neighbor is cased.

    A character can be both Cased and Case_Ignorable, so the order of these
    checks matters. The shared Rust tables apply the same contextual rule.
    """
    for character in characters:
        if not has_property(character, "caseIgnorable"):
            return character
    return None


def _is_final_sigma(text, index):
    before = _first_non_ignorable(reversed(text[:index]))
    after = _first_non_ignorable(text[index + 1 :])
    return (
        before is not None
        and has_property(before, "cased")
        and (after is None or not has_property(after, "cased"))
    )


def _lower_units(text):
    mapping = tables()["lower"]
    result = []
    for index, character in enumerate(text):
        if character == "Σ":
            result.append("ς" if _is_final_sigma(text, index) else "σ")
        else:
            result.append(mapping.get(str(ord(character)), character))
    return result


def lower(text):
    return "".join(_lower_units(text))


def upper(text):
    mapping = tables()["upper"]
    return "".join(mapping.get(str(ord(character)), character) for character in text)


class UnicodeText(str):
    """String carrier keeping Unicode 18 semantics at audited Stanza boundaries."""

    def __getitem__(self, key):
        return UnicodeText(super().__getitem__(key))

    def __iter__(self):
        return (UnicodeText(character) for character in super().__iter__())

    def lower(self):
        return UnicodeText(lower(str(self)))

    def upper(self):
        return UnicodeText(upper(str(self)))

    def isupper(self):
        return _cased_all(self, "uppercase")

    def islower(self):
        return _cased_all(self, "lowercase")

    def isdecimal(self):
        return _all(self, "decimal")

    def isdigit(self):
        return _all(self, "digit")

    def isnumeric(self):
        return _all(self, "numeric")

    def isalpha(self):
        return _all(self, "alpha")

    def isalnum(self):
        return _all(self, "alnum")

    def isspace(self):
        return _all(self, "whitespace")

    def istitle(self):
        previous_cased = False
        seen_cased = False
        for character in self:
            if has_property(character, "uppercase") or has_property(
                character, "titlecase"
            ):
                if previous_cased:
                    return False
                previous_cased = True
                seen_cased = True
            elif has_property(character, "lowercase"):
                if not previous_cased:
                    return False
                previous_cased = True
                seen_cased = True
            else:
                previous_cased = False
        return seen_cased

    def capitalize(self):
        if not self:
            return UnicodeText("")
        first = tables()["title"].get(str(ord(self[0])), self[0])
        return UnicodeText(first + "".join(_lower_units(str(self))[1:]))

    def title(self):
        mapping = tables()["title"]
        lowered = _lower_units(str(self))
        result = []
        previous_cased = False
        for index, character in enumerate(self):
            if previous_cased:
                result.append(lowered[index])
            else:
                result.append(mapping.get(str(ord(character)), character))
            previous_cased = has_property(character, "cased")
        return UnicodeText("".join(result))

    def __add__(self, other):
        result = super().__add__(other)
        return UnicodeText(result) if result is not NotImplemented else result

    def __radd__(self, other):
        if isinstance(other, str):
            return UnicodeText(str.__add__(other, self))
        return NotImplemented

    def replace(self, old, new, count=-1):
        return UnicodeText(super().replace(old, new, count))

    def join(self, iterable):
        return UnicodeText(super().join(iterable))

    def strip(self, chars=None):
        return self.lstrip(chars).rstrip(chars)

    def lstrip(self, chars=None):
        if chars is not None:
            return UnicodeText(super().lstrip(chars))
        index = 0
        while index < len(self) and has_property(self[index], "whitespace"):
            index += 1
        return self[index:]

    def rstrip(self, chars=None):
        if chars is not None:
            return UnicodeText(super().rstrip(chars))
        end = len(self)
        while end > 0 and has_property(self[end - 1], "whitespace"):
            end -= 1
        return self[:end]

    def split(self, sep=None, maxsplit=-1):
        # Explicit separators use no Unicode classification; preserve carriers.
        # Default whitespace has the same stable 29-codepoint set in UCD 18 as
        # supported Python hosts. The scoped shim verifies that set explicitly.
        return [UnicodeText(value) for value in super().split(sep, maxsplit)]

    def rsplit(self, sep=None, maxsplit=-1):
        return [UnicodeText(value) for value in super().rsplit(sep, maxsplit)]


def verify_host_whitespace():
    """Audit ordinary-string strip/split operations that Stanza creates via join.

    Casing and numbers changed and are fully table-driven. Whitespace has not:
    this check prevents an unadapted strip/split from silently using another set.
    """
    expected = {
        code for start, end in tables()["whitespace"] for code in range(start, end + 1)
    }
    actual = {code for code in range(0x110000) if chr(code).isspace()}
    if actual != expected:
        raise ValueError(
            "Host whitespace differs from UCD 18; ordinary Stanza strip/split operations need adaptation"
        )


@lru_cache(maxsize=None)
def regex_class(property_name, inverse=False):
    ranges = tables()[property_name]
    if property_name == "alnum":
        # The underscore is the sole non-alphanumeric addition to Python \w.
        ranges = sorted([*ranges, [95, 95]])
    if inverse:
        complement = []
        next_code = 0
        for start, end in ranges:
            if start > next_code:
                complement.append([next_code, start - 1])
            next_code = max(next_code, end + 1)
        if next_code <= 0x10FFFF:
            complement.append([next_code, 0x10FFFF])
        ranges = complement
    return "".join(
        f"\\U{start:08x}" if start == end else f"\\U{start:08x}-\\U{end:08x}"
        for start, end in ranges
    )


def rewrite_regex(pattern):
    """Replace Unicode classes explicitly, including classes inside brackets."""
    output = []
    in_class = False
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\" and index + 1 < len(pattern):
            code = pattern[index + 1]
            if code in "dDsSwW":
                property_name = {"d": "decimal", "s": "whitespace", "w": "alnum"}[
                    code.lower()
                ]
                body = regex_class(property_name, inverse=code.isupper())
                output.append(body if in_class else f"[{body}]")
            elif code in "bB" and not in_class:
                word = f"[{regex_class('alnum')}]"
                boundary = f"(?:(?<={word})(?!{word})|(?<!{word})(?={word}))"
                output.append(boundary if code == "b" else f"(?!{boundary})(?!\\A\\Z)")
            else:
                output.append(pattern[index : index + 2])
            index += 2
            continue
        if character == "[":
            in_class = True
        elif character == "]":
            in_class = False
        output.append(character)
        index += 1
    return "".join(output)


def compile_regex(pattern, flags=0):
    if isinstance(pattern, re.Pattern):
        if flags:
            raise ValueError("Cannot process compiled regex with additional flags")
        flags = pattern.flags
        pattern = pattern.pattern
    if not isinstance(pattern, str):
        raise TypeError("Unicode18 reference regexes must operate on text")
    if flags & re.IGNORECASE:
        raise ValueError(
            "Unicode18 IGNORECASE adaptation is not audited for selected Stanza processors"
        )
    if not flags & re.ASCII:
        pattern = rewrite_regex(pattern)
    return re.compile(pattern, flags)
