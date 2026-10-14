# Bengali rule tokenizer

The Bengali pack is a downstream rule implementation, not a pretrained Stanza
pipeline. It supplies word boundaries, conservative sentence boundaries and
half-open source offsets. Each token has one word with `upos`, `xpos`, `feats`
and `lemma` set to `null`. There is no multiword expansion or morphological
analysis. The pinned Stanza catalog has Bengali character-language-model
resources but no tokenizer/POS/lemma pipeline used by this pack.

## Rule contract

The package is `bengali-rule-v1`, with `rules_version: 1`,
`tokenizerKind: "bengali-rule"` and Unicode 18.0.0 character properties. Input is
preserved without Unicode normalization. The rules follow the practical Bengali
convention that words generally use whitespace and punctuation is tokenized
separately. [Universal Dependencies Bengali guidelines](https://universaldependencies.org/bn/)

- Whitespace-only grapheme clusters separate tokens. Bengali words retain
  combining marks and joiners. A mixed cluster such as space plus a combining
  mark is retained whole to preserve its source and grapheme boundary.
- Punctuation becomes separate tokens. A period or comma directly between
  decimal digits stays inside the number, such as `১২.৫` or `1,000`.
- Danda, double danda, `?`, `!` and terminal periods end sentences. Contiguous
  terminal punctuation and closing quotes/brackets stay in the preceding
  sentence. Periods between ASCII Latin letters split tokens without ending a
  sentence; this handles simple domains and abbreviations only.
- A single line break is whitespace. Blank lines and U+2029 paragraph separators
  end the current sentence.
- Tokens start and end at Unicode 18 extended grapheme boundaries from the
  pinned UNIC fork, preserving Indic conjuncts, combining sequences, flags and
  joined emoji. Word and sentence rules are tailored separately.

Examples:

| Input | Sentence tokens |
| --- | --- |
| `আমি বাংলা শিখছি।` | `আমি · বাংলা · শিখছি · ।` |
| `মূল্য ১২.৫ টাকা।` | `মূল্য · ১২.৫ · টাকা · ।` |
| `সে বলে, “যাব!”` | `সে · বলে · , · “ · যাব · ! · ”` |
| `ড. রহমান আসেন।` | `ড · .` / `রহমান · আসেন · ।` |

The last example is an intentional limitation: Bengali abbreviations are not
recognized. URLs, email addresses, initials and punctuation in quotations also
need more sophisticated handling. No accuracy claim is made for the authored
fixtures. Unicode distinguishes word, sentence and grapheme boundaries and
allows language tailoring; these word/sentence rules do not claim full UAX #29 conformance.
[Unicode text segmentation](https://www.unicode.org/reports/tr29/)

## Export and run without ML dependencies

From the checkout, set `PYTHONPATH` to `exporter/src`. Then Python's standard
library alone can create the metadata-only bundle:

```sh
python -S -m stanza_onnx export --lang bn --output .cache/bundles/bn
```

Use a fresh or empty output directory. No model cache, checkpoint download,
Stanza, Torch or ONNX installation is needed. The bundle contains `bundle.json`
and `tokenizer.json`; the rule implementation lives in the native runtime.

```sh
cargo build --locked -p stanza-onnx
target/debug/stanza-onnx --model-dir .cache/bundles/bn analyze \
  --text "আমি বাংলা শিখছি।" --offset-unit utf16
```

On Windows, use `target/debug/stanza-onnx.exe`. No ONNX Runtime shared library is
required for this pack. Native `analyze`, `batch` and `serve` use the same rules;
`analyze` and persistent requests support `unicode-code-point`, `utf16` and
`utf8` offsets. Batch output uses code-point offsets. The Node adapter can omit
`ortLibrary` when loading a rule-only bundle.

## Validate the rule contract

```sh
python scripts/test_bengali.py --binary target/debug/stanza-onnx \
  --bundle .cache/bengali-check
npm ci
npm run build:tests
node .cache/adapter-tests/adapters-tests/bengali.js \
  --binary target/debug/stanza-onnx --bundle .cache/bengali-check
```

The checker exports with Python `-S`, poisons `ORT_DYLIB_PATH`, compares authored
token/sentence expectations, verifies all three offset units against original
source slices, and exercises persistent reuse, request errors and shutdown.
The Node check uses the real executable and omits `ortLibrary`. Neither check
uses Stanza parity. `reference --lang bn` explicitly fails because no pretrained
Stanza pipeline can provide the corresponding reference.

Segmentation uses the Unicode 18 fork of [rust-unic](https://github.com/eyeplum/rust-unic/tree/d0a257bc7c21a48556d9bb204abdcb55ebe4c52b),
pinned to that full Git revision. It is not the older crates.io `unic-segment 0.9`
release. Source builds fetch the pinned Git dependency; deployment still needs
only the compiled native runtime. Publishing the Rust crate to crates.io would
require an appropriate Unicode 18 release of that dependency first.
