# Language bundles

A bundle is a directory containing `bundle.json` and flat runtime assets. The Python exporter produces this format for the Rust runtime. Bundles are independent of Fern and the Python exporter environment; ONNX graphs alone do not supply the complete preprocessing and decoding pipeline.

## Manifest version 1

```json
{
  "schemaVersion": 1,
  "language": "en",
  "languageName": "English",
  "profile": "default_fast",
  "stanzaVersion": "1.11.0",
  "resourceVersion": "1.11.0",
  "runtimeVersion": "0.2.0",
  "unicodeVersion": "18.0.0",
  "offsetUnit": "unicode-code-point",
  "processors": {
    "tokenize": "combined",
    "mwt": "combined",
    "pos": "combined_nocharlm",
    "lemma": "combined_nocharlm"
  },
  "capabilities": {
    "tokenize": true,
    "sentenceSplit": true,
    "multiwordExpansion": true,
    "pos": true,
    "morphology": true,
    "lemma": "trained"
  },
  "files": [
    { "path": "tokenizer.onnx", "size": 123, "sha256": "64 lowercase hexadecimal digits" }
  ],
  "sources": [
    {
      "processor": "tokenize",
      "package": "combined",
      "url": "pinned upstream checkpoint URL",
      "md5": "upstream checksum",
      "license": "NOASSERTION"
    }
  ]
}
```

Sizes, hashes and URLs in this example are placeholders. Real bundles enumerate every ONNX graph, processor metadata JSON and external embedding table they use. Export diagnostics are omitted. Sources include checkpoint dependencies such as pretrained vectors and character language models. ONNX Runtime shared libraries are deployed separately and do not belong to the language bundle.

Only selected source packages are supported; catalog presence alone does not imply a usable pipeline. Missing processors are omitted. `capabilities.lemma` is `trained`, `identity` or `none`. Identity lemmas copy the written word without morphological normalization. Converter metadata for an omitted processor uses `kind: "none"`; `kind: "identity"` is reserved for an explicitly selected identity lemma processor. The runtime requires a tokenizer pipeline; no stock Bengali bundle or fallback tokenizer is supplied.

## Loading and integrity

Loading checks schema/runtime compatibility, capability consistency, required assets, decoder metadata and portable flat asset paths. Default loading also verifies sizes and SHA256 content hashes. The current compatibility series is runtime 0.2.x with schema version 1. The native runtime requires `unicodeVersion: "18.0.0"` to match its generated preprocessing tables and rejects runtime 0.1/Unicode15 bundles.

Regenerate bundle metadata in a fresh destination using existing converted graphs:

```sh
python -m stanza_onnx bundle --lang en --cache-dir .cache/models \
  --artifacts /path/to/converted-assets/en --output .cache/bundles/en-unicode18
```

The graph weights do not change. This upgrade requires fresh bundle metadata and fresh reference outputs; retraining or re-exporting neural weights is unnecessary.

For a trusted local cache, native CLI callers can use `--skip-integrity`, and Rust library callers can pass `false` for `Pipeline::load`'s `verify_integrity` argument. These options skip content hashing; asset, size and metadata checks still run. The JavaScript subprocess adapter currently uses the native default integrity check.

## Source alignment and Unicode

Bundle source offsets count Unicode code points in the original input. The native runtime can convert final offsets to UTF-16 code units or UTF-8 byte positions. The native CLI selects them through `--offset-unit`; the persistent protocol and JavaScript adapter use `offsetUnit`. Accepted values are `unicode-code-point`, `utf16` and `utf8`. Token `start`/`end` positions are half-open. None of these units equals a grapheme cluster.

Rust and the Python reference use identical tables generated directly from checksum-pinned official Unicode Character Database 18.0.0 files. These cover Python-style whitespace and character predicates, decimal features, default casing and contextual Greek FinalSigma. Default casing is language-neutral; it does not introduce locale-specific Turkish or Lithuanian rules. Table generation and these operations do not depend on the host Python or Rust Unicode database.

Verification runs Stanza 1.11.0 with an explicit, scoped adaptation of Unicode-dependent preprocessing. It retains the selected model processors and weights, supplies Unicode18 string operations and regex classes at audited boundaries, and restores the adapted module attributes afterward. References identify `referenceMode: "stanza-unicode18-preprocessing"`, `unicodeVersion: "18.0.0"`, the UCD/table hashes and the host interpreter's Unicode version. Native parity requires the Unicode version to match; it does not treat an unadapted host-Unicode Stanza output as the Unicode18 reference. Passing fixture parity does not establish equivalence for every Unicode input.

Original input is not NFC/NFKC-normalized before inference. Dictionary/search normalization can be implemented separately while preserving source alignment. Word/grapheme utilities and fallback tokenizers remain separate capabilities.

Tokenizer metadata retains the vocabulary's actual `lang_replaces_spaces` flag. When absent, it defaults to `false`, matching Stanza 1.11 checkpoint loading. The runtime does not infer this policy from a Chinese, Japanese or Korean language code; internal token spaces and source offsets follow the loaded model's behavior.

## Native deployment

Deploy a native executable or embedded Rust library, one or more language bundles and the compatible ONNX Runtime library/dependencies for the target OS and architecture. Python and JavaScript are not required for the native inference process. The optional JavaScript adapter invokes a persistent native subprocess and leaves models loaded across requests. It has no runtime npm dependencies.

`scripts/setup_ort.py` obtains checksum-verified ONNX Runtime 1.23.2 CPU distributions and retains their license/notice files. It is a preparation helper, not an inference dependency. Native library selection uses `--ort-library`, `ORT_DYLIB_PATH` or supported filenames beside the executable. CoreML/DirectML selections require provider-enabled libraries and their dependencies; the CPU setup helper does not promise accelerated execution.

The Rust library initializes ONNX Runtime once per process. Explicit `initialize_library(Some(path))` chooses the library before pipelines load; otherwise loading discovers the environment or adjacent installation. Dropping individual pipelines releases their sessions and does not unload the process-global runtime.

## Distribution and licenses

Model bundles are generated or downloaded separately from project code. `sources[].license` records resource provenance, not the code license. `NOASSERTION` represents an unresolved license assessment. Model conversion does not change upstream terms, and no blanket redistribution claim is made for exported models or embeddings. Audit each selected resource before publishing its bundle.

Retain applicable native dependency notices when shipping ONNX Runtime, and model notices when shipping language bundles. Binary model releases are not part of the initial source bootstrap.

The native API accepts well-formed Unicode scalar text. The JavaScript adapter rejects unpaired UTF-16 surrogates before sending a request, without replacing the input or closing the process.
