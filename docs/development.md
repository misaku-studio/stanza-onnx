# Development

The repository separates model preparation from deployment:

| Location | Responsibility |
| --- | --- |
| `exporter/` | Python package, pinned Stanza resources, ONNX converters, download/export/bundle/reference CLI |
| `crates/stanza-onnx/` | Native Rust runtime, processor implementations, shared-library loading and CLI |
| `adapters/node/` | Optional Node.js process adapter, shipped as compiled ESM and declarations |
| `scripts/`, `fixtures/` | Native verification tooling; checked-in integration text is newly authored |
| `adapters-tests/`, `exporter/tests/` | Model-free adapter and preparation tests |

The native executable and Rust library have no Python, JavaScript or Fern dependency during inference. ONNX Runtime and model assets remain native deployment dependencies. The optional JavaScript adapter uses standard Node.js process APIs.

The shared lemma/MWT sequence decoder is in `crates/stanza-onnx/src/lemma_mwt/sequence.rs`; processor orchestration remains in `lemma_mwt.rs`. POS and lemma/MWT tests live in their own `tests.rs` modules. Python converters separate neural wrapper math from loading, exporting, validation and metadata assembly.

Library initialization is process-global. Call `initialize_library(Some(path))` before the first pipeline when selecting a specific deployment library. Otherwise pipeline loading uses `ORT_DYLIB_PATH` or a supported adjacent library. Subsequent pipelines share that initialized runtime.

## Unit and build checks

Use Rust 1.85 or newer, Python 3.12 or newer for exporter tooling, and Node.js 22 or newer for the optional adapter. ML integration uses Python 3.12; model-free CI also checks Python 3.14. Run from the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --release --locked -p stanza-onnx

python -m pip install -e "./exporter[dev]"
python -m black --check exporter/src exporter/tests scripts
python -m unittest discover -s exporter/tests -v

npm ci
npm run format:js:check
npm run build
npm test
npm run typecheck
npm run test:tools
```

`python -m black exporter/src exporter/tests scripts` and `npm run format:js` apply the checked-in formatting rules. CI checks Python, JavaScript/TypeScript and Rust formatting. Generated catalog and Unicode JSON files retain their compact machine-generated form.

`npm ci` installs development dependencies from the checked-in `package-lock.json`. The package contains compiled ESM JavaScript and declarations, without runtime npm dependencies.

Normal Rust and adapter unit tests do not require model downloads or Python. An ignored POS metadata audit needs separately exported bundles; run it with `STANZA_ONNX_POS_BUNDLE_ROOT` pointing to a directory of language bundles. Optional integration checks are additional to model-free unit tests.

## Native English parity

Start from a clean model cache and empty bundle destination:

```sh
python -m stanza_onnx download --lang en --cache-dir .cache/models
python -m stanza_onnx export --lang en --cache-dir .cache/models --output .cache/bundles/en
python -m stanza_onnx reference --lang en --cache-dir .cache/models \
  --input fixtures/en.json --output .cache/references/en/reference.json
python scripts/setup_ort.py --output .cache/ort
```

Use the `libraryPath` returned by setup. These commands show the macOS library filename:

```sh
python scripts/native_parity.py \
  --binary target/release/stanza-onnx \
  --ort-library .cache/ort/libonnxruntime.1.23.2.dylib \
  --bundles-dir .cache/bundles \
  --references-dir .cache/references \
  --output-dir .cache/native-parity --languages en

python scripts/test_native_protocol.py \
  --binary target/release/stanza-onnx \
  --ort-library .cache/ort/libonnxruntime.1.23.2.dylib \
  --bundle .cache/bundles/en
```

For Linux, substitute `libonnxruntime.so.1.23.2`; on Windows, substitute `onnxruntime.dll` and the `.exe` executable. The setup helper supports `--platform` for preparing another deployment target. It installs CPU distributions; provider-specific libraries and dependencies must be supplied separately when selecting acceleration.

The parity runner compares complete sentence/token/word annotations and metadata with saved references from the explicit Stanza 1.11.0 Unicode18 adaptation, writes per-language reports and exits with a failing status on differences or runner errors. References and native output must identify `unicodeVersion: "18.0.0"`; regenerate older references. The English protocol check verifies persistent requests, error recovery, empty input, UTF-16 offsets and acknowledged shutdown. Python runs these checks as development tooling; the native process itself uses no Python.

## Adapter integration

After preparing the English bundle and native library above, build the adapter tests and run the model-dependent check under Node.js:

```sh
npm run build:tests
node ./.cache/adapter-tests/adapters-tests/integration.js \
  --binary target/release/stanza-onnx \
  --ort-library .cache/ort/libonnxruntime.1.23.2.dylib \
  --bundle .cache/bundles/en
```

For Windows, use the `.exe` executable and `onnxruntime.dll`. This checks persistent analysis, UTF-16 spans, lemmas, argument-error recovery and graceful shutdown. The adapter uses standard Node.js built-ins and starts the Rust executable directly without a shell.

## Architecture matrix

After changing converters or shared processor behavior, validate representative paths before the full catalog:

| Language | Path covered |
| --- | --- |
| English | Classifier-based multiword expansion |
| French | Sequence-decoder multiword expansion |
| Dutch | Dictionary-based multiword expansion |
| Simplified Chinese | Dictionary-feature/CJK tokenization |
| Thai | Contextual POS and identity lemmas |
| Sindhi | Contextual POS and trained lemmas |
| Myanmar | Tokenizer-only pipeline |

Download/export/reference each language using its checked-in `fixtures/LANG.json`, then point the native parity runner at the resulting directories. In a POSIX shell:

```sh
for language in en fr nl zh-hans th sd my; do
  python -m stanza_onnx download --lang "$language" --cache-dir .cache/models
  python -m stanza_onnx export --lang "$language" --cache-dir .cache/models \
    --output ".cache/matrix-bundles/$language"
  python -m stanza_onnx reference --lang "$language" --cache-dir .cache/models \
    --input "fixtures/$language.json" --output ".cache/matrix-references/$language/reference.json"
done
python scripts/native_parity.py \
  --binary target/release/stanza-onnx \
  --ort-library .cache/ort/libonnxruntime.1.23.2.dylib \
  --bundles-dir .cache/matrix-bundles \
  --references-dir .cache/matrix-references \
  --output-dir .cache/native-matrix --workers 3
```

Choose a fresh bundle destination when repeating exports; the exporter refuses to overwrite nonempty directories. The native parity reports can be regenerated in an existing report destination.

To check all selected pipelines with privately prepared bundles and references:

```sh
python scripts/native_parity.py \
  --binary target/release/stanza-onnx \
  --ort-library /path/to/native/onnxruntime-library \
  --bundles-dir /path/to/all-language-bundles \
  --references-dir /path/to/stanza-references \
  --output-dir .cache/native-full --workers 3
```

Each reference lives at `LANG/reference.json` and must contain a nonempty `parity` array. The runner checks every language with a reference file unless `--languages` limits the selection. Preparing all models requires substantial download and storage capacity; keep binaries and third-party corpus samples outside Git.

`fixtures/unicode.json` contains newly authored classification, casing and offset checks for Unicode16–18 additions and existing combining/astral text. Run it through the same `reference` command and native parity runner for representative pipelines. These synthetic strings check implementation behavior; they are not native-language accuracy data. The model integration workflow includes this input for English.

The sweep command prepares models and runs native parity in one operation, without a JavaScript runtime:

```sh
python scripts/sweep.py \
  --cache-dir .cache/models \
  --artifacts-dir .cache/native-artifacts \
  --bundles-dir .cache/native-sweep-bundles \
  --output-dir .cache/native-sweep-reports \
  --native-binary target/release/stanza-onnx \
  --ort-library .cache/ort/libonnxruntime.1.23.2.dylib \
  --download --languages en fr nl zh-hans th sd my
```

Use `--all` in place of `--languages` for every eligible catalog selection, or `--skip-export` with existing converted artifacts. Keep output bundle and artifact destinations fresh.

## Native CPU batching

The native CPU POS runtime sorts whole sentences by word count and limits each batch to 128 padded word positions. The ordinary word-character encoder separately sorts inputs by character length, limits each batch to 128 padded character positions and deduplicates exact word text within that POS batch. Feature scattering preserves source order and every repeated occurrence. Oversized individual sentences/words stay whole; these limits do not truncate text or split model context.

This policy is private to the native CPU implementation. CoreML and DirectML use a 5,000-position sentence budget and single character dispatch without deduplication until their dispatch costs are measured. Re-exporting models is unnecessary. Changes to batching still require label/offset parity checks because different FP32 batch shapes can affect numerical results. See [validation](validation.md#native-cpu-performance) for the measured English effect and its limits.

## Compatibility and versions

Keep Rust, Python and JavaScript package versions aligned. Runtime 0.3 accepts Stanza bundles from runtime 0.2.x and 0.3.x; Bengali rule bundles require runtime 0.3.x with schema version 1 and `unicodeVersion: "18.0.0"`; runtime 0.1/Unicode15 bundles are rejected. Reassemble bundles from existing exported graphs into a fresh destination and regenerate references when adopting this contract. The model weights do not need re-exporting. Upgrade pinned Stanza/resources explicitly and regenerate references before claiming parity with a new upstream release. Catalog presence and export support are different: new architecture variants require converter/runtime work.

Generate the shared native/reference tables directly from official Unicode 18.0.0 `ReadMe.txt`, `UnicodeData.txt`, `SpecialCasing.txt`, `DerivedCoreProperties.txt` and `extracted/DerivedNumericType.txt`. Their source URLs and SHA256 values are pinned in `scripts/generate_unicode.py` and retained in the generated data. The interpreter's built-in Unicode database does not generate these tables.

```sh
python scripts/generate_unicode.py --cache-dir /path/to/ucd18-cache --check
```

The generator fetches only missing source files, verifies every checksum and verifies the final release marker. `--check` reproduces both checked-in table assets without changing them. Reference preprocessing uses the same data through a scoped Stanza 1.11.0 adaptation; neural models/trainers and weights are unchanged. References record their adaptation mode and Unicode/table provenance explicitly. Character predicates, default casing, FinalSigma context and Unicode-sensitive regexes must keep the shared semantics. Unicode code-point, UTF-16, UTF-8 byte and grapheme offsets remain distinct, and source text is not NFC/NFKC-normalized.

CPU is the validated native execution path. English CPU model/adapter integration passed on macOS and Windows. Native CoreML/DirectML performance remains to be established. Measure acceleration on target hardware before changing the default provider.

## Future work

- Published package validation and optional acceleration measurements.
- General Unicode word/grapheme utilities and broader Bengali abbreviation/URL rules.
- Broader raw-input coverage and oversized individual sentence handling.
- Additional Stanza processors and transformer profiles.
- Audited model redistribution metadata and hosted language bundles.

Keep fixtures, source provenance and reproducible commands with each behavior change. The initial API remains experimental.

The downstream Bengali rule pack has no stock Stanza tokenizer reference.
`scripts/sweep.py --all` selects only the 91 Stanza pipelines; check custom
Bengali rules with `scripts/test_bengali.py` and the real Node adapter check
described in [Bengali documentation](bengali.md). CI runs those checks without
installing ML dependencies or ONNX Runtime on Linux, macOS and Windows.

Bengali segmentation uses the Git-pinned Unicode 18 UNIC fork. Keep its exact
revision and `Cargo.lock` aligned; do not substitute the older crates.io release.
The tokenizer checks the dependency's Unicode data version. Native deployments
ship the compiled code and require no local rust-unic checkout.
