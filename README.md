# stanza-onnx

**Stanza models and pipelines, without Python at runtime.**

> **Experimental project.** APIs, bundle formats and deployment paths may change. This is an early implementation and is not yet production-ready.

An independent downstream toolkit for exporting selected [Stanza](https://github.com/stanfordnlp/stanza) models to ONNX and running complete pipelines in Rust. Python handles model conversion and reference verification. The native runtime provides preprocessing, inference and decoding through ONNX Runtime, with a CLI and an embeddable Rust library. A small optional adapter connects a persistent native process to JavaScript applications such as Fern.

## Status

The initial processors are tokenization and sentence splitting, multiword expansion, POS and morphological features, and lemmatization. The pinned Stanza 1.11.0 catalog selects 91 neural pipelines. A downstream Bengali rule pack adds model-free tokenization:

| Capability | Selected pipelines |
| --- | ---: |
| Tokenization and sentence splitting | 92 (91 Stanza + Bengali rules) |
| POS and morphology | 90 |
| Trained lemmatization | 85 |
| Identity lemmas (copy the written form) | 5 |

Myanmar and Bengali provide tokenization only. Bengali uses the explicit [`bengali-rule-v1` pack](docs/bengali.md): conservative token/sentence boundaries and source offsets, with no POS, morphology or lemmas. Malayalam, Odia and Sinhala still have partial upstream resources without a selected tokenizer pipeline. Capabilities and `tokenizerKind` make these distinctions explicit. Dependency parsing, NER, transformer profiles and additional fallback tokenizers remain future work.

The Rust runtime matched the saved Stanza references exactly on **728 raw-text fixtures across all 91 selected pipelines, covering 8,909 words**, using the existing converted models. The seven-language authored matrix passed separately, as did longer English documents and a persistent process's request handling, error recovery, UTF-16 offsets and shutdown. These checks measure implementation parity; they do not establish linguistic accuracy on unseen text. See [validation](docs/validation.md).

A one-thread English CPU benchmark with Unicode18 preprocessing on M5 Max measured direct native medians of 4.32, 17.65 and 55.68 ms for short, medium and long inputs. These results cover the tested English bundle and machine. See [performance measurements and limits](docs/validation.md#native-cpu-performance).

CPU inference on macOS arm64 is validated. CoreML on macOS and DirectML on Windows are optional provider selections that require compatible native ONNX Runtime libraries. Their native execution paths and acceleration performance have not been validated; enabling a provider does not guarantee that every operation is accelerated or that inference is faster. English CPU model and JavaScript adapter integration checks passed on macOS and Windows.

## Build models and the runtime

Use Python 3.12 or newer for model preparation and Rust 1.85 or newer for the native runtime. The Unicode data comes from pinned official Unicode 18.0.0 files, independently of the interpreter's Unicode version. The ML integration environment uses Python 3.12. Run these commands from the repository root:

```sh
python3.12 -m venv .venv
# On Windows: py -3.12 -m venv .venv
# Another supported Python >=3.12 can be used instead.
# Activate .venv using your shell's normal activation command.
python -m pip install -e ./exporter
python -m stanza_onnx list
python -m stanza_onnx download --lang en --cache-dir .cache/models
python -m stanza_onnx export --lang en --cache-dir .cache/models --output .cache/bundles/en

cargo build --release --locked -p stanza-onnx
python scripts/setup_ort.py --output .cache/ort
```

The setup helper downloads the pinned ONNX Runtime 1.23.2 CPU distribution for the current platform, verifies its archive checksum and retains license/notice files under `licenses/`. Its JSON output includes the native library path. Use an absent or empty destination for setup and export. Model bundles and native runtime libraries are separate from source code.

**Python is needed during preparation and verification only.** Neural pipelines need the native executable or library, a language bundle, the compatible ONNX Runtime shared library and its native dependencies. The Bengali rule pack needs only the native runtime and its metadata bundle. Retain applicable model and runtime notices when distributing those artifacts.

## Bengali without model downloads

From the checkout, prepare the Bengali metadata bundle using only Python's standard library:

```sh
PYTHONPATH=exporter/src python -S -m stanza_onnx export --lang bn --output .cache/bundles/bn
cargo build --release --locked -p stanza-onnx
./target/release/stanza-onnx --model-dir .cache/bundles/bn \
  analyze --text "আমি বাংলা শিখছি।" --offset-unit utf16
```

No ONNX Runtime shared library is loaded for this pack. In Rust, use `Pipeline::load` without calling `initialize_library`; in JavaScript, omit `ortLibrary`. See [Bengali rules and their limits](docs/bengali.md) for Windows commands, abbreviation behavior and validation.

## Run the native CLI

For macOS, using the library produced by the setup command:

```sh
./target/release/stanza-onnx \
  --model-dir .cache/bundles/en \
  --ort-library .cache/ort/libonnxruntime.1.23.2.dylib \
  analyze --text "The children were reading books."
```

On Linux, the library is `libonnxruntime.so.1.23.2`. On Windows, use `target/release/stanza-onnx.exe` with `onnxruntime.dll`. The helper supports x64 and arm64 targets for these operating systems; platform availability is separate from completed validation.

The runtime also accepts `ORT_DYLIB_PATH`, or discovers a supported ONNX Runtime library placed beside the executable. `--provider cpu` and `--threads 1` are the defaults. Use `info` to inspect the loaded pipeline's capabilities.

To keep models loaded across requests, start `serve` with the same global flags. It accepts one JSON request per line on stdin and writes one JSON response per line on stdout:

```json
{"id":1,"method":"info"}
{"id":2,"method":"analyze","params":{"text":"The children read books.","offsetUnit":"utf16"}}
{"id":3,"method":"shutdown"}
```

Responses contain the request `id` and either `result` or `error`. An invalid request returns an error and subsequent requests can continue. `shutdown` acknowledges closure and exits. Diagnostics use stderr.

## Embed in Rust

The crate under `crates/stanza-onnx/` exposes the same pipeline for native applications:

```rust
use std::path::Path;
use stanza_onnx::{initialize_library, OffsetUnit, Pipeline, SessionOptions};

fn main() -> anyhow::Result<()> {
    initialize_library(Some(Path::new(".cache/ort/libonnxruntime.1.23.2.dylib")))?;
    let mut pipeline = Pipeline::load(
        Path::new(".cache/bundles/en"),
        &SessionOptions::default(),
        true, // Verify bundle integrity.
    )?;
    let sentences = pipeline.analyze("The children read books.", OffsetUnit::CodePoint)?;
    println!("{sentences:?}");
    drop(pipeline); // Sessions are also released automatically when it leaves scope.
    Ok(())
}
```

ONNX Runtime initialization for neural pipelines is process-global; select the library before loading them. `Pipeline::load_with_library` also accepts an explicit path and initializes it only for a neural bundle. Rule-only pipelines skip this initialization. If no explicit initialization is supplied, pipeline loading uses the configured environment path or an adjacent library. Dropping a pipeline releases its model sessions while the shared runtime remains initialized for the process.

## Optional JavaScript adapter

The adapter uses standard Node.js APIs and supports Node.js 22 or newer. It has no runtime npm dependencies and delegates inference to a persistent Rust process. Deploy the native executable, ONNX Runtime library and model bundle with your application. Both package entry points, `stanza-onnx` and `stanza-onnx/native`, expose the adapter:

```js
import { createNativePipeline } from "stanza-onnx";

const pipeline = await createNativePipeline({
  executable: "./target/release/stanza-onnx",
  modelDir: ".cache/bundles/en",
  ortLibrary: ".cache/ort/libonnxruntime.1.23.2.dylib",
  offsetUnit: "utf16",
});
try {
  console.log(await pipeline.analyze("The children were reading books."));
} finally {
  await pipeline.close();
}
```

The package ships compiled ESM JavaScript and TypeScript declarations; consumers do not need a TypeScript loader or native npm bindings. Source lives in `adapters/node/native.ts` and the built entry point is `dist/native.js`. Build the checkout with `npm ci && npm run build`, using the checked-in `package-lock.json`. No npm publication or hosted model releases are configured yet.

Tokens have half-open `start`/`end` offsets into the original input and one or more expanded words. Available word annotations are `upos`, `xpos`, `feats` and `lemma`. Native default offsets count Unicode code points, matching Stanza; choose `utf16` for JavaScript slicing or `utf8` for byte positions. These units are distinct from grapheme clusters. Close the adapter to finish accepted work and release the child process; native Rust pipelines release sessions on drop. The adapter rejects unpaired UTF-16 surrogates rather than replacing source characters.

## Verification and resources

```sh
cargo test --workspace --locked
python -m unittest discover -s exporter/tests -v
npm ci
npm run build
npm test
npm run typecheck
```

Integration checks require separately prepared language bundles and an ONNX Runtime shared library. See [development](docs/development.md) for native parity commands, the architecture matrix and persistent-process checks, and [the bundle format](docs/bundles.md) for capabilities, integrity and provenance.

Project code is Apache-2.0; retained upstream attribution is in [NOTICE](NOTICE). Models, embeddings and native dependencies are separate artifacts with their own terms. `NOASSERTION` in source metadata means redistribution terms have not been audited. Exporting a model does not change its license or grant permission to publish every bundle.

The Rust runtime and Python reference use the same generated Unicode 18.0.0 property and casing tables from checksum-pinned official UCD files. Reference verification uses Stanza 1.11.0 with an explicit Unicode18 preprocessing adaptation; its neural models and weights are unchanged. This contract updates Unicode-dependent model inputs and is identified by `unicodeVersion` in bundles and reference/native output. Runtime 0.3 accepts Stanza bundles from runtime 0.2.x and 0.3.x with Unicode18, and rejects runtime 0.1/Unicode15 bundles. Bengali rules require a runtime 0.3.x bundle. Existing exported graphs can be repackaged without retraining or re-exporting weights. See [Unicode and bundle compatibility](docs/bundles.md#source-alignment-and-unicode).
