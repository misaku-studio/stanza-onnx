# Validation

Recorded on 2026-10-14 for runtime 0.2.0 / Unicode 18.0.0 on macOS arm64, Apple M5 Max. The model/reference environment used Python 3.12.14, Stanza/resources 1.11.0, PyTorch 2.10.0, NumPy 2.5.4 and ONNX 1.20.1. Native inference used the standalone ONNX Runtime 1.23.2 CPU library with one inference thread.

The current source passed fresh exports for all 91 pipelines. Fresh Python references were identical to the saved Unicode18 references; the refactored native runtime matched all 1,154 case/pipeline checks, including the authored matrix, Unicode edge cases and long documents. Formatting checks cover hand-written Python, JavaScript/TypeScript and Rust source.

## Native Rust runtime

| Check | Result |
| --- | --- |
| Complete native pipelines versus fresh Stanza Unicode18 references | 91/91 pipelines, 728/728 exact cases, 8,909 words |
| Authored native architecture matrix: en/fr/nl/zh-hans/th/sd/my | 56/56 exact cases, 349 words |
| Unicode16–18 classification, casing and source-offset fixtures | 338/338 exact case/pipeline pairs across 13 pipelines, 1,181 words |
| English benchmark corpus | 30/30 exact cases, 650 words |
| Mixed-length English document through Rust versus Stanza | 41 sentences / 885 words, exact parity |
| Multi-batch English document through Rust versus Stanza | 601 sentences / 6,010 words, exact parity |
| Persistent English JSON-lines protocol | Five requests passed: info, analysis, invalid-request recovery, empty input and shutdown |
| UTF-16 source offsets through the native protocol | Passed, including an astral emoji |
| Rust model-free unit tests | 38 passed; optional metadata audit also passed against the 91 bundles |
| Rust formatting and strict Clippy checks | Passed |
| Rust crate packaging and package compilation | Passed |
| Generated Unicode compatibility data | Byte-for-byte reproduction passed under Python 3.12.14 and 3.14.6 |
| Missing or invalid native-library installation | Actionable error with exit code 1; no process panic |
| JavaScript adapter model-free subprocess tests | 21/21 under Node.js 24.15.0 |
| JavaScript adapter with the actual English native runtime | Two analyses, UTF-16 spans, lemmas, argument-error recovery and graceful close passed under Node.js 24.15.0 |
| Installed compiled adapter in an isolated Node.js consumer | 30/30 exact English cases, 650 words; both package exports and TypeScript declarations passed |
| Fresh English export and Unicode18 bundle | Component export checks passed; 26/26 exact Unicode edge cases, 91 words |
| Packaged Python exporter on Python 3.14.6 | Unicode18 asset, source hashes, casing and license files verified |
| Exporter and native sweep tool tests | 26 exporter tests and seven tool tests passed; model-free Python 3.14 checks also passed |

The sweep assembled fresh runtime 0.2 / Unicode18 language bundles and computed new references using Stanza 1.11.0 with the explicit Unicode18 preprocessing adaptation. `scripts/sweep.py` ran the standalone native executable; Python coordinated verification and was absent from inference. All 91 executable language/script selections passed. The neural weights were unchanged.

Exact comparisons cover sentence/token boundaries, normalized token and expanded word text, original-source code-point offsets, UPOS, XPOS, morphological features and lemmas where selected. Eight small cases per executable language include four native-language inputs and four stability cases. Four partial-resource catalog entries are excluded because they lack selected tokenizer pipelines. Myanmar is tokenizer-only; five languages use identity lemmas.

The persistent protocol check keeps one native process alive across requests and verifies that a rejected request does not prevent later analysis. The JavaScript adapter exchanges JSON through this protocol. Model-free subprocess tests cover split UTF-8, offset units, shutdown, deadlines, child failures and rejection of unpaired surrogates; JavaScript contains no ONNX inference implementation.

The authored matrix, Unicode fixtures and longer English documents were checked separately against fresh Unicode18 Python references. The Unicode fixtures cover en/el/ru/tr/my/th/sd/fr/nl/zh-hans/zh-hant/ja/ko. These also verify checkpoint-specific CJK token-space behavior and default casing without language-specific rules. The 6,010-word document exercises bounded POS batches across many sentences. These checks add input coverage; performance was measured separately below.

## Native CPU performance

The Rust CPU path now groups whole sentences into batches of at most 128 padded word positions and ordinary word-character inputs into batches of at most 128 padded character positions. It encodes repeated exact word text once within each POS batch and scatters the features back to every occurrence. Oversized sentences and words remain whole in isolated batches. Contextual character language models retain complete sentence context. CoreML/DirectML use their separate batching policy until dispatch costs are measured.

These changes reuse the existing graphs and weights. The Unicode18 contract requires fresh runtime 0.2 bundle metadata, without weight re-export. The final CPU policy passed the full 91-pipeline sweep, authored architecture matrix, 30-case English benchmark reference, both longer English documents and persistent protocol checks listed above.

English `default_fast` performance with freshly exported assets and Unicode18 preprocessing was measured on M5 Max with the standalone ONNX Runtime 1.23.2 CPU library and one inference thread. Three fresh Rust processes were measured sequentially, with five warmups and 50 timed analyses per input per process. The median column is the median of the three process medians; p95 is the median of their individual p95s, not a pooled percentile. Timing includes analysis and release of the prior result, excluding JSON serialization and process transport.

| Input | Words / sentences | Median | p95 |
| --- | --- | ---: | ---: |
| Short: 69 code points | 11 / 1 | 4.32 ms | 4.63 ms |
| Medium: 385 code points | 61 / 4 | 17.65 ms | 18.28 ms |
| Long: 1,422 code points | 234 / 14 | 55.68 ms | 57.65 ms |

All captured outputs matched the freshly generated Stanza Unicode18 reference exactly. External peak RSS, sampled at a target 10 ms interval, was 391.9–398.3 MiB. Sampling can miss brief peaks.

Median direct-Rust startup was 452 ms, including ONNX Runtime initialization, bundle integrity verification and model loading after fixture reads. Filesystem caches were not flushed; the measurement excludes process work before the harness timer.

These timings cover the direct Rust runtime, not the JavaScript adapter.

The batching limits are CPU heuristics tested for English performance, not optimal settings for every language, input or device. No performance claim is made for Windows, CoreML or DirectML. Model parity across the catalog is separate from cross-language performance and gold linguistic accuracy.

## Fixtures and limits

The all-language checks used private raw-text corpus fixtures. These third-party fixture files and model binaries remain outside Git. Checked-in fixtures are newly authored and cover the seven representative architecture paths plus Unicode edge cases. Reproduction commands are in [development](development.md).

The shared table generator was independently checked against historical official UCD15 and UCD16 sources. Generated properties and full casing matched Python 3.12.14 and 3.14.6 respectively across all 1,112,064 Unicode scalar values, with additional FinalSigma context probes. This validates the generator against known interpreter implementations; Unicode18 behavior is derived directly from the official UCD18 sources.

These are implementation and extraction parity checks, not gold linguistic-accuracy evaluations. They do not establish equivalence for every Unicode input, arbitrarily long documents, alternative upstream profiles or additional Stanza processors. The current contract uses identical native/reference tables generated from checksum-pinned official Unicode 18.0.0 data. Its reference runs Stanza 1.11.0 with an explicit Unicode18 preprocessing adaptation and unchanged neural weights; the host interpreter's Unicode version is recorded separately. Newly authored `fixtures/unicode.json` covers selected new cased letters, decimal digits, FinalSigma contexts and source offsets without normalization. Word/grapheme utilities and fallback tokenizers have not been implemented.

The complete catalog is validated on macOS arm64 CPU. GitHub Actions also passed English model export/parity, Unicode edge cases, persistent requests and the compiled JavaScript adapter on macOS and Windows. This Windows coverage is for English; native CoreML and DirectML paths and their performance remain unvalidated. The English CPU measurements above do not establish cross-language acceleration performance.
