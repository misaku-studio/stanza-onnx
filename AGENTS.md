# stanza-onnx

Independent Stanza deployment toolkit. Rust implements inference; Python handles
conversion and reference verification. The optional JavaScript adapter uses standard
Node.js process APIs. Keep native inference free of Python,
JavaScript and Fern dependencies, and the adapter free of native npm bindings.

- Run `cargo test --workspace --locked`, `cargo fmt --all -- --check` and
  `cargo clippy --workspace --all-targets --locked -- -D warnings` for Rust changes.
- Install JavaScript development dependencies with `npm ci` using `package-lock.json`.
  Build with `npm run build`, and run `npm test` and `npm run typecheck`.
- Run `python -m unittest discover -s exporter/tests -v` for exporter changes.
- Format Python with `python -m black exporter/src exporter/tests scripts` and
  JavaScript/TypeScript with `npm run format:js`. Check formatting with Black
  `--check` and `npm run format:js:check`; Rust uses `cargo fmt`.
- Keep one operation per line, give intermediate tensors descriptive names,
  and separate orchestration from model math. Document tensor layouts and subtle
  decoding invariants. Generated data tables may stay compact.
- Processor/converter changes require raw-text parity against pinned Stanza.
  Check English, the architecture matrix in `docs/development.md`, then the
  full catalog when shared behavior changes.
- Preserve source offsets and distinguish Unicode code points, UTF-16 units,
  UTF-8 bytes and grapheme clusters.
- Preserve model preprocessing and avoid implicit Unicode normalization.
- Keep model binaries, caches and third-party corpus downloads outside Git.
- Preserve resource provenance and applicable notices with bundles.
- Keep package versions and bundle compatibility rules consistent.
- Describe per-language capabilities explicitly; catalog presence alone is not
  a runnable pipeline, and identity lemmas are not trained lemmatization.
