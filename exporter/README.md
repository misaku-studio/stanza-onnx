# stanza-onnx exporter

Python build tooling for exporting selected Stanza 1.11.0 models and packaging
language bundles for the stanza-onnx Rust runtime. The optional JavaScript
adapter uses standard Node.js APIs to call the native runtime. Python is needed only for conversion and
reference verification.

Use Python 3.12 or newer and install this directory with `python -m pip install .`.
Unicode preprocessing uses packaged, checksum-pinned official Unicode 18.0.0
data rather than the host interpreter's Unicode database. The ML integration
environment uses Python 3.12.

```sh
stanza-onnx list
stanza-onnx download --lang en --cache-dir .cache/models
stanza-onnx export --lang en --cache-dir .cache/models --output .cache/bundles/en
```

The cache uses Stanza's standard layout: `resources.json` and language
subdirectories. `export` verifies selected checkpoint checksums and writes a
complete bundle to an absent or empty output directory. The bundle includes
processor capabilities, SHA256 checksums, source URLs and source MD5 checksums.

`bundle` packages already exported assets; `reference` creates normalized
output from Stanza 1.11.0 with an explicit Unicode18 preprocessing adaptation,
using input JSON containing `texts: [{id, text}]`. The Stanza model processors
and weights are retained; Unicode-dependent string operations and regex
classes use the same tables as the native runtime. Reference metadata records
`unicodeVersion`, `referenceMode`, UCD source hashes and the host Unicode version.
Use each
command's `--help` for required arguments. Help and catalog listing do not
import training dependencies.

Runtime 0.2/0.3 bundles require `unicodeVersion: "18.0.0"`; runtime 0.1/Unicode15
bundles are rejected. Repackage existing converted graphs with `bundle` into
a fresh destination and regenerate references. Retraining and weight
re-export are unnecessary for this Unicode contract change.

The catalog contains 91 selected Stanza tokenizer pipelines, one downstream
Bengali rule pack and three partial-resource entries. Catalog presence does not imply a runnable pipeline. A selected
profile covers only tokenization, multiword expansion, POS/morphology and
lemmatization where available. The project and bundle contract are experimental.

Code is Apache-2.0; upstream attribution is retained in NOTICE. Pretrained
models and embeddings are downloaded separately and retain their own terms.
`NOASSERTION` in a bundle means that the resource's license has not been
reviewed for redistribution. This package includes catalog metadata, but no
pretrained model binaries or third-party corpus fixtures.

The Bengali pack exports with no ML dependencies or model cache. From the
checkout, set `PYTHONPATH` to `exporter/src`, then run
`python -S -m stanza_onnx export --lang bn --output .cache/bundles/bn`.
It provides tokenization, sentence splitting and source offsets only.
`sources` is empty because no upstream checkpoint is used. Its rule implementation
is Apache-2.0 project code. `reference --lang bn` rejects an upstream parity
comparison explicitly. See [Bengali documentation](../docs/bengali.md).
