# Adapted from Stanford Stanza 1.11.0 (Apache-2.0).
# https://github.com/stanfordnlp/stanza/tree/v1.11.0/stanza/models
# See the repository NOTICE for upstream attribution.
"""Export language-generic Stanza tokenizers without contextual encoders."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stanza.models.tokenization.data import SortedDataset, TokenizationDataset
from stanza.models.tokenization.trainer import Trainer
from stanza.models.tokenization.utils import MASK_RE
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stanza_onnx.catalog import language_profile


class Wrapped(nn.Module):
    """Expose boundary log probabilities with dynamic batch and time axes."""

    def __init__(self, model):
        super().__init__()
        # Keep the attribute name: it determines exported initializer scopes.
        self.model = model

    def forward(self, units, features, lengths):
        model = self.model
        # units: [batch, time]; features: [batch, time, feature_dim].
        embeddings = torch.cat([model.embeddings(units), features], 2)
        hidden, _ = model.rnn(
            pack_padded_sequence(embeddings, lengths, batch_first=True)
        )
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)

        if model.args["conv_res"] is not None:
            convolution_input = embeddings.transpose(1, 2).contiguous()
            if not model.args.get("hier_conv_res", False):
                for layer in model.conv_res:
                    hidden = (
                        hidden + layer(convolution_input).transpose(1, 2).contiguous()
                    )
            else:
                convolution_hidden = torch.cat(
                    [layer(convolution_input) for layer in model.conv_res], 1
                )
                hidden = (
                    hidden
                    + model.conv_res2(F.relu(convolution_hidden))
                    .transpose(1, 2)
                    .contiguous()
                )

        token_logits, sentence_logits = model.tok_clf(hidden), model.sent_clf(hidden)
        mwt_logits = model.mwt_clf(hidden) if model.args["use_mwt"] else None

        if model.args["hierarchical"]:
            hierarchy_input = hidden
            if model.args["hier_invtemp"] > 0:
                hierarchy_input = hierarchy_input * (
                    1 - torch.sigmoid(-token_logits * model.args["hier_invtemp"])
                )
            hierarchy_hidden, _ = model.rnn2(
                pack_padded_sequence(hierarchy_input, lengths, batch_first=True)
            )
            hierarchy_hidden, _ = pad_packed_sequence(
                hierarchy_hidden, batch_first=True
            )
            token_logits, sentence_logits = (
                token_logits + model.tok_clf2(hierarchy_hidden),
                sentence_logits + model.sent_clf2(hierarchy_hidden),
            )
            if mwt_logits is not None:
                mwt_logits = mwt_logits + model.mwt_clf2(hierarchy_hidden)

        non_token, token = -F.softplus(token_logits), -F.softplus(-token_logits)
        non_sentence, sentence = -F.softplus(sentence_logits), -F.softplus(
            -sentence_logits
        )
        if mwt_logits is None:
            return torch.cat([non_token, token + non_sentence, token + sentence], 2)

        non_mwt, mwt = -F.softplus(mwt_logits), -F.softplus(-mwt_logits)
        # Preserve Stanza's class order: continuation, token, sentence, MWT,
        # and sentence-final MWT. The runtime decodes these integer classes.
        return torch.cat(
            [
                non_token,
                token + non_sentence + non_mwt,
                token + sentence + non_mwt,
                token + non_sentence + mwt,
                token + sentence + mwt,
            ],
            2,
        )


def tokenizer_batch(text, trainer):
    """Use Stanza's own preprocessing and descending-length paragraph order."""
    dataset = TokenizationDataset(
        trainer.args,
        input_text=text,
        vocab=trainer.vocab,
        evaluation=True,
        dictionary=trainer.dictionary,
    )
    sorted_data = SortedDataset(dataset)
    batch = sorted_data.collate(
        [sorted_data[index] for index in range(len(sorted_data))]
    )
    lengths = torch.tensor([len(raw) for raw in batch[3]], dtype=torch.int64)
    return batch[0], batch[2], lengths


def export_graph(model, sample, path):
    """Export the tokenizer's stable public tensor interface."""
    torch.onnx.export(
        model,
        sample,
        str(path),
        dynamo=False,
        input_names=["units", "features", "lengths"],
        output_names=["log_probs"],
        opset_version=17,
        dynamic_axes={
            "units": {0: "batch", 1: "time"},
            "features": {0: "batch", 1: "time"},
            "lengths": {0: "batch"},
            "log_probs": {0: "batch", 1: "time"},
        },
    )
    onnx.checker.check_model(str(path))


def validate_graph(trainer, path, texts):
    """Require identical boundary classes against the original Stanza model."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    errors = []
    for text in texts:
        inputs = tokenizer_batch(text, trainer)
        with torch.inference_mode():
            expected = trainer.model(*inputs).numpy()
        feed = dict(
            zip(["units", "features", "lengths"], [tensor.numpy() for tensor in inputs])
        )
        actual = session.run(None, feed)[0]
        equal = np.array_equal(expected.argmax(-1), actual.argmax(-1))
        error = float(np.max(np.abs(expected - actual)))
        errors.append(
            {
                "codepoints": len(text),
                "batch": int(inputs[0].shape[0]),
                "max_abs_error": error,
                "argmax_equal": bool(equal),
            }
        )
        if not equal:
            raise ValueError(f"Tokenizer boundary classes differ: {errors[-1]}")
    return errors


def tokenizer_metadata(trainer, language, package):
    """Include preprocessing state needed to reproduce the loaded checkpoint."""
    vocab_state = trainer.vocab.state_dict()
    # This is runtime state, not a policy inferred from the language code.
    # Stanza 1.11 load_state_dict initializes this flag before restoring lang.
    vocab_state["lang_replaces_spaces"] = bool(trainer.vocab.lang_replaces_spaces)
    return {
        "stanza": "1.11.0",
        "language": language,
        "package": package,
        "config": trainer.args,
        "vocab": vocab_state,
        "mask_regex": MASK_RE.pattern,
        "lexicon": sorted(trainer.lexicon) if trainer.lexicon else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    package = language_profile(args.lang)["processors"]["tokenize"]

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_path = args.cache_dir / args.lang / "tokenize" / (package + ".pt")
    trainer = Trainer(model_file=str(model_path), device="cpu")
    trainer.model.eval()
    if trainer.model.charmodel is not None:
        raise ValueError("Contextual tokenizer requires a separate charLM export")

    texts = ["123.", "Test!\n\nA longer second paragraph.", "😀 123.\n\n456!"]
    model = Wrapped(trainer.model).eval()
    sample = tokenizer_batch(texts[0], trainer)
    graph_path = output_dir / "tokenizer.onnx"
    export_graph(model, sample, graph_path)
    errors = validate_graph(trainer, graph_path, texts)

    metadata = tokenizer_metadata(trainer, args.lang, package)
    (output_dir / "tokenizer.json").write_text(
        json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "tokenizer-validation.json").write_text(
        json.dumps(errors, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "language": args.lang,
                "tokenizer_validation": errors,
                "onnx_bytes": graph_path.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
