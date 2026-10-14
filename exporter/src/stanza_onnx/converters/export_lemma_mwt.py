# Adapted from Stanford Stanza 1.11.0 (Apache-2.0).
# https://github.com/stanfordnlp/stanza/tree/v1.11.0/stanza/models
# See the repository NOTICE for upstream attribution.
"""Export pinned Stanza multiword expansion and lemmatization checkpoints."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from stanza.models.lemma.trainer import Trainer as LemmaTrainer
from stanza.models.mwt.trainer import Trainer as MwtTrainer
from torch import nn

from stanza_onnx.catalog import language_profile


torch.set_num_threads(1)


class MwtClassifier(nn.Module):
    """Predict character boundaries for exact-piece MWT expansion models."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, lengths):
        # Copy-only character IDs lie outside the learned embedding vocabulary.
        embeddings = self.model.embedding(
            torch.where(src < self.model.vocab_size, src, 1)
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True
        )
        encoded, _ = self.model.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        return self.model.output_layer(encoded)


class Encoder(nn.Module):
    """Encode source characters and expose the initial decoder state."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, lengths, pos):
        embeddings = self.model.embedding(
            torch.where(src < self.model.vocab_size, src, 1)
        )
        if self.model.use_pos:
            embeddings = torch.cat(
                [self.model.pos_embedding(pos).unsqueeze(1), embeddings], dim=1
            )
            lengths = lengths + 1
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True
        )
        encoded, (hidden, cell) = self.model.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        # Stanza concatenates the final backward state before the forward state.
        hidden = torch.cat((hidden[-1], hidden[-2]), 1)
        cell = torch.cat((cell[-1], cell[-2]), 1)
        edit_logits = (
            self.model.edit_clf(hidden)
            if self.model.edit
            else hidden.new_zeros((hidden.shape[0], 3))
        )
        return encoded, hidden, cell, edit_logits


class Step(nn.Module):
    """Export one decoder step; beam search stays in the native runtime."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, previous, h, c, ctx, mask):
        embeddings = self.model.embedding(
            torch.where(previous < self.model.vocab_size, previous, 1)
        )
        next_hidden, next_cell = self.model.decoder.lstm_cell(embeddings, (h, c))
        if self.model.copy:
            attended, log_attention = self.model.decoder.attention_layer(
                next_hidden, ctx, mask=mask, return_logattn=True
            )
            if self.model.use_pos:
                # The prepended POS position cannot be copied into a word.
                log_attention = log_attention[:, 1:]
            log_attention = torch.log_softmax(log_attention, -1)
            copy_gate = self.model.copy_gate(attended)
        else:
            attended, _ = self.model.decoder.attention_layer(
                next_hidden, ctx, mask=mask
            )
            copy_gate = attended.new_zeros((attended.shape[0], 1))
            log_attention = ctx.new_zeros((ctx.shape[0], ctx.shape[1]))
        return (
            torch.log_softmax(self.model.dec2vocab(attended), -1),
            copy_gate,
            log_attention,
            next_hidden,
            next_cell,
        )


def export(module, inputs, names, outputs, axes, path):
    """Export an evaluation graph and open a single-threaded CPU session."""
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=names,
            output_names=outputs,
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def check(module, session, inputs, names):
    """Check every exported tensor against PyTorch at the existing tolerance."""
    with torch.no_grad():
        expected = module(*inputs)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    # Optional POS/mask inputs can disappear when tracing a model without them.
    exported_inputs = {tensor.name for tensor in session.get_inputs()}
    feed = {
        name: tensor.numpy()
        for name, tensor in zip(names, inputs)
        if name in exported_inputs
    }
    actual = session.run(None, feed)
    if len(expected) != len(actual):
        raise ValueError("Exported output count differs")

    errors = []
    for expected_tensor, actual_tensor in zip(expected, actual):
        reference = expected_tensor.numpy()
        if reference.shape != actual_tensor.shape or not np.allclose(
            reference, actual_tensor, rtol=1e-4, atol=1e-3
        ):
            raise ValueError("MWT/lemma numerical export parity check failed")
        errors.append(float(np.max(np.abs(reference - actual_tensor))))
    return errors


def sources(words, vocab):
    """Pad character IDs, including Stanza's start/end markers, for export."""
    lengths = [len(word) + 2 for word in words]
    source_ids = torch.zeros((len(words), max(lengths)), dtype=torch.long)
    for index, word in enumerate(words):
        character_ids = vocab.map(["<SOS>"] + list(word) + ["<EOS>"])
        source_ids[index, : lengths[index]] = torch.tensor(character_ids)
    return source_ids, torch.tensor(lengths, dtype=torch.long)


def export_seq(model, vocab, out, prefix):
    """Export the encoder and single-step decoder, checking dynamic batches."""
    encoder = Encoder(model)
    source_ids, source_lengths = sources(["testing", "cats"], vocab)
    pos_ids = torch.tensor([1, 1], dtype=torch.long)
    encoder_inputs = (source_ids, source_lengths, pos_ids)
    encoder_names = ["src", "lengths", "pos"]
    encoder_axes = {
        "src": {0: "batch", 1: "chars"},
        "lengths": {0: "batch"},
        "pos": {0: "batch"},
        "ctx": {0: "batch", 1: "ctxchars"},
        "h": {0: "batch"},
        "c": {0: "batch"},
        "edit": {0: "batch"},
    }
    encoder_session = export(
        encoder,
        encoder_inputs,
        encoder_names,
        ["ctx", "h", "c", "edit"],
        encoder_axes,
        out / f"{prefix}_encoder.onnx",
    )
    report = {"encoder": check(encoder, encoder_session, encoder_inputs, encoder_names)}
    dynamic_ids, dynamic_lengths = sources(["kitten"], vocab)
    report["encoder_dynamic"] = check(
        encoder,
        encoder_session,
        (dynamic_ids, dynamic_lengths, torch.tensor([1], dtype=torch.long)),
        encoder_names,
    )

    step = Step(model)
    with torch.no_grad():
        context, hidden, cell, edit = encoder(*encoder_inputs)
    padding_mask = source_ids.eq(0)
    if model.use_pos:
        padding_mask = torch.cat(
            [torch.zeros((source_ids.shape[0], 1), dtype=torch.bool), padding_mask], 1
        )
    step_inputs = (torch.tensor([2, 2]), hidden, cell, context, padding_mask)
    step_names = ["previous", "h", "c", "ctx", "mask"]
    step_outputs = ["logprobs", "copygate", "logattn", "next_h", "next_c"]
    step_axes = {name: {0: "batch"} for name in step_names + step_outputs}
    step_axes["ctx"][1] = "ctxchars"
    step_axes["mask"][1] = "ctxchars"
    step_axes["logattn"][1] = "chars"
    step_session = export(
        step,
        step_inputs,
        step_names,
        step_outputs,
        step_axes,
        out / f"{prefix}_step.onnx",
    )
    report["step"] = check(step, step_session, step_inputs, step_names)
    report["step_dynamic"] = check(
        step, step_session, tuple(tensor[:1] for tensor in step_inputs), step_names
    )
    return report


def json_dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def find_model(root, lang, processor):
    """Resolve the catalog choice, leaving absent and identity stages graph-free."""
    package = language_profile(lang)["processors"].get(processor)
    if not package or package == "identity":
        return None
    expected = root / lang / processor / f"{package}.pt"
    if not expected.exists():
        raise FileNotFoundError(
            f"Selected {processor} model has not been downloaded: {expected}"
        )
    return expected


def export_mwt(root, language, output_dir):
    model_path = find_model(root, language, "mwt")
    if model_path is None:
        json_dump(output_dir / "mwt.json", {"kind": "none"})
        return {"kind": "none"}

    trainer = MwtTrainer(model_file=str(model_path), device="cpu")
    if trainer.args["dict_only"]:
        kind = "dictionary"
    elif trainer.args.get("force_exact_pieces"):
        kind = "classifier"
    else:
        kind = "seq2seq"
    metadata = {
        "kind": kind,
        "args": trainer.args,
        "vocab": trainer.vocab.state_dict(),
        "dict": trainer.expansion_dict,
    }
    if kind == "seq2seq":
        # Stanza 1.11 MWT Trainer passes the configured beam as the third
        # positional (POS) argument. Effective decoding uses this default.
        metadata["decode_beam_size"] = (
            inspect.signature(trainer.model.predict).parameters["beam_size"].default
        )
    json_dump(output_dir / "mwt.json", metadata)

    if kind == "classifier":
        module = MwtClassifier(trainer.model)
        inputs = sources(["testing", "cats"], trainer.vocab)
        names = ["src", "lengths"]
        session = export(
            module,
            inputs,
            names,
            ["logits"],
            {
                "src": {0: "batch", 1: "chars"},
                "lengths": {0: "batch"},
                "logits": {0: "batch", 1: "chars"},
            },
            output_dir / "mwt.onnx",
        )
        return {
            "kind": kind,
            "static": check(module, session, inputs, names),
            "dynamic": check(
                module, session, sources(["kitten"], trainer.vocab), names
            ),
        }
    if kind == "seq2seq":
        return {
            "kind": kind,
            **export_seq(trainer.model, trainer.vocab, output_dir, "mwt"),
        }
    return {"kind": kind}


def export_lemma(root, language, output_dir):
    model_path = find_model(root, language, "lemma")
    if model_path is None:
        kind = (
            "identity"
            if language_profile(language)["processors"].get("lemma") == "identity"
            else "none"
        )
        json_dump(output_dir / "lemma.json", {"kind": kind})
        return {"kind": kind}

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if checkpoint.get("contextual"):
        raise RuntimeError(
            f"{language}: contextual lemma classifiers require an additional export; "
            "unsupported by this converter"
        )
    if checkpoint["config"].get("charlm_forward_file") or checkpoint["config"].get(
        "charlm_backward_file"
    ):
        raise RuntimeError(
            f"{language}: lemma charLM requires an additional export; "
            "select nocharlm model where available"
        )
    trainer = LemmaTrainer(model_file=str(model_path), device="cpu")
    if trainer.model is not None and trainer.model.contextual_embedding is not None:
        raise RuntimeError(f"{language}: contextual lemma embedding unsupported")
    kind = "dictionary" if trainer.args["dict_only"] else "seq2seq"
    metadata = {
        "kind": kind,
        "args": trainer.args,
        "char": trainer.vocab["char"].state_dict(),
        "pos": trainer.vocab["pos"].state_dict(),
        "word_dict": trainer.word_dict,
        "composite_dict": [
            [word, pos, lemma] for (word, pos), lemma in trainer.composite_dict.items()
        ],
    }
    json_dump(output_dir / "lemma.json", metadata)
    graph_report = (
        export_seq(trainer.model, trainer.vocab["char"], output_dir, "lemma")
        if kind == "seq2seq"
        else {}
    )
    return {"kind": kind, **graph_report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"language": args.lang}
    report["mwt"] = export_mwt(args.cache_dir, args.lang, output_dir)
    report["lemma"] = export_lemma(args.cache_dir, args.lang, output_dir)
    json_dump(output_dir / "lemma_mwt_export_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
