# Adapted from Stanford Stanza 1.11.0 (Apache-2.0).
# https://github.com/stanfordnlp/stanza/tree/v1.11.0/stanza/models
# See the repository NOTICE for upstream attribution.
"""Export selected Stanza POS checkpoints with dynamic batch and token axes.

Character and POS modules retain checkpoint weights. Recurrent dropout cells are
replaced by weight-identical native LSTMs for evaluation only. Large pretrained
vectors are stored as one external float32 lookup table shared by stages.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stanza.models.common.doc import Document
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.vocab import CompositeVocab
from stanza.models.pos.data import Dataset
from stanza.models.pos.trainer import Trainer, unpack_batch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stanza_onnx.catalog import language_profile, resources


def native_lstm(module):
    """Copy evaluation weights from Stanza's recurrent dropout implementation."""
    if isinstance(module, nn.LSTM):
        return module
    if not hasattr(module, "cells"):
        raise TypeError(f"Unrecognized recurrent module: {type(module)}")
    first_cell = module.cells[0]
    result = nn.LSTM(
        first_cell.input_size,
        module.hidden_size,
        module.num_layers,
        bias=first_cell.bias,
        batch_first=module.batch_first,
        bidirectional=module.num_directions == 2,
    )
    with torch.no_grad():
        for layer in range(module.num_layers):
            for direction in range(module.num_directions):
                cell = module.cells[layer * module.num_directions + direction]
                suffix = f"_l{layer}" + ("_reverse" if direction else "")
                for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
                    if getattr(cell, name) is not None:
                        getattr(result, name + suffix).copy_(getattr(cell, name))
    return result.eval()


class Characters(nn.Module):
    """Encode each word independently into one projected character feature."""

    def __init__(self, model):
        super().__init__()
        # These attribute names also define exported initializer scopes.
        self.c = model.charmodel
        self.rnn = native_lstm(self.c.charlstm.lstm)
        self.projection = model.trans_char

    def forward(self, chars, lengths):
        character_model = self.c
        packed = pack_padded_sequence(
            character_model.char_emb(chars), lengths, batch_first=True
        )
        word_count = chars.size(0)
        initial_state = (
            character_model.charlstm_h_init.expand(-1, word_count, -1).contiguous(),
            character_model.charlstm_c_init.expand(-1, word_count, -1).contiguous(),
        )
        output, hidden = self.rnn(packed, initial_state)
        output, _ = pad_packed_sequence(output, batch_first=True)
        if character_model.attn:
            valid_characters = torch.arange(
                output.size(1), device=chars.device
            ).unsqueeze(0) < lengths.unsqueeze(1)
            representations = (
                output
                * torch.sigmoid(character_model.char_attn(output))
                * valid_characters.unsqueeze(2)
            ).sum(1)
        else:
            representations = (
                hidden[0][-2:].transpose(0, 1).contiguous().reshape(word_count, -1)
            )
        return self.projection(representations)


class ContextCharacters(nn.Module):
    """Expose character-LM features at every position in a sentence stream."""

    def __init__(self, model, projection=None):
        super().__init__()
        self.c = model
        self.rnn = native_lstm(model.charlstm.lstm)
        self.projection = projection

    def forward(self, chars, lengths):
        packed = pack_padded_sequence(self.c.char_emb(chars), lengths, batch_first=True)
        batch_size = chars.size(0)
        initial_state = (
            self.c.charlstm_h_init.expand(-1, batch_size, -1).contiguous(),
            self.c.charlstm_c_init.expand(-1, batch_size, -1).contiguous(),
        )
        output, _ = self.rnn(packed, initial_state)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return self.projection(output) if self.projection is not None else output


def context_inputs(text, model, backwards):
    """Build sorted character streams and each word's trailing-space position."""
    vocab = model.char_vocab()
    rows = []
    for sentence_index, words in enumerate(text):
        if backwards:
            words = [word[::-1] for word in reversed(words)]
        characters, word_offsets = ["\n"], []
        for word in words:
            characters.extend(word)
            characters.append(" ")
            word_offsets.append(len(characters) - 1)
        if backwards:
            word_offsets.reverse()
        rows.append((vocab.map(characters), word_offsets, sentence_index))
    rows.sort(key=lambda row: len(row[0]), reverse=True)
    lengths = torch.tensor([len(row[0]) for row in rows], dtype=torch.long)
    chars = torch.full(
        (len(rows), int(lengths.max())), vocab.unit2id(" "), dtype=torch.long
    )
    for row_index, row in enumerate(rows):
        chars[row_index, : len(row[0])] = torch.tensor(row[0], dtype=torch.long)
    return (chars, lengths), rows


def context_gather(output, rows, text):
    """Restore sentence order and select the character-LM feature for each word."""
    padded = output.new_zeros(len(text), max(map(len, text)), output.shape[-1])
    for row_index, (_, word_offsets, sentence_index) in enumerate(rows):
        padded[sentence_index, : len(word_offsets)] = output[row_index, word_offsets]
    return padded


def classify(clf, hid, upos_emb, share_hid):
    """Apply a shared linear head or Stanza's UPOS-conditioned bilinear head."""
    if share_hid:
        return clf(hid)
    # Appended ones incorporate the bilinear head's affine terms. Shapes are
    # [tokens, hidden+1], [classes, hidden+1, upos+1], [tokens, upos+1].
    hidden_with_bias = torch.cat([hid, torch.ones_like(hid[:, :1])], -1)
    upos_with_bias = torch.cat([upos_emb, torch.ones_like(upos_emb[:, :1])], -1)
    return (
        torch.einsum(
            "ni,oij,nj->no", hidden_with_bias, clf.W_bilin.weight, upos_with_bias
        )
        + clf.W_bilin.bias
    )


class Sentences(nn.Module):
    """Run the highway BiLSTM and concatenate all POS classification heads."""

    def __init__(self, model):
        super().__init__()
        self.args = model.args
        self.share_hid = model.share_hid
        if model.args["word_emb_dim"] > 0:
            self.word_emb = model.word_emb
        if model.args["pretrain"]:
            self.pretrain_projection = model.trans_pretrained
        self.h = model.taggerlstm
        self.rnns = nn.ModuleList([native_lstm(layer.lstm) for layer in self.h.lstm])
        self.h_init = model.taggerlstm_h_init
        self.c_init = model.taggerlstm_c_init
        self.upos_hid, self.upos_clf = model.upos_hid, model.upos_clf
        if not self.share_hid:
            self.xpos_hid, self.ufeats_hid = model.xpos_hid, model.ufeats_hid
            self.upos_emb = model.upos_emb
        self.xpos_clfs = (
            model.xpos_clf
            if isinstance(model.xpos_clf, nn.ModuleList)
            else nn.ModuleList([model.xpos_clf])
        )
        self.feats_clfs = model.ufeats_clf

    def forward(self, words, pretrained_vectors, character_features, lengths):
        inputs = []
        if self.args["word_emb_dim"] > 0:
            inputs.append(self.word_emb(words))
        if self.args["pretrain"]:
            inputs.append(self.pretrain_projection(pretrained_vectors))
        if self.args["char"] and self.args["char_emb_dim"] > 0:
            inputs.append(character_features)
        hidden = torch.cat(inputs, -1)
        batch_size, token_count = words.shape
        for layer in range(self.h.num_layers):
            packed = pack_padded_sequence(hidden, lengths, batch_first=True)
            initial_state = (
                self.h_init[2 * layer : 2 * layer + 2]
                .expand(-1, batch_size, -1)
                .contiguous(),
                self.c_init[2 * layer : 2 * layer + 2]
                .expand(-1, batch_size, -1)
                .contiguous(),
            )
            output, _ = self.rnns[layer](packed, initial_state)
            output, _ = pad_packed_sequence(output, batch_first=True)
            projected = self.h.highway[layer](hidden)
            if self.h.highway_func is not None:
                projected = self.h.highway_func(projected)
            hidden = output + torch.sigmoid(self.h.gate[layer](hidden)) * projected
        hidden = hidden.reshape(batch_size * token_count, -1)
        upos_hidden = torch.relu(self.upos_hid(hidden))
        upos_logits = self.upos_clf(upos_hidden)
        if self.share_hid:
            xpos_hidden, features_hidden, upos_embeddings = (
                upos_hidden,
                upos_hidden,
                None,
            )
        else:
            xpos_hidden = torch.relu(self.xpos_hid(hidden))
            features_hidden = torch.relu(self.ufeats_hid(hidden))
            upos_embeddings = self.upos_emb(upos_logits.argmax(-1))
        xpos_logits = torch.cat(
            [
                classify(classifier, xpos_hidden, upos_embeddings, self.share_hid)
                for classifier in self.xpos_clfs
            ],
            -1,
        )
        features_logits = (
            torch.cat(
                [
                    classify(
                        classifier, features_hidden, upos_embeddings, self.share_hid
                    )
                    for classifier in self.feats_clfs
                ],
                -1,
            )
            if len(self.feats_clfs)
            else hidden[:, :0]
        )
        return (
            upos_logits.reshape(batch_size, token_count, -1),
            xpos_logits.reshape(batch_size, token_count, -1),
            features_logits.reshape(batch_size, token_count, -1),
        )


def native_batch(sentences, trainer, pretrain):
    """Delegate vocabulary lookup and batch sorting to Stanza's POS dataset."""
    doc = Document(
        [
            [{"id": (index + 1,), "text": word} for index, word in enumerate(sentence)]
            for sentence in sentences
        ]
    )
    dataset = Dataset(doc, trainer.args, pretrain, trainer.vocab, evaluation=True)
    return next(iter(dataset.to_loader(batch_size=len(sentences))))


def tensors(batch, trainer, characters, pretrain, contexts=()):
    """Prepare separate character and tagger inputs with Stanza's sorted order."""
    (
        inputs,
        original_indices,
        word_original_indices,
        sentence_lengths,
        word_lengths,
        text,
    ) = unpack_batch(batch, "cpu")
    words, _, chars, _, _, _, _, pretrained_ids = inputs
    batch_size, token_count = words.shape
    sorted_features = (
        characters(chars, torch.tensor(word_lengths, dtype=torch.long))
        if characters
        else torch.zeros(len(word_lengths), 0)
    )
    # Character rows are sorted by word length; tagger rows by sentence length.
    word_features = sorted_features[np.argsort(word_original_indices).tolist()]
    character_features = torch.zeros(batch_size, token_count, word_features.shape[-1])
    start = 0
    for sentence_index, length in enumerate(sentence_lengths):
        character_features[sentence_index, :length] = word_features[
            start : start + length
        ]
        start += length
    if contexts:
        parts = []
        for context in contexts:
            context_args, rows = context_inputs(
                text, context["model"], context["backwards"]
            )
            parts.append(context_gather(context["wrapper"](*context_args), rows, text))
        character_features = torch.cat(parts, -1)
    pretrained_vectors = (
        pretrain.emb[pretrained_ids]
        if pretrain
        else torch.zeros(batch_size, token_count, 0)
    )
    return (
        (chars, torch.tensor(word_lengths, dtype=torch.long)),
        (
            words,
            pretrained_vectors,
            character_features,
            torch.tensor(sentence_lengths, dtype=torch.long),
        ),
    )


def vocab_metadata(vocab):
    """Describe each head's vocabulary and composite subhead widths."""
    state = dict(vocab.state_dict())
    state["composite"] = isinstance(vocab, CompositeVocab)
    if state["composite"]:
        state["keys"] = list(vocab._id2unit)
        state["sizes"] = vocab.lens()
    else:
        state["keys"], state["sizes"] = [], [len(vocab)]
    return state


def load_fixtures():
    # Newly authored synthetic stage inputs avoid distributing training corpora.
    return [
        [["Test", "123", "."], ["A"]],
        [["Café", "ãntennae", "🦊", "."]],
        [["one"], ["two", "three"], ["four", "five", "six", "."]],
    ]


def find_model(root, lang, processor):
    package = language_profile(lang)["processors"][processor]
    return root / lang / processor / (package + ".pt")


def find_dependency(root, lang, pos_path, kind):
    dependencies = resources()[lang]["pos"][pos_path.stem].get("dependencies", [])
    packages = [
        dependency["package"]
        for dependency in dependencies
        if dependency["model"] == kind
    ]
    if len(packages) != 1:
        raise ValueError(f"Expected one {kind} dependency for {lang}/{pos_path.stem}")
    return root / lang / kind / (packages[0] + ".pt")


def load_trainer(root, language, model_path):
    """Load the selected checkpoint and its pretrained/character-LM dependencies."""
    checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
    config = checkpoint["config"]
    if config.get("bert_model"):
        raise ValueError(
            f"Contextual POS requires a separate exporter: {language} {model_path.name}"
        )
    pretrain = (
        Pretrain(filename=str(find_dependency(root, language, model_path, "pretrain")))
        if config["pretrain"]
        else None
    )
    trainer_args = None
    if config.get("charlm"):
        trainer_args = {
            "charlm_forward_file": str(
                find_dependency(root, language, model_path, "forward_charlm")
            ),
            "charlm_backward_file": str(
                find_dependency(root, language, model_path, "backward_charlm")
            ),
        }
    trainer = Trainer(
        model_file=str(model_path), args=trainer_args, pretrain=pretrain, device="cpu"
    )
    trainer.model.eval()
    return trainer, pretrain, config


def character_contexts(trainer, config):
    """Select contextual character wrappers in forward/backward concatenation order."""
    has_characters = bool(config["char"] and config["char_emb_dim"] > 0)
    contexts = []
    if has_characters and config.get("charlm"):
        for direction in ["forward", "backward"]:
            model = getattr(trainer.model, "charmodel_" + direction)
            projection = getattr(trainer.model, "charmodel_" + direction + "_transform")
            wrapper = ContextCharacters(model, projection).eval()
            contexts.append(
                {
                    "model": model,
                    "wrapper": wrapper,
                    "backwards": direction == "backward",
                    "filename": f"pos-charlm-{direction}.onnx",
                    "direction": direction,
                }
            )
    return has_characters, contexts


def context_feature_dim(context):
    projection = context["wrapper"].projection
    return projection.out_features if projection else context["model"].hidden_dim()


def pos_metadata(language, model_path, trainer, pretrain, config, characters, contexts):
    """Describe lookup tables, feature widths, and each tagger output vocabulary."""
    vocab = {key: vocab_metadata(value) for key, value in trainer.vocab._vocabs.items()}
    has_characters = bool(config["char"] and config["char_emb_dim"] > 0)
    if contexts:
        character_feature_dim = sum(
            context_feature_dim(context) for context in contexts
        )
    elif has_characters:
        character_feature_dim = config["transformed_dim"]
    else:
        character_feature_dim = 0
    return {
        "language": language,
        "model": model_path.stem,
        "vocab": vocab,
        "has_char": bool(characters),
        "has_pretrain": bool(pretrain),
        "pretrained_vocab": dict(pretrain.vocab.state_dict()) if pretrain else None,
        "pretrained_shape": list(pretrain.emb.shape) if pretrain else [0, 0],
        "character_feature_dim": character_feature_dim,
        "contexts": [
            {
                "filename": context["filename"],
                "backwards": context["backwards"],
                "vocab": dict(context["model"].char_vocab().state_dict()),
                "feature_dim": context_feature_dim(context),
            }
            for context in contexts
        ],
        "share_hid": trainer.model.share_hid,
        "rec_dropout": config["rec_dropout"],
        "char_rec_dropout": config["char_rec_dropout"],
    }


def write_metadata(artifacts, metadata, pretrain):
    (artifacts / "pos-vocab.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    if pretrain:
        # The native runtime reads this contiguous little-endian float32 table.
        pretrain.emb.numpy().astype("<f4").tofile(artifacts / "pretrained.f32")


def export_graphs(artifacts, trainer, pretrain, characters, contexts, tagger, fixtures):
    """Export independent word/stream character graphs and the sentence tagger."""
    batch = native_batch(fixtures[0], trainer, pretrain)
    with torch.inference_mode():
        character_args, sentence_args = tensors(
            batch, trainer, characters, pretrain, contexts
        )
        _, _, _, _, _, text = unpack_batch(batch, "cpu")
        if characters:
            torch.onnx.export(
                characters,
                character_args,
                str(artifacts / "pos-char.onnx"),
                input_names=["chars", "lengths"],
                output_names=["features"],
                dynamic_axes={
                    "chars": {0: "words", 1: "chars"},
                    "lengths": {0: "words"},
                    "features": {0: "words"},
                },
                opset_version=17,
                dynamo=False,
            )
        for context in contexts:
            context_args, _ = context_inputs(
                text, context["model"], context["backwards"]
            )
            torch.onnx.export(
                context["wrapper"],
                context_args,
                str(artifacts / context["filename"]),
                input_names=["chars", "lengths"],
                output_names=["features"],
                dynamic_axes={
                    "chars": {0: "batch", 1: "chars"},
                    "lengths": {0: "batch"},
                    "features": {0: "batch", 1: "chars"},
                },
                opset_version=17,
                dynamo=False,
            )
        input_names = ["words", "pretrained_vectors", "character_features", "lengths"]
        output_names = ["upos_logits", "xpos_logits", "feats_logits"]
        torch.onnx.export(
            tagger,
            sentence_args,
            str(artifacts / "pos-tagger.onnx"),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                name: ({0: "batch"} if name == "lengths" else {0: "batch", 1: "tokens"})
                for name in input_names + output_names
            },
            opset_version=17,
            dynamo=False,
        )
    graph_names = (
        (["pos-char.onnx"] if characters else [])
        + [context["filename"] for context in contexts]
        + ["pos-tagger.onnx"]
    )
    for name in graph_names:
        onnx.checker.check_model(str(artifacts / name))
    return input_names


def graph_sessions(artifacts, characters, contexts):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    character_session = (
        ort.InferenceSession(
            str(artifacts / "pos-char.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        if characters
        else None
    )
    for context in contexts:
        context["session"] = ort.InferenceSession(
            str(artifacts / context["filename"]),
            options,
            providers=["CPUExecutionProvider"],
        )
    tagger_session = ort.InferenceSession(
        str(artifacts / "pos-tagger.onnx"), options, providers=["CPUExecutionProvider"]
    )
    return character_session, tagger_session


def validate_character_features(
    feed,
    character_args,
    sentence_args,
    word_original_indices,
    sentence_lengths,
    text,
    characters,
    character_session,
    contexts,
):
    """Replace PyTorch character inputs with ONNX outputs before tagger validation."""
    max_error = 0.0
    if character_session:
        actual = character_session.run(
            None,
            dict(
                zip(["chars", "lengths"], [tensor.numpy() for tensor in character_args])
            ),
        )[0]
        max_error = float(np.max(np.abs(actual - characters(*character_args).numpy())))
        word_features = actual[np.argsort(word_original_indices)]
        padded = np.zeros_like(sentence_args[2].numpy())
        start = 0
        for sentence_index, length in enumerate(sentence_lengths):
            padded[sentence_index, :length] = word_features[start : start + length]
            start += length
        feed["character_features"] = padded
    if contexts:
        pieces = []
        for context in contexts:
            context_args, rows = context_inputs(
                text, context["model"], context["backwards"]
            )
            actual = context["session"].run(
                None,
                dict(
                    zip(
                        ["chars", "lengths"],
                        [tensor.numpy() for tensor in context_args],
                    )
                ),
            )[0]
            expected = context["wrapper"](*context_args).numpy()
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            pieces.append(context_gather(torch.from_numpy(actual), rows, text).numpy())
        feed["character_features"] = np.concatenate(pieces, -1)
    return max_error


def validate_tag_outputs(
    expected, actual, native, trainer, vocab, original_indices, lengths
):
    """Compare decoded Stanza labels and each independent ONNX subhead argmax."""
    native_original = [None] * len(native)
    wrapper_parity, onnx_parity = True, True
    max_error = 0.0
    widths = [vocab["upos"]["sizes"], vocab["xpos"]["sizes"], vocab["feats"]["sizes"]]
    expected_heads = [
        torch.split(output, sizes, dim=-1) for output, sizes in zip(expected, widths)
    ]
    actual_heads = [
        np.split(output, np.cumsum(sizes)[:-1], axis=-1)
        for output, sizes in zip(actual, widths)
    ]
    for sentence_index, length in enumerate(lengths):
        native_original[original_indices[sentence_index]] = native[sentence_index]
        for expected_output, actual_output in zip(expected, actual):
            if expected_output.shape[-1]:
                max_error = max(
                    max_error,
                    float(
                        np.max(
                            np.abs(
                                expected_output[sentence_index, :length].numpy()
                                - actual_output[sentence_index, :length]
                            )
                        )
                    ),
                )
        for word_index, native_tags in enumerate(native[sentence_index]):
            decoded = []
            for head_index, vocab_name in enumerate(["upos", "xpos", "feats"]):
                ids = [
                    int(head[sentence_index, word_index].argmax(-1))
                    for head in expected_heads[head_index]
                ]
                decoded.append(
                    trainer.vocab[vocab_name].id2unit(
                        ids if vocab[vocab_name]["composite"] else ids[0]
                    )
                )
            wrapper_parity &= decoded == native_tags
        for expected_subheads, actual_subheads in zip(expected_heads, actual_heads):
            for expected_head, actual_head in zip(expected_subheads, actual_subheads):
                onnx_parity &= np.array_equal(
                    expected_head[sentence_index, :length].argmax(-1).numpy(),
                    actual_head[sentence_index, :length].argmax(-1),
                )
    return max_error, wrapper_parity, onnx_parity, native_original


def validate_graphs(
    artifacts,
    trainer,
    pretrain,
    characters,
    contexts,
    tagger,
    vocab,
    fixtures,
    input_names,
):
    character_session, tagger_session = graph_sessions(artifacts, characters, contexts)
    report = []
    with torch.inference_mode():
        for sentences in fixtures:
            batch = native_batch(sentences, trainer, pretrain)
            character_args, sentence_args = tensors(
                batch, trainer, characters, pretrain, contexts
            )
            expected = tagger(*sentence_args)
            _, original_indices, word_original_indices, sentence_lengths, _, text = (
                unpack_batch(batch, "cpu")
            )
            feed = {
                name: tensor.numpy()
                for name, tensor in zip(input_names, sentence_args)
                if name in [tensor.name for tensor in tagger_session.get_inputs()]
            }
            character_error = validate_character_features(
                feed,
                character_args,
                sentence_args,
                word_original_indices,
                sentence_lengths,
                text,
                characters,
                character_session,
                contexts,
            )
            actual = tagger_session.run(None, feed)
            native = trainer.predict(batch, unsort=False)
            max_error, wrapper_parity, onnx_parity, native_original = (
                validate_tag_outputs(
                    expected,
                    actual,
                    native,
                    trainer,
                    vocab,
                    original_indices,
                    sentence_lengths,
                )
            )
            entry = {
                "sentences": sentences,
                "char_max_abs_error": character_error,
                "tagger_max_abs_error": max_error,
                "wrapper_native_label_parity": bool(wrapper_parity),
                "onnx_label_parity": bool(onnx_parity),
                "native_tags": native_original,
            }
            report.append(entry)
            if not wrapper_parity or not onnx_parity:
                raise ValueError(f"POS parity check failed: {entry}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.out
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    model_path = find_model(args.cache_dir, args.lang, "pos")
    trainer, pretrain, config = load_trainer(args.cache_dir, args.lang, model_path)
    has_characters, contexts = character_contexts(trainer, config)
    characters = (
        Characters(trainer.model).eval() if has_characters and not contexts else None
    )
    tagger = Sentences(trainer.model).eval()
    metadata = pos_metadata(
        args.lang, model_path, trainer, pretrain, config, characters, contexts
    )
    write_metadata(artifacts, metadata, pretrain)

    fixtures = load_fixtures()
    input_names = export_graphs(
        artifacts, trainer, pretrain, characters, contexts, tagger, fixtures
    )
    report = validate_graphs(
        artifacts,
        trainer,
        pretrain,
        characters,
        contexts,
        tagger,
        metadata["vocab"],
        fixtures,
        input_names,
    )
    (artifacts / "pos-export-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "language": args.lang,
                "model": model_path.stem,
                "char_error": max(entry["char_max_abs_error"] for entry in report),
                "tagger_error": max(entry["tagger_max_abs_error"] for entry in report),
                "fixtures": len(report),
                "parity": True,
                "share_hid": metadata["share_hid"],
                "xpos_composite": metadata["vocab"]["xpos"]["composite"],
                "rec_dropout": config["rec_dropout"],
                "char_rec_dropout": config["char_rec_dropout"],
            }
        )
    )


if __name__ == "__main__":
    main()
