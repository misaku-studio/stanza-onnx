//! Shared lemma/MWT encoder-decoder inference, including copy and beam decoding.
//! Adapted from Stanford Stanza 1.11.0 (Apache-2.0); see LICENSE and NOTICE.
//! Copyright 2019 The Board of Trustees of The Leland Stanford Junior University.
//! https://github.com/stanfordnlp/stanza/tree/v1.11.0/stanza/models/common/seq2seq_model.py

use std::{collections::HashMap, path::Path};

use anyhow::{bail, Context, Result};

use super::{Args, Vocab};
use crate::engine::{ModelSession, SessionOptions, TensorData, TensorOutput};
use crate::unicode::python_lower;

const UNKNOWN_ID: usize = 1;
const START_ID: usize = 2;
const END_ID: usize = 3;

pub(super) struct Sources {
    pub(super) chars: Vec<Vec<char>>,
    pub(super) lengths: Vec<usize>,
    pub(super) size: usize,
    pub(super) numbers: Vec<i64>,
}

impl Sources {
    pub(super) fn new(texts: &[String], vocab: &Vocab) -> Self {
        let chars: Vec<Vec<char>> = texts.iter().map(|s| s.chars().collect()).collect();
        let lengths: Vec<usize> = chars.iter().map(|c| c.len() + 2).collect();
        let size = lengths.iter().copied().max().unwrap_or(0);
        let mut numbers = vec![0; texts.len() * size];
        for (i, cs) in chars.iter().enumerate() {
            numbers[i * size] = START_ID as i64;
            for (j, c) in cs.iter().enumerate() {
                let value = if vocab.lower {
                    python_lower(&c.to_string())
                } else {
                    c.to_string()
                };
                numbers[i * size + j + 1] = *vocab.ids.get(&value).unwrap_or(&UNKNOWN_ID) as i64;
            }
            numbers[i * size + cs.len() + 1] = END_ID as i64;
        }
        Self {
            chars,
            lengths,
            size,
            numbers,
        }
    }

    pub(super) fn feeds(&self) -> Vec<(&'static str, TensorData)> {
        vec![
            (
                "src",
                TensorData::i64(vec![self.chars.len(), self.size], self.numbers.clone()),
            ),
            (
                "lengths",
                TensorData::i64(
                    vec![self.lengths.len()],
                    self.lengths.iter().map(|&x| x as i64).collect(),
                ),
            ),
        ]
    }
}

fn argmax(data: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..data.len() {
        if data[i] > data[best] {
            best = i;
        }
    }
    best
}

pub(super) fn output<'a>(
    outputs: &'a HashMap<String, TensorOutput>,
    key: &str,
) -> Result<&'a TensorOutput> {
    outputs
        .get(key)
        .with_context(|| format!("Sequence output {key} is missing"))
}

fn take_output(outputs: &mut HashMap<String, TensorOutput>, key: &str) -> Result<TensorOutput> {
    outputs
        .remove(key)
        .with_context(|| format!("Sequence output {key} is missing"))
}

fn feed_f32(output: &TensorOutput) -> TensorData {
    TensorData::f32(output.shape.clone(), output.data.clone())
}

pub(super) fn repeat_f32(input: TensorOutput, count: usize) -> TensorOutput {
    let mut shape = input.shape;
    shape[0] *= count;
    let mut data = Vec::with_capacity(input.data.len() * count);
    for _ in 0..count {
        data.extend_from_slice(&input.data);
    }
    TensorOutput { shape, data }
}

pub(super) fn reorder(input: TensorOutput, rows: &[usize]) -> Result<TensorOutput> {
    let width: usize = input.shape[1..].iter().product();
    let mut data = Vec::with_capacity(rows.len() * width);
    for &row in rows {
        data.extend_from_slice(
            input
                .data
                .get(row * width..(row + 1) * width)
                .context("Sequence state row is outside output tensor")?,
        );
    }
    let mut shape = input.shape;
    shape[0] = rows.len();
    Ok(TensorOutput { shape, data })
}

fn softplus(value: f64) -> f64 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}
fn logadd(a: f64, b: f64) -> f64 {
    let maximum = a.max(b);
    maximum + ((a - maximum).exp() + (b - maximum).exp()).ln()
}

pub(super) fn sequence_scores(
    args: &Args,
    outputs: &HashMap<String, TensorOutput>,
    inputs: &Sources,
    batch_size: usize,
    beam_size: usize,
    never_unk: bool,
) -> Result<(Vec<f32>, usize)> {
    let generation_logprobs = &output(outputs, "logprobs")?.data;
    let output_vocab_size = if args.copy {
        args.vocab_size
            .max(inputs.numbers.iter().copied().max().unwrap_or(0) as usize + 1)
    } else {
        args.vocab_size
    };
    if output_vocab_size == 0
        || generation_logprobs.len() != batch_size * beam_size * args.vocab_size
    {
        bail!("Sequence vocabulary/output dimensions are inconsistent");
    }
    let mut combined_scores = vec![0.0; batch_size * beam_size * output_vocab_size];
    if !args.copy {
        combined_scores.copy_from_slice(generation_logprobs);
        if never_unk {
            for row in 0..batch_size * beam_size {
                combined_scores[row * output_vocab_size + UNKNOWN_ID] = f32::NEG_INFINITY;
            }
        }
        return Ok((combined_scores, output_vocab_size));
    }
    let copy_gates = &output(outputs, "copygate")?.data;
    let log_attention = &output(outputs, "logattn")?.data;
    if copy_gates.len() < batch_size * beam_size
        || log_attention.len() != batch_size * beam_size * inputs.size
    {
        bail!("Sequence copy output dimensions are inconsistent");
    }
    for beam_index in 0..beam_size {
        for example_index in 0..batch_size {
            let row = beam_index * batch_size + example_index;
            let mut copy_probabilities = vec![0.0_f64; output_vocab_size];
            // Stanza repeats context/state in beam decoding but leaves the source
            // at its original batch size. Consequently only beam 0 copies.
            if beam_size == 1 || beam_index == 0 {
                for source_position in 0..inputs.size {
                    let source_id =
                        inputs.numbers[example_index * inputs.size + source_position] as usize;
                    let attention_probability =
                        (log_attention[row * inputs.size + source_position] as f64).exp();
                    copy_probabilities[source_id] += attention_probability;
                }
            }
            let no_copy_log = -softplus(copy_gates[row] as f64);
            let copy_log = -softplus(-(copy_gates[row] as f64));
            for token_id in 0..output_vocab_size {
                let generation_id = if token_id < args.vocab_size {
                    token_id
                } else {
                    UNKNOWN_ID
                };
                combined_scores[row * output_vocab_size + token_id] =
                    if never_unk && token_id == UNKNOWN_ID {
                        f32::NEG_INFINITY
                    } else {
                        logadd(
                            generation_logprobs[row * args.vocab_size + generation_id] as f64
                                + no_copy_log,
                            if copy_probabilities[token_id] > 0.0 {
                                copy_probabilities[token_id].ln() + copy_log
                            } else {
                                -1e12
                            },
                        ) as f32
                    };
            }
        }
    }
    Ok((combined_scores, output_vocab_size))
}

pub(super) struct Sequence {
    encoder: ModelSession,
    decoder: ModelSession,
    args: Args,
}

pub(super) struct Prediction {
    pub(super) predictions: Vec<String>,
    pub(super) edits: Vec<usize>,
}

struct BeamState {
    done: bool,
    scores: Vec<f32>,
    previous: Vec<usize>,
    history: Vec<(Vec<usize>, Vec<usize>)>,
}

fn decode_ids(ids: impl IntoIterator<Item = usize>, vocab: &Vocab) -> Result<String> {
    let mut text = String::new();
    for id in ids {
        text.push_str(
            vocab
                .units
                .get(id)
                .context("Sequence prediction ID is outside vocabulary")?,
        );
    }
    Ok(text)
}

impl Sequence {
    pub(super) fn load(
        dir: &Path,
        prefix: &str,
        args: Args,
        options: &SessionOptions,
    ) -> Result<Self> {
        if args.vocab_size < 4 || args.max_dec_len == 0 {
            bail!("Invalid sequence decoder configuration");
        }
        let encoder = ModelSession::load(&dir.join(format!("{prefix}_encoder.onnx")), options)?;
        let decoder = ModelSession::load(&dir.join(format!("{prefix}_step.onnx")), options)?;
        Ok(Self {
            encoder,
            decoder,
            args,
        })
    }

    pub(super) fn predict(
        &mut self,
        texts: &[String],
        vocab: &Vocab,
        pos_ids: Vec<i64>,
        never_unk: bool,
    ) -> Result<Prediction> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(Prediction {
                predictions: vec![],
                edits: vec![],
            });
        }
        let inputs = Sources::new(texts, vocab);
        let mut feeds = inputs.feeds();
        if self.encoder.has_input("pos") {
            feeds.push(("pos", TensorData::i64(vec![batch_size], pos_ids)));
        }
        let mut encoded = self.encoder.run(feeds)?;
        let edit = &output(&encoded, "edit")?.data;
        if edit.len() != batch_size * 3 {
            bail!("Sequence edit dimensions are inconsistent");
        }
        let edits = edit.chunks_exact(3).map(argmax).collect();
        let pos_context_offset = usize::from(self.args.pos);
        let mut mask = vec![false; batch_size * (inputs.size + pos_context_offset)];
        for example_index in 0..batch_size {
            for j in
                inputs.lengths[example_index] + pos_context_offset..inputs.size + pos_context_offset
            {
                mask[example_index * (inputs.size + pos_context_offset) + j] = true;
            }
        }
        let mut context = take_output(&mut encoded, "ctx")?;
        let mut hidden_state = take_output(&mut encoded, "h")?;
        let mut cell_state = take_output(&mut encoded, "c")?;
        let beam = self.args.beam_size.max(1);
        if beam == 1 {
            let mut previous = vec![START_ID as i64; batch_size];
            let mut done = vec![false; batch_size];
            let mut predicted_ids = vec![Vec::new(); batch_size];
            for _ in 0..self.args.max_dec_len {
                if done.iter().all(|&v| v) {
                    break;
                }
                let mut outputs = self.decoder.run(vec![
                    ("previous", TensorData::i64(vec![batch_size], previous)),
                    ("h", feed_f32(&hidden_state)),
                    ("c", feed_f32(&cell_state)),
                    ("ctx", feed_f32(&context)),
                    (
                        "mask",
                        TensorData::bool(
                            vec![batch_size, inputs.size + pos_context_offset],
                            mask.clone(),
                        ),
                    ),
                ])?;
                let (scores, vocab_size) =
                    sequence_scores(&self.args, &outputs, &inputs, batch_size, 1, never_unk)?;
                previous = (0..batch_size)
                    .map(|example_index| {
                        argmax(
                            &scores[example_index * vocab_size..(example_index + 1) * vocab_size],
                        ) as i64
                    })
                    .collect();
                for example_index in 0..batch_size {
                    if !done[example_index] {
                        if previous[example_index] == END_ID as i64 {
                            done[example_index] = true;
                        } else {
                            predicted_ids[example_index].push(previous[example_index] as usize);
                        }
                    }
                }
                hidden_state = take_output(&mut outputs, "next_h")?;
                cell_state = take_output(&mut outputs, "next_c")?;
            }
            let predictions = predicted_ids
                .into_iter()
                .map(|token_ids| decode_ids(token_ids, vocab))
                .collect::<Result<_>>()?;
            return Ok(Prediction { predictions, edits });
        }
        // Rows are beam-major: beam_index * batch + example_index. Recurrent
        // states, scores and parent-row selection must keep this same layout.
        context = repeat_f32(context, beam);
        hidden_state = repeat_f32(hidden_state, beam);
        cell_state = repeat_f32(cell_state, beam);
        let original_mask = mask;
        mask = Vec::with_capacity(original_mask.len() * beam);
        for _ in 0..beam {
            mask.extend_from_slice(&original_mask);
        }
        let mut beams: Vec<BeamState> = (0..batch_size)
            .map(|_| {
                let mut previous = vec![0; beam];
                previous[0] = START_ID;
                BeamState {
                    done: false,
                    scores: vec![0.0; beam],
                    previous,
                    history: vec![],
                }
            })
            .collect();
        for step in 0..self.args.max_dec_len {
            let previous = (0..batch_size * beam)
                .map(|row| beams[row % batch_size].previous[row / batch_size] as i64)
                .collect();
            let mut outputs = self.decoder.run(vec![
                (
                    "previous",
                    TensorData::i64(vec![batch_size * beam], previous),
                ),
                ("h", feed_f32(&hidden_state)),
                ("c", feed_f32(&cell_state)),
                ("ctx", feed_f32(&context)),
                (
                    "mask",
                    TensorData::bool(
                        vec![batch_size * beam, inputs.size + pos_context_offset],
                        mask.clone(),
                    ),
                ),
            ])?;
            let (scores, vocab_size) =
                sequence_scores(&self.args, &outputs, &inputs, batch_size, beam, never_unk)?;
            let mut rows = vec![0; batch_size * beam];
            for example_index in 0..batch_size {
                let state = &mut beams[example_index];
                if !state.done {
                    let mut candidates = Vec::with_capacity(beam * vocab_size);
                    for beam_index in 0..if step == 0 { 1 } else { beam } {
                        for id in 0..vocab_size {
                            candidates.push((
                                scores[(beam_index * batch_size + example_index) * vocab_size + id]
                                    + if step == 0 {
                                        0.0
                                    } else {
                                        state.scores[beam_index]
                                    },
                                beam_index,
                                id,
                            ));
                        }
                    }
                    // Keep insertion order for tied scores: ties can change the decoded word.
                    candidates
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    if candidates.len() < beam {
                        bail!("Beam exceeds sequence vocabulary");
                    }
                    state.scores = candidates[..beam]
                        .iter()
                        .map(|candidate| candidate.0)
                        .collect();
                    state.previous = candidates[..beam]
                        .iter()
                        .map(|candidate| candidate.2)
                        .collect();
                    state.history.push((
                        candidates[..beam]
                            .iter()
                            .map(|candidate| candidate.1)
                            .collect(),
                        state.previous.clone(),
                    ));
                    state.done = state.previous[0] == END_ID;
                }
                let parents = &state.history.last().context("Empty beam history")?.0;
                for beam_index in 0..beam {
                    rows[beam_index * batch_size + example_index] =
                        parents[beam_index] * batch_size + example_index;
                }
            }
            hidden_state = reorder(take_output(&mut outputs, "next_h")?, &rows)?;
            cell_state = reorder(take_output(&mut outputs, "next_c")?, &rows)?;
            if beams.iter().all(|b| b.done) {
                break;
            }
        }
        let predictions = beams
            .into_iter()
            .map(|state| {
                let mut beam_index = argmax(&state.scores);
                let mut token_ids = Vec::with_capacity(state.history.len());
                for (parents, values) in state.history.iter().rev() {
                    token_ids.push(values[beam_index]);
                    beam_index = parents[beam_index];
                }
                token_ids.reverse();
                decode_ids(token_ids.into_iter().take_while(|&id| id != END_ID), vocab)
            })
            .collect::<Result<_>>()?;
        Ok(Prediction { predictions, edits })
    }
}
