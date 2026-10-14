// Adapted from Stanford Stanza 1.11.0 POS preprocessing and vocabulary decoding.
// Copyright 2019 The Board of Trustees of The Leland Stanford Junior University.
// Upstream: https://github.com/stanfordnlp/stanza/tree/v1.11.0/stanza/models/pos
// Licensed under Apache-2.0; see this project's LICENSE and NOTICE.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::engine::{ModelSession, SessionOptions, TensorData, TensorOutput};
use crate::types::{Sentence, Word};
use crate::unicode::python_lower;

const MAX_PADDED_WORDS: usize = 5000;
const MAX_PADDED_WORDS_CPU: usize = 128;
const MAX_PADDED_CHARS: usize = 128;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum VocabKey {
    Text(String),
    Number(usize),
}

impl VocabKey {
    fn name(&self) -> String {
        match self {
            Self::Text(value) => value.clone(),
            Self::Number(value) => value.to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum IdUnits {
    Flat(Vec<String>),
    Composite(HashMap<String, Vec<String>>),
}

#[derive(Debug, Deserialize)]
struct Vocab {
    #[serde(default)]
    lower: bool,
    #[serde(default)]
    ignore: Vec<String>,
    #[serde(default)]
    composite: bool,
    #[serde(default)]
    keyed: bool,
    #[serde(default)]
    sep: Option<String>,
    #[serde(default)]
    keys: Vec<VocabKey>,
    // Input-only vocabularies (including pretrained embeddings) have no heads.
    #[serde(default)]
    sizes: Vec<usize>,
    #[serde(rename = "_unit2id", default)]
    // Composite vocabularies store one nested map per head. Input lookups use
    // only flat word/character vocabularies; decoding uses id_to_unit below.
    unit_to_id: HashMap<String, serde_json::Value>,
    #[serde(rename = "_id2unit")]
    id_to_unit: IdUnits,
}

impl Vocab {
    fn id(&self, text: &str) -> i64 {
        if self.ignore.iter().any(|ignored| ignored == text) {
            return 2;
        }
        if self.lower {
            self.unit_to_id
                .get(&python_lower(text))
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(1)
        } else {
            self.unit_to_id
                .get(text)
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(1)
        }
    }

    fn decode(&self, logits: &[f32]) -> Result<String> {
        if !self.composite {
            let width = *self.sizes.first().context("flat vocabulary has no size")?;
            let value = argmax(logits, width)?;
            if value == 2 && !self.ignore.is_empty() {
                return Ok("_".to_string());
            }
            let IdUnits::Flat(units) = &self.id_to_unit else {
                bail!("flat vocabulary has composite units");
            };
            return units
                .get(value)
                .cloned()
                .context("predicted vocabulary index is out of bounds");
        }
        ensure!(
            self.keys.len() == self.sizes.len(),
            "composite vocabulary keys and sizes differ"
        );
        let IdUnits::Composite(units) = &self.id_to_unit else {
            bail!("composite vocabulary has flat units");
        };
        let mut parts = Vec::new();
        let mut offset = 0;
        for (key, width) in self.keys.iter().zip(&self.sizes) {
            let scores = logits
                .get(offset..offset + width)
                .context("composite logits are shorter than the vocabulary")?;
            let value = argmax(scores, *width)?;
            if value != 2 {
                let key = key.name();
                let unit = units
                    .get(&key)
                    .and_then(|values| values.get(value))
                    .context("predicted composite vocabulary index is out of bounds")?;
                parts.push(if self.keyed {
                    format!("{key}={unit}")
                } else {
                    unit.clone()
                });
            }
            offset += width;
        }
        Ok(if parts.is_empty() {
            "_".to_string()
        } else {
            parts.join(self.sep.as_deref().unwrap_or(""))
        })
    }
}

fn argmax(logits: &[f32], width: usize) -> Result<usize> {
    ensure!(width > 0 && logits.len() >= width, "invalid logits width");
    let mut best = 0;
    for index in 1..width {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    Ok(best)
}

#[derive(Debug, Deserialize)]
struct Metadata {
    vocab: HashMap<String, Vocab>,
    has_char: bool,
    has_pretrain: bool,
    pretrained_vocab: Option<Vocab>,
    pretrained_shape: [usize; 2],
    character_feature_dim: usize,
    #[serde(default)]
    contexts: Vec<ContextMetadata>,
}

impl Metadata {
    fn vocab(&self, name: &str) -> Result<&Vocab> {
        self.vocab
            .get(name)
            .with_context(|| format!("missing POS {name} vocabulary"))
    }
}

#[derive(Debug, Deserialize)]
struct ContextMetadata {
    filename: String,
    backwards: bool,
    vocab: Vocab,
    feature_dim: usize,
}

struct CharacterContext {
    metadata: ContextMetadata,
    session: ModelSession,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WordLocation {
    sentence: usize,
    token: usize,
    word: usize,
}

impl WordLocation {
    fn get(self, sentences: &[Sentence]) -> &Word {
        &sentences[self.sentence].tokens[self.token].words[self.word]
    }

    fn get_mut(self, sentences: &mut [Sentence]) -> &mut Word {
        &mut sentences[self.sentence].tokens[self.token].words[self.word]
    }
}

#[derive(Debug)]
struct PosRow {
    words: Vec<WordLocation>,
}

/// Sort by length and bound padding without losing positions in the input document.
fn plan_pos_batches(sentences: &[Sentence], max_padded_words: usize) -> Result<Vec<Vec<PosRow>>> {
    ensure!(max_padded_words > 0, "POS batch limit must be positive");
    let mut sorted: Vec<_> = sentences
        .iter()
        .enumerate()
        .map(|(sentence_index, sentence)| PosRow {
            words: sentence
                .tokens
                .iter()
                .enumerate()
                .flat_map(|(token_index, token)| {
                    (0..token.words.len()).map(move |word_index| WordLocation {
                        sentence: sentence_index,
                        token: token_index,
                        word: word_index,
                    })
                })
                .collect(),
        })
        .filter(|row| !row.words.is_empty())
        .collect();
    // Stable sorting also preserves source ordering when lengths are equal.
    sorted.sort_by_key(|row| std::cmp::Reverse(row.words.len()));
    let mut batches = Vec::new();
    let mut batch: Vec<PosRow> = Vec::new();
    for row in sorted {
        if !batch.is_empty() && (batch.len() + 1) * batch[0].words.len() > max_padded_words {
            batches.push(std::mem::take(&mut batch));
        }
        batch.push(row);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    Ok(batches)
}

#[derive(Debug, PartialEq, Eq)]
struct CharWord {
    ids: Vec<i64>,
    positions: Vec<usize>,
}

/// The ordinary encoder treats words independently; keep their source positions.
fn plan_char_words(texts: &[&str], vocab: &Vocab, deduplicate: bool) -> Vec<CharWord> {
    let mut rows: Vec<CharWord> = Vec::new();
    let mut by_text: HashMap<&str, usize> = HashMap::new();
    for (position, text) in texts.iter().enumerate() {
        if deduplicate {
            if let Some(&index) = by_text.get(text) {
                rows[index].positions.push(position);
                continue;
            }
            by_text.insert(*text, rows.len());
        }
        rows.push(CharWord {
            ids: text
                .chars()
                .map(|character| vocab.id(&character.to_string()))
                .collect(),
            positions: vec![position],
        });
    }
    rows.sort_by_key(|row| std::cmp::Reverse(row.ids.len()));
    rows
}

#[derive(Debug, PartialEq, Eq)]
struct CharRange {
    start: usize,
    end: usize,
    max_chars: usize,
}

/// Bound independent word padding without splitting oversized words.
fn plan_char_ranges(lengths: &[usize], max_padded_chars: usize) -> Result<Vec<CharRange>> {
    ensure!(
        max_padded_chars > 0,
        "Character batch budget must be positive"
    );
    ensure!(
        lengths.iter().all(|&length| length > 0),
        "POS character encoder cannot process empty words"
    );
    ensure!(
        lengths.windows(2).all(|pair| pair[0] >= pair[1]),
        "Character lengths must be sorted descending"
    );
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < lengths.len() {
        let max_chars = lengths[start];
        let mut end = start + 1;
        while end < lengths.len() && end - start < max_padded_chars / max_chars {
            end += 1;
        }
        ranges.push(CharRange {
            start,
            end,
            max_chars,
        });
        start = end;
    }
    Ok(ranges)
}

fn scatter_char_features(
    rows: &[CharWord],
    encoded: &[f32],
    destinations: &[usize],
    features: &mut [f32],
    feature_width: usize,
) -> Result<()> {
    ensure!(
        rows.len().checked_mul(feature_width) == Some(encoded.len()),
        "Unexpected word-character feature length"
    );
    for (index, row) in rows.iter().enumerate() {
        let source = &encoded[index * feature_width..(index + 1) * feature_width];
        for position in &row.positions {
            let destination = *destinations
                .get(*position)
                .context("Word-character position is out of bounds")?;
            features
                .get_mut(destination..destination + feature_width)
                .context("Word-character destination is out of bounds")?
                .copy_from_slice(source);
        }
    }
    Ok(())
}

#[derive(Debug)]
struct ContextRow {
    original_index: usize,
    chars: Vec<i64>,
    offsets: Vec<usize>,
}

/// CharLM inputs begin with a newline and record features on the following space.
fn context_row(words: &[&str], original_index: usize, metadata: &ContextMetadata) -> ContextRow {
    let mut chars = vec!['\n'];
    let mut offsets = Vec::with_capacity(words.len());
    if metadata.backwards {
        for word in words.iter().rev() {
            chars.extend(word.chars().rev());
            chars.push(' ');
            offsets.push(chars.len() - 1);
        }
        offsets.reverse();
    } else {
        for word in words {
            chars.extend(word.chars());
            chars.push(' ');
            offsets.push(chars.len() - 1);
        }
    }
    ContextRow {
        original_index,
        chars: chars
            .into_iter()
            .map(|character| metadata.vocab.id(&character.to_string()))
            .collect(),
        offsets,
    }
}

fn read_pretrained(path: &Path, shape: [usize; 2]) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let elements = shape[0]
        .checked_mul(shape[1])
        .context("pretrained shape overflows")?;
    ensure!(
        elements.checked_mul(4) == Some(bytes.len()),
        "pretrained.f32 length does not match its declared shape"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn feature_output(
    mut output: HashMap<String, TensorOutput>,
    shape: &[usize],
) -> Result<TensorOutput> {
    let features = output
        .remove("features")
        .context("missing character features")?;
    ensure!(
        features.shape == shape,
        "unexpected character feature shape: {:?}, expected {shape:?}",
        features.shape
    );
    ensure!(
        features.data.len() == shape.iter().product::<usize>(),
        "unexpected character feature length"
    );
    Ok(features)
}

/// Encode full sentence context, then copy each word's boundary features back
/// into the original POS row. Character-language-model sorting is independent
/// of the POS sentence order, so original_index determines each destination.
fn fill_context_features(
    contexts: &mut [CharacterContext],
    sentences: &[Sentence],
    sorted: &[PosRow],
    words: usize,
    feature_width: usize,
    character_features: &mut [f32],
) -> Result<()> {
    let batch = sorted.len();
    let mut context_feature_offset = 0;
    for context in contexts {
        let cm = &context.metadata;
        let mut rows: Vec<_> = sorted
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let texts: Vec<_> = row
                    .words
                    .iter()
                    .map(|location| location.get(sentences).text.as_str())
                    .collect();
                context_row(&texts, index, cm)
            })
            .collect();
        rows.sort_by_key(|row| std::cmp::Reverse(row.chars.len()));
        let max_chars = rows[0].chars.len();
        let mut char_ids = vec![cm.vocab.id(" "); batch * max_chars];
        for (index, row) in rows.iter().enumerate() {
            char_ids[index * max_chars..index * max_chars + row.chars.len()]
                .copy_from_slice(&row.chars);
        }
        let encoded = feature_output(
            context.session.run(vec![
                ("chars", TensorData::i64(vec![batch, max_chars], char_ids)),
                (
                    "lengths",
                    TensorData::i64(
                        vec![batch],
                        rows.iter().map(|row| row.chars.len() as i64).collect(),
                    ),
                ),
            ])?,
            &[batch, max_chars, cm.feature_dim],
        )?;
        for (index, row) in rows.iter().enumerate() {
            for (word_index, offset) in row.offsets.iter().enumerate() {
                let source = (index * max_chars + offset) * cm.feature_dim;
                let destination = (row.original_index * words + word_index) * feature_width
                    + context_feature_offset;
                character_features[destination..destination + cm.feature_dim]
                    .copy_from_slice(&encoded.data[source..source + cm.feature_dim]);
            }
        }
        context_feature_offset += cm.feature_dim;
    }
    Ok(())
}

/// Portable Stanza POS/XPOS/morphology, including optional character language models.
pub struct Pos {
    chars: Option<ModelSession>,
    tagger: ModelSession,
    metadata: Metadata,
    pretrained: Option<Vec<f32>>,
    contexts: Vec<CharacterContext>,
    max_padded_words: usize,
    max_padded_chars: usize,
}

impl Pos {
    pub fn load(dir: &Path, options: &SessionOptions) -> Result<Self> {
        let mut metadata: Metadata = serde_json::from_slice(
            &fs::read(dir.join("pos-vocab.json")).context("read POS metadata")?,
        )
        .context("parse POS metadata")?;
        for name in ["word", "upos", "xpos", "feats"] {
            metadata.vocab(name)?;
        }
        if metadata.has_char {
            metadata.vocab("char")?;
        }
        let pretrained = if metadata.has_pretrain {
            ensure!(
                metadata.pretrained_vocab.is_some(),
                "missing pretrained vocabulary"
            );
            Some(read_pretrained(
                &dir.join("pretrained.f32"),
                metadata.pretrained_shape,
            )?)
        } else {
            None
        };
        let chars = if metadata.has_char {
            Some(ModelSession::load(&dir.join("pos-char.onnx"), options)?)
        } else {
            None
        };
        let tagger = ModelSession::load(&dir.join("pos-tagger.onnx"), options)?;
        let mut contexts = Vec::new();
        for context in std::mem::take(&mut metadata.contexts) {
            let session = ModelSession::load(&dir.join(&context.filename), options)?;
            contexts.push(CharacterContext {
                metadata: context,
                session,
            });
        }
        ensure!(
            contexts
                .iter()
                .map(|context| context.metadata.feature_dim)
                .sum::<usize>()
                <= metadata.character_feature_dim,
            "contextual character dimensions exceed the POS feature width"
        );
        Ok(Self {
            chars,
            tagger,
            metadata,
            pretrained,
            contexts,
            max_padded_words: if options.provider == "cpu" {
                MAX_PADDED_WORDS_CPU
            } else {
                MAX_PADDED_WORDS
            },
            // Smaller CPU batches avoid padding work. Retain the original
            // dispatch count for accelerators until their tradeoff is measured.
            max_padded_chars: if options.provider == "cpu" {
                MAX_PADDED_CHARS
            } else {
                usize::MAX
            },
        })
    }

    pub fn process(&mut self, sentences: &mut [Sentence]) -> Result<()> {
        for batch in plan_pos_batches(sentences, self.max_padded_words)? {
            self.process_batch(sentences, &batch)?;
        }
        Ok(())
    }

    fn process_batch(&mut self, sentences: &mut [Sentence], sorted: &[PosRow]) -> Result<()> {
        let metadata = &self.metadata;
        let flat: Vec<_> = sorted
            .iter()
            .flat_map(|row| row.words.iter().copied())
            .collect();
        let batch = sorted.len();
        let words = sorted[0].words.len();
        let feature_width = metadata.character_feature_dim;
        let mut character_features = vec![0.0f32; batch * words * feature_width];
        fill_context_features(
            &mut self.contexts,
            sentences,
            sorted,
            words,
            feature_width,
            &mut character_features,
        )?;

        if let Some(chars) = &mut self.chars {
            let char_vocab = metadata.vocab("char")?;
            let texts: Vec<_> = flat
                .iter()
                .map(|location| location.get(sentences).text.as_str())
                .collect();
            let char_words =
                plan_char_words(&texts, char_vocab, self.max_padded_chars != usize::MAX);
            let lengths: Vec<_> = char_words.iter().map(|row| row.ids.len()).collect();
            let destinations: Vec<_> = sorted
                .iter()
                .enumerate()
                .flat_map(|(index, row)| {
                    (0..row.words.len())
                        .map(move |word_index| (index * words + word_index) * feature_width)
                })
                .collect();
            for range in plan_char_ranges(&lengths, self.max_padded_chars)? {
                let rows = &char_words[range.start..range.end];
                let count = rows.len();
                let width = range.max_chars;
                let mut char_ids = vec![0; count * width];
                for (index, row) in rows.iter().enumerate() {
                    char_ids[index * width..index * width + row.ids.len()]
                        .copy_from_slice(&row.ids);
                }
                let encoded = feature_output(
                    chars.run(vec![
                        ("chars", TensorData::i64(vec![count, width], char_ids)),
                        (
                            "lengths",
                            TensorData::i64(
                                vec![count],
                                rows.iter().map(|row| row.ids.len() as i64).collect(),
                            ),
                        ),
                    ])?,
                    &[count, feature_width],
                )?;
                scatter_char_features(
                    rows,
                    &encoded.data,
                    &destinations,
                    &mut character_features,
                    feature_width,
                )?;
            }
        }
        // Pretrained embeddings and word IDs share padded [batch, words] rows.
        let pretrained_width = metadata.pretrained_shape[1];
        let mut pretrained_vectors = vec![0.0f32; batch * words * pretrained_width];
        let mut word_ids = vec![0; batch * words];
        let word_vocab = metadata.vocab("word")?;
        for (index, row) in sorted.iter().enumerate() {
            for (word_index, location) in row.words.iter().enumerate() {
                let word = location.get(sentences);
                let offset = index * words + word_index;
                word_ids[offset] = word_vocab.id(&word.text);
                if let (Some(pretrained), Some(vocab)) =
                    (&self.pretrained, &metadata.pretrained_vocab)
                {
                    let id = vocab.id(&python_lower(&word.text).replace(' ', "\u{a0}"));
                    let start = usize::try_from(id).context("negative pretrained index")?
                        * pretrained_width;
                    let embedding = pretrained
                        .get(start..start + pretrained_width)
                        .context("pretrained index out of bounds")?;
                    pretrained_vectors[offset * pretrained_width..(offset + 1) * pretrained_width]
                        .copy_from_slice(embedding);
                }
            }
        }
        let mut feeds = vec![
            ("words", TensorData::i64(vec![batch, words], word_ids)),
            (
                "lengths",
                TensorData::i64(
                    vec![batch],
                    sorted.iter().map(|row| row.words.len() as i64).collect(),
                ),
            ),
        ];
        if self.tagger.has_input("pretrained_vectors") {
            feeds.push((
                "pretrained_vectors",
                TensorData::f32(vec![batch, words, pretrained_width], pretrained_vectors),
            ));
        }
        if self.tagger.has_input("character_features") {
            feeds.push((
                "character_features",
                TensorData::f32(vec![batch, words, feature_width], character_features),
            ));
        }
        // Decode all three heads before mutating their original word locations.
        let result = self.tagger.run(feeds)?;
        let outputs: Vec<_> = ["upos", "xpos", "feats"]
            .into_iter()
            .map(|name| {
                let output = result
                    .get(&format!("{name}_logits"))
                    .with_context(|| format!("missing {name} logits"))?;
                ensure!(
                    output.shape.len() == 3 && output.shape[0] == batch && output.shape[1] == words,
                    "unexpected {name} logits shape"
                );
                let width = output.shape[2];
                ensure!(
                    output.data.len() == batch * words * width,
                    "unexpected {name} logits length"
                );
                let vocab = metadata.vocab(name)?;
                ensure!(
                    vocab.sizes.iter().sum::<usize>() == width,
                    "{name} logits width differs from vocabulary"
                );
                Ok((vocab, output, width))
            })
            .collect::<Result<_>>()?;
        for (index, row) in sorted.iter().enumerate() {
            for (word_index, location) in row.words.iter().enumerate() {
                let offset = index * words + word_index;
                let mut annotations = Vec::with_capacity(3);
                for (vocab, output, width) in &outputs {
                    let value = vocab.decode(&output.data[offset * width..(offset + 1) * width])?;
                    annotations.push(if value == "_" { None } else { Some(value) });
                }
                let word = location.get_mut(sentences);
                word.upos = annotations[0].take();
                word.xpos = annotations[1].take();
                word.feats = annotations[2].take();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
