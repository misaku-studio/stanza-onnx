//! Adapted from Stanford Stanza 1.11.0 lemma/MWT processing and sequence decoding.
//! Copyright 2019 The Board of Trustees of The Leland Stanford Junior University.
//! Upstream: https://github.com/stanfordnlp/stanza/tree/v1.11.0/stanza/models/lemma
//! and https://github.com/stanfordnlp/stanza/tree/v1.11.0/stanza/models/mwt
//! Licensed under Apache-2.0; see this project's LICENSE and NOTICE.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::engine::{ModelSession, SessionOptions};
use crate::types::{Sentence, Word};

mod sequence;
use crate::unicode::{
    python_islower, python_isupper, python_lower, python_upper, python_uppercase, python_whitespace,
};
use sequence::{output, Sequence, Sources};

#[derive(Clone, Debug, Default, Deserialize)]
struct Vocab {
    #[serde(rename = "_id2unit", default)]
    units: Vec<String>,
    #[serde(rename = "_unit2id", default)]
    ids: HashMap<String, usize>,
    #[serde(default)]
    lower: bool,
}

fn one() -> usize {
    1
}
fn fifty() -> usize {
    50
}

#[derive(Clone, Debug, Deserialize)]
struct Args {
    #[serde(default)]
    ensemble_dict: bool,
    #[serde(default = "fifty")]
    max_dec_len: usize,
    #[serde(default = "one")]
    beam_size: usize,
    #[serde(default)]
    edit: bool,
    #[serde(default)]
    pos: bool,
    #[serde(default)]
    copy: bool,
    #[serde(default = "fifty")]
    batch_size: usize,
    #[serde(default)]
    caseless: bool,
    #[serde(default)]
    vocab_size: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            ensemble_dict: false,
            max_dec_len: 50,
            beam_size: 1,
            edit: false,
            pos: false,
            copy: false,
            batch_size: 50,
            caseless: false,
            vocab_size: 0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct MwtMeta {
    kind: String,
    #[serde(default)]
    args: Args,
    #[serde(default)]
    vocab: Vocab,
    #[serde(default)]
    dict: HashMap<String, String>,
    #[serde(default)]
    decode_beam_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct LemmaMeta {
    kind: String,
    #[serde(default)]
    args: Args,
    #[serde(default)]
    char: Vocab,
    #[serde(default)]
    pos: Vocab,
    #[serde(default)]
    word_dict: HashMap<String, String>,
    #[serde(default)]
    composite_dict: Vec<(String, String, String)>,
}

fn expanded(vocab: &Vocab, texts: &[String]) -> Vocab {
    // Sorting Unicode scalars matches Stanza's dynamic copy vocabulary.
    let unknown: BTreeSet<char> = texts
        .iter()
        .flat_map(|t| t.chars())
        .filter(|c| !vocab.ids.contains_key(&c.to_string()))
        .collect();
    let mut result = vocab.clone();
    result
        .units
        .extend(unknown.into_iter().map(|c| c.to_string()));
    result.ids = result
        .units
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), i))
        .collect();
    result
}

fn dictionary_expansion(text: &str, dictionary: &HashMap<String, String>) -> Option<String> {
    if let Some(exact) = dictionary.get(text) {
        return Some(exact.clone());
    }
    let lower = python_lower(text);
    if python_isupper(text) {
        return dictionary.get(&lower).map(|value| python_upper(value));
    }
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        let tail = chars.as_str();
        if python_uppercase(first) && python_islower(tail) {
            return dictionary.get(&lower).map(|value| {
                let mut chars = value.chars();
                chars
                    .next()
                    .map(|c| python_upper(&c.to_string()) + chars.as_str())
                    .unwrap_or_default()
            });
        }
    }
    None
}

/// Stanza multiword-token expansion, including dictionary and neural variants.
pub struct Mwt {
    meta: MwtMeta,
    session: Option<ModelSession>,
    sequence: Option<Sequence>,
}

impl Mwt {
    pub fn load(dir: &Path, options: &SessionOptions) -> Result<Self> {
        let meta: MwtMeta =
            serde_json::from_slice(&fs::read(dir.join("mwt.json"))?).context("Invalid mwt.json")?;
        if !["none", "dictionary", "classifier", "seq2seq"].contains(&meta.kind.as_str()) {
            bail!("Unknown MWT model kind {}", meta.kind);
        }
        if meta.args.batch_size == 0 {
            bail!("MWT batch size must be positive");
        }
        let session = if meta.kind == "classifier" {
            Some(ModelSession::load(&dir.join("mwt.onnx"), options)?)
        } else {
            None
        };
        let sequence = if meta.kind == "seq2seq" {
            let mut args = meta.args.clone();
            // Stanza calls predict positionally, so stock MWT decoding uses 5
            // beams even when the checkpoint's configured beam_size is 1.
            args.beam_size = meta.decode_beam_size.unwrap_or(5);
            Some(Sequence::load(dir, "mwt", args, options)?)
        } else {
            None
        };
        Ok(Self {
            meta,
            session,
            sequence,
        })
    }

    pub fn process(&mut self, sentences: &mut [Sentence]) -> Result<()> {
        if self.meta.kind == "none" {
            return Ok(());
        }
        let mut tokens = Vec::new();
        for (si, sentence) in sentences.iter().enumerate() {
            for (ti, token) in sentence.tokens.iter().enumerate() {
                if token.mwt {
                    tokens.push((si, ti, token.text.clone()));
                }
            }
        }
        if tokens.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = tokens.iter().map(|t| t.2.clone()).collect();
        let vocab = expanded(&self.meta.vocab, &texts);
        let mut sorted: Vec<usize> = (0..tokens.len()).collect();
        sorted.sort_by_key(|&i| {
            (
                std::cmp::Reverse(texts[i].chars().count()),
                std::cmp::Reverse(i),
            )
        });
        for part in sorted.chunks(self.meta.args.batch_size) {
            let part_texts: Vec<String> = part.iter().map(|&i| texts[i].clone()).collect();
            let predictions: Vec<String> = match self.meta.kind.as_str() {
                "dictionary" => part_texts
                    .iter()
                    .map(|t| dictionary_expansion(t, &self.meta.dict).unwrap_or_else(|| t.clone()))
                    .collect(),
                "classifier" => {
                    let inputs = Sources::new(&part_texts, &vocab);
                    let out = self
                        .session
                        .as_mut()
                        .context("MWT classifier is missing")?
                        .run(inputs.feeds())?;
                    let logits = &output(&out, "logits")?.data;
                    if logits.len() != part.len() * inputs.size * 2 {
                        bail!("MWT classifier output dimensions are inconsistent");
                    }
                    inputs
                        .chars
                        .iter()
                        .enumerate()
                        .map(|(i, chars)| {
                            let mut value = String::new();
                            for (j, &c) in chars.iter().enumerate() {
                                let index = (i * inputs.size + j + 1) * 2;
                                if logits[index + 1] > logits[index] {
                                    value.push(' ');
                                }
                                value.push(c);
                            }
                            value.trim_matches(python_whitespace).to_string()
                        })
                        .collect()
                }
                "seq2seq" => {
                    let predicted = self
                        .sequence
                        .as_mut()
                        .context("MWT sequence model is missing")?
                        .predict(&part_texts, &vocab, vec![1; part.len()], true)?
                        .predictions;
                    predicted
                        .into_iter()
                        .enumerate()
                        .map(|(i, value)| {
                            if !value.is_empty() && value.contains(' ') {
                                value
                            } else {
                                part_texts[i].clone()
                            }
                        })
                        .collect()
                }
                _ => bail!("Unexpected MWT model kind"),
            };
            for (&index, prediction) in part.iter().zip(predictions) {
                let (si, ti, _) = &tokens[index];
                let token = &mut sentences[*si].tokens[*ti];
                let expansion = if self.meta.args.ensemble_dict {
                    dictionary_expansion(&token.text, &self.meta.dict).unwrap_or(prediction)
                } else {
                    prediction
                };
                let mut pieces: Vec<&str> =
                    expansion.split(' ').filter(|s| !s.is_empty()).collect();
                if pieces.len() == 1 {
                    pieces = vec![&token.text];
                }
                token.words = pieces.into_iter().map(Word::new).collect();
            }
        }
        Ok(())
    }
}

fn set_lemma(word: &mut Word, value: String) {
    word.lemma = if value == "_" && word.text != "_" {
        None
    } else {
        Some(value)
    };
}

type WordIndex = (usize, usize, usize);

fn word_at(sentences: &[Sentence], index: WordIndex) -> &Word {
    &sentences[index.0].tokens[index.1].words[index.2]
}

fn word_at_mut(sentences: &mut [Sentence], index: WordIndex) -> &mut Word {
    &mut sentences[index.0].tokens[index.1].words[index.2]
}

/// Stanza lemmatization, including identity/dictionary/edit/copy variants.
pub struct Lemma {
    meta: LemmaMeta,
    composite: HashMap<(String, String), String>,
    sequence: Option<Sequence>,
}

impl Lemma {
    pub fn load(dir: &Path, options: &SessionOptions) -> Result<Self> {
        let mut meta: LemmaMeta = serde_json::from_slice(&fs::read(dir.join("lemma.json"))?)
            .context("Invalid lemma.json")?;
        if !["identity", "dictionary", "seq2seq"].contains(&meta.kind.as_str()) {
            bail!("Unknown lemma model kind {}", meta.kind);
        }
        if meta.args.batch_size == 0 {
            bail!("Lemma batch size must be positive");
        }
        let sequence = if meta.kind == "seq2seq" {
            Some(Sequence::load(dir, "lemma", meta.args.clone(), options)?)
        } else {
            None
        };
        let composite = std::mem::take(&mut meta.composite_dict)
            .into_iter()
            .map(|(text, pos, lemma)| ((text, pos), lemma))
            .collect();
        Ok(Self {
            meta,
            composite,
            sequence,
        })
    }

    pub fn process(&mut self, sentences: &mut [Sentence]) -> Result<()> {
        let mut positions = Vec::new();
        for (si, sentence) in sentences.iter().enumerate() {
            for (ti, token) in sentence.tokens.iter().enumerate() {
                for wi in 0..token.words.len() {
                    positions.push((si, ti, wi));
                }
            }
        }
        if self.meta.kind == "identity" {
            for index in positions {
                let word = word_at_mut(sentences, index);
                set_lemma(word, word.text.clone());
            }
            return Ok(());
        }
        let mut neural = Vec::new();
        for index in positions {
            let word = word_at_mut(sentences, index);
            let text = if self.meta.args.caseless {
                python_lower(&word.text)
            } else {
                word.text.clone()
            };
            let key = (
                text.clone(),
                word.upos.as_deref().unwrap_or("_").to_string(),
            );
            let known = self
                .composite
                .get(&key)
                .or_else(|| self.meta.word_dict.get(&text));
            if self.meta.kind == "dictionary" {
                let value = known.cloned().unwrap_or(text);
                set_lemma(word, if value.is_empty() { "_".into() } else { value });
            } else if self.meta.args.ensemble_dict && known.is_some() {
                let value = known.unwrap().clone();
                set_lemma(word, if value.is_empty() { "_".into() } else { value });
            } else {
                neural.push((index, text));
            }
        }
        if neural.is_empty() {
            return Ok(());
        }
        let mut vocab_texts = Vec::with_capacity(neural.len() * 3);
        for &(index, ref text) in &neural {
            vocab_texts.push(text.clone());
            vocab_texts.push(
                word_at(sentences, index)
                    .upos
                    .as_deref()
                    .unwrap_or("_")
                    .to_string(),
            );
            vocab_texts.push("_".to_string());
        }
        let vocab = expanded(&self.meta.char, &vocab_texts);
        let mut sorted: Vec<usize> = (0..neural.len()).collect();
        sorted.sort_by_key(|&i| {
            (
                std::cmp::Reverse(neural[i].1.chars().count()),
                std::cmp::Reverse(i),
            )
        });
        for part in sorted.chunks(self.meta.args.batch_size) {
            let texts: Vec<String> = part.iter().map(|&i| neural[i].1.clone()).collect();
            let pos = part
                .iter()
                .map(|&i| {
                    let word = word_at(sentences, neural[i].0);
                    let upos = word.upos.as_deref().unwrap_or("_");
                    let lookup = if self.meta.pos.lower {
                        python_lower(upos)
                    } else {
                        upos.to_string()
                    };
                    *self.meta.pos.ids.get(&lookup).unwrap_or(&1) as i64
                })
                .collect();
            let predicted = self
                .sequence
                .as_mut()
                .context("Lemma sequence model is missing")?
                .predict(&texts, &vocab, pos, false)?;
            for (i, &index) in part.iter().enumerate() {
                let word = word_at_mut(sentences, neural[index].0);
                let mut lemma = predicted.predictions[i].clone();
                if self.meta.args.edit {
                    if predicted.edits[i] == 1 {
                        lemma = word.text.clone();
                    } else if predicted.edits[i] == 2 {
                        lemma = python_lower(&word.text);
                    }
                }
                if lemma.is_empty() || lemma.contains("<UNK>") {
                    lemma = word.text.clone();
                }
                set_lemma(
                    word,
                    if lemma.is_empty() {
                        "_".to_string()
                    } else {
                        lemma
                    },
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
