//! Stanza 1.11.0 tokenizer preprocessing, inference and output assembly.
//!
//! Adapted from Stanford Stanza, Copyright 2019 The Board of Trustees of
//! The Leland Stanford Junior University. Licensed under Apache-2.0;
//! see LICENSE and NOTICE. Upstream:
//! https://github.com/stanfordnlp/stanza/tree/v1.11.0/stanza/models/tokenization

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use fancy_regex::Regex as MaskRegex;
use regex::Regex;
use serde::Deserialize;

use crate::engine::{ModelSession, SessionOptions, TensorData};
use crate::types::{Sentence, Token, Word};
use crate::unicode::{
    python_decimal, python_lower, python_uppercase, python_whitespace, whitespace_regex_class,
};

const MAX_SEQUENCE: usize = 1000;
fn paragraph_break_pattern() -> String {
    // Use the same pinned Unicode data as Python-style whitespace classification.
    format!(r"\n{}*\n", whitespace_regex_class())
}

#[derive(Debug, Deserialize)]
struct Config {
    shorthand: String,
    #[serde(default)]
    skip_newline: bool,
    #[serde(default)]
    use_dictionary: bool,
    feat_funcs: Vec<String>,
    feat_dim: usize,
    #[serde(default)]
    num_dict_feat: usize,
}

#[derive(Debug, Deserialize)]
struct Vocab {
    #[serde(rename = "_unit2id")]
    unit_to_id: HashMap<String, i64>,
    // Stanza 1.11 constructs loaded vocabularies with an empty language before
    // restoring their state. This flag defaults to false and is not inferred
    // again from the restored CJK language name.
    #[serde(default)]
    lang_replaces_spaces: bool,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    config: Config,
    vocab: Vocab,
    mask_regex: String,
    #[serde(default)]
    lexicon: Vec<String>,
}

#[derive(Default)]
struct Dictionary {
    words: HashSet<String>,
    prefixes: HashSet<String>,
    suffixes: HashSet<String>,
}

impl Dictionary {
    fn new(words: Vec<String>) -> Self {
        let words: HashSet<_> = words.into_iter().collect();
        let mut prefixes = HashSet::new();
        let mut suffixes = HashSet::new();
        for word in &words {
            let chars: Vec<_> = word.chars().collect();
            for len in 1..chars.len() {
                prefixes.insert(chars[..len].iter().collect());
                suffixes.insert(chars[chars.len() - len..].iter().collect());
            }
        }
        Self {
            words,
            prefixes,
            suffixes,
        }
    }

    fn features(&self, raw: &[char], index: usize, count: usize) -> Vec<f32> {
        let mut features = vec![0.0; count * 2];
        // Stanza does not lowercase the starting character, only neighbours.
        let mut forward = raw[index].to_string();
        let mut backward = forward.clone();
        let mut has_prefix = true;
        let mut has_suffix = true;
        for window in 1..=count {
            if index + window < raw.len() && has_prefix {
                forward.push_str(&python_lower(&raw[index + window].to_string()));
                features[window - 1] = u8::from(self.words.contains(&forward)) as f32;
                has_prefix = self.prefixes.contains(&forward);
            }
            if window <= index && has_suffix {
                backward = python_lower(&raw[index - window].to_string()) + &backward;
                features[count + window - 1] = u8::from(self.words.contains(&backward)) as f32;
                has_suffix = self.suffixes.contains(&backward);
            }
            if !has_prefix && !has_suffix {
                break;
            }
        }
        features
    }
}

pub struct Tokenizer {
    session: ModelSession,
    meta: Metadata,
    dictionary: Dictionary,
    paragraph_break: Regex,
    mask: MaskRegex,
}

impl Tokenizer {
    pub fn load(dir: &Path, options: &SessionOptions) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let mut meta: Metadata = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?;
        validate_features(&meta.config)?;
        let mask = MaskRegex::new(&meta.mask_regex).context("compiling tokenizer mask regex")?;
        let dictionary = Dictionary::new(std::mem::take(&mut meta.lexicon));
        let paragraph_break =
            Regex::new(&paragraph_break_pattern()).expect("generated paragraph regex");
        let session = ModelSession::load(&dir.join("tokenizer.onnx"), options)?;
        Ok(Self {
            session,
            meta,
            dictionary,
            paragraph_break,
            mask,
        })
    }

    fn features(&self, raw: &[char]) -> Vec<f32> {
        let config = &self.meta.config;
        let mut features = Vec::with_capacity(raw.len() * config.feat_dim);
        for (i, &c) in raw.iter().enumerate() {
            for name in &config.feat_funcs {
                let value = match name.as_str() {
                    "end_of_para" | "start_of_para" => continue,
                    "space_before" => c == ' ',
                    "capitalized" => python_uppercase(c),
                    "numeric" => python_decimal(c),
                    _ => unreachable!("validated feature names"),
                };
                features.push(u8::from(value) as f32);
            }
            // Stanza puts positional features after the regular features,
            // regardless of their position in feat_funcs.
            if config.feat_funcs.iter().any(|name| name == "end_of_para") {
                features.push(u8::from(i == raw.len() - 1) as f32);
            }
            if config.feat_funcs.iter().any(|name| name == "start_of_para") {
                features.push(u8::from(i == 0) as f32);
            }
            if config.use_dictionary {
                features.extend(self.dictionary.features(raw, i, config.num_dict_feat));
            }
        }
        features
    }

    fn infer(&mut self, ids: &[i64], features: &[f32]) -> Result<Vec<usize>> {
        let n = ids.len();
        let output = self.session.run(vec![
            ("units", TensorData::i64(vec![1, n], ids.to_vec())),
            (
                "features",
                TensorData::f32(vec![1, n, self.meta.config.feat_dim], features.to_vec()),
            ),
            ("lengths", TensorData::i64(vec![1], vec![n as i64])),
        ])?;
        let scores = output
            .get("log_probs")
            .context("tokenizer missing log_probs output")?;
        if scores.shape.len() != 3 || scores.shape[0] != 1 || scores.shape[1] != n {
            bail!("invalid tokenizer output shape {:?}", scores.shape);
        }
        let classes = scores.shape[2];
        if classes < 3 || scores.data.len() != n * classes {
            bail!("invalid tokenizer output size");
        }
        Ok(scores
            .data
            .chunks_exact(classes)
            .map(|row| {
                let mut best = 0;
                for c in 1..classes {
                    if row[c] > row[best] {
                        best = c;
                    }
                }
                best
            })
            .collect())
    }

    fn predict(&mut self, raw: &[char]) -> Result<Vec<usize>> {
        let len = raw.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut ids = Vec::with_capacity(len + 1);
        for &c in raw {
            ids.push(*self.meta.vocab.unit_to_id.get(&c.to_string()).unwrap_or(&1));
        }
        ids.push(0); // Stanza SortedDataset adds one padding unit.
        let mut features = self.features(raw);
        features.resize((len + 1) * self.meta.config.feat_dim, 0.0);
        let mut pred;
        if len < MAX_SEQUENCE {
            pred = self.infer(&ids, &features)?;
            pred.truncate(len);
        } else {
            pred = Vec::with_capacity(len);
            let mut offset = 0;
            while offset < len {
                let size = MAX_SEQUENCE.min(len - offset);
                let fd = self.meta.config.feat_dim;
                let next = self.infer(
                    &ids[offset..offset + size],
                    &features[offset * fd..(offset + size) * fd],
                )?;
                let advance = if offset < len.saturating_sub(MAX_SEQUENCE) {
                    next.iter()
                        .rposition(|&p| p == 2 || p == 4)
                        .map_or(size, |last| last + 1)
                } else {
                    size
                };
                pred.extend_from_slice(&next[..advance]);
                offset += advance;
            }
        }
        pred[len - 1] = if pred[len - 1] < 2 {
            2
        } else if pred[len - 1] > 2 {
            4
        } else {
            2
        };
        mask_predictions(&self.mask, raw, &mut pred)?;
        Ok(pred)
    }

    /// Return offsets in Unicode code points, preserving original source spans.
    /// The pipeline may convert them to another caller-selected offset unit.
    pub fn process(&mut self, text: &str) -> Result<Vec<Sentence>> {
        let normalized: String = text
            .chars()
            .map(|c| if python_whitespace(c) { ' ' } else { c })
            .collect();
        let boundaries = codepoint_boundaries(&normalized);
        let mut char_offset = 0; // Byte offset into normalized source, never the raw model units.
        let mut sentences = Vec::new();
        // Copy paragraph slices before mutably borrowing the session in predict.
        let paragraphs: Vec<_> = self
            .paragraph_break
            .split(text)
            .map(|part| part.trim_end_matches(python_whitespace))
            .filter(|part| !part.is_empty())
            .map(str::to_owned)
            .collect();
        for paragraph in paragraphs {
            let raw = normalize_paragraph(&paragraph, self.meta.config.skip_newline);
            if raw.is_empty() {
                continue;
            }
            let pred = self.predict(&raw)?;
            let mut current = String::new();
            let mut tokens = Vec::new();
            for (i, &c) in raw.iter().enumerate() {
                current.push(c);
                let p = if self.meta.config.shorthand == "la_ittb" && (c == ':' || c == ';') {
                    2
                } else {
                    pred[i]
                };
                if p == 0 {
                    continue;
                }
                let mut token_text = normalize_token(&current, &self.meta.vocab);
                if token_text.is_empty() {
                    current.clear();
                    continue;
                }
                let (start, end) = align_token(
                    &normalized,
                    &current,
                    &mut char_offset,
                    self.meta.config.skip_newline,
                    &boundaries,
                )?;
                if token_text.chars().count() > MAX_SEQUENCE {
                    token_text = "<UNK>".to_owned();
                }
                let word = Word::new(token_text.clone());
                tokens.push(Token {
                    text: token_text,
                    start,
                    end,
                    mwt: p == 3 || p == 4,
                    words: vec![word],
                });
                current.clear();
                if p == 2 || p == 4 {
                    sentences.push(Sentence {
                        tokens: std::mem::take(&mut tokens),
                    });
                }
            }
            if !current.is_empty() {
                bail!("tokenizer left unconsumed text");
            }
            if !tokens.is_empty() {
                sentences.push(Sentence { tokens });
            }
        }
        Ok(sentences)
    }
}

fn validate_features(config: &Config) -> Result<()> {
    let mut regular = 0;
    for name in &config.feat_funcs {
        match name.as_str() {
            "space_before" | "capitalized" | "numeric" => regular += 1,
            "end_of_para" | "start_of_para" => (),
            _ => bail!("unsupported tokenizer feature {name}"),
        }
    }
    let width = regular
        + usize::from(config.feat_funcs.iter().any(|name| name == "end_of_para"))
        + usize::from(config.feat_funcs.iter().any(|name| name == "start_of_para"))
        + if config.use_dictionary {
            config.num_dict_feat * 2
        } else {
            0
        };
    if width != config.feat_dim {
        bail!(
            "tokenizer feature width {width} differs from checkpoint width {}",
            config.feat_dim
        );
    }
    Ok(())
}

fn normalize_paragraph(paragraph: &str, skip_newline: bool) -> Vec<char> {
    let mut raw = Vec::new();
    for c in paragraph.chars() {
        if skip_newline && c == '\n' {
            continue;
        }
        let unit = if python_whitespace(c) { ' ' } else { c };
        if unit != ' ' || raw.last() != Some(&' ') {
            raw.push(unit);
        }
    }
    raw
}

fn normalize_token(current: &str, vocab: &Vocab) -> String {
    current
        .trim_start_matches(python_whitespace)
        .chars()
        .map(|c| if python_whitespace(c) { ' ' } else { c })
        .filter(|&c| !vocab.lang_replaces_spaces || c != ' ')
        .collect()
}

fn codepoint_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries: Vec<_> = text.char_indices().map(|(index, _)| index).collect();
    boundaries.push(text.len());
    boundaries
}

fn mask_predictions(mask: &MaskRegex, raw: &[char], pred: &mut [usize]) -> Result<()> {
    let joined: String = raw.iter().collect();
    let boundaries = codepoint_boundaries(&joined);
    for result in mask.find_iter(&joined) {
        let found = result.context("matching tokenizer mask regex")?;
        let begin = boundaries
            .binary_search(&found.start())
            .map_err(|_| anyhow!("mask start is not a character boundary"))?;
        let end = boundaries
            .binary_search(&found.end())
            .map_err(|_| anyhow!("mask end is not a character boundary"))?;
        if begin == end {
            continue;
        }
        pred[begin..end - 1].fill(0);
        if pred[end - 1] == 0 {
            pred[end - 1] = 1;
        }
    }
    Ok(())
}

/// Equivalent to Stanza SPACE_SPLIT_RE.split: retain runs of leading spaces
/// with the following non-space text, plus any trailing run of spaces.
fn alignment_parts(current: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut seen_nonspace = false;
    for (index, c) in current.char_indices() {
        if c == ' ' {
            if seen_nonspace {
                result.push(&current[start..index]);
                start = index;
                seen_nonspace = false;
            }
        } else {
            seen_nonspace = true;
        }
    }
    if start < current.len() {
        result.push(&current[start..]);
    }
    result
}

fn align_token(
    normalized: &str,
    current: &str,
    offset: &mut usize,
    skip_newline: bool,
    boundaries: &[usize],
) -> Result<(usize, usize)> {
    let mut token_start = None;
    for part in alignment_parts(current) {
        let (found, part_len) = if skip_newline {
            // normalized source has all Python whitespace replaced by ASCII
            // spaces, making this equivalent to Python's r'\s*'.join(...).
            let pattern = part
                .chars()
                .map(|c| regex::escape(&c.to_string()))
                .collect::<Vec<_>>()
                .join(" *");
            let matcher = Regex::new(&pattern).context("compiling skip-newline alignment regex")?;
            let found = matcher
                .find(&normalized[*offset..])
                .with_context(|| format!("cannot align skip-newline token {part:?} at {offset}"))?;
            (*offset + found.start(), found.end() - found.start())
        } else {
            let found = normalized[*offset..]
                .find(part)
                .with_context(|| format!("cannot align token {part:?} at {offset}"))?;
            (*offset + found, part.len())
        };
        if token_start.is_none() {
            let matched = &normalized[found..found + part_len];
            token_start =
                Some(found + part_len - matched.trim_start_matches(python_whitespace).len());
        }
        *offset = found + part_len;
    }
    let start = boundaries
        .binary_search(&token_start.context("empty token alignment")?)
        .map_err(|_| anyhow!("token start is not a character boundary"))?;
    let end = boundaries
        .binary_search(offset)
        .map_err(|_| anyhow!("token end is not a character boundary"))?;
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_cjk_vocabulary_preserves_spaces_unless_explicitly_configured() {
        let loaded: Vocab = serde_json::from_value(serde_json::json!({
            "lang": "zh-hans", "_unit2id": {}
        }))
        .unwrap();
        let constructed: Vocab = serde_json::from_value(serde_json::json!({
            "lang": "zh-hans", "_unit2id": {}, "lang_replaces_spaces": true
        }))
        .unwrap();
        assert!(!loaded.lang_replaces_spaces);
        assert_eq!(normalize_token(" \tI İ 甲 乙", &loaded), "I İ 甲 乙");
        assert_eq!(normalize_token(" \tI İ 甲 乙", &constructed), "Iİ甲乙");
        let normalized = "  I   İ";
        let mut offset = 0;
        assert_eq!(normalize_token(" I İ", &loaded), "I İ");
        assert_eq!(
            align_token(
                normalized,
                " I İ",
                &mut offset,
                false,
                &codepoint_boundaries(normalized)
            )
            .unwrap(),
            (2, 7)
        );
    }

    #[test]
    fn normalization_keeps_python_whitespace_and_skip_newline_semantics() {
        assert_eq!(
            normalize_paragraph(" \t甲\n乙\u{85} 丙\u{feff}", false),
            vec![' ', '甲', ' ', '乙', ' ', '丙', '\u{feff}']
        );
        assert_eq!(
            normalize_paragraph("甲\n乙\r\n丙", true),
            vec!['甲', '乙', ' ', '丙']
        );
        let breaks = Regex::new(&paragraph_break_pattern()).unwrap();
        assert_eq!(
            breaks.split("a\n \u{85}\n  b\n\n\n c").collect::<Vec<_>>(),
            vec!["a", "  b", " c"]
        );
    }

    #[test]
    fn alignment_preserves_source_spacing_and_codepoint_offsets() {
        let source = "  A😀   B C";
        let boundaries = codepoint_boundaries(source);
        let mut offset = 0;
        assert_eq!(
            align_token(source, " A😀 B", &mut offset, false, &boundaries).unwrap(),
            (2, 8)
        );
        assert_eq!(
            align_token(source, " C", &mut offset, false, &boundaries).unwrap(),
            (9, 10)
        );
        assert_eq!(alignment_parts(" A B  "), vec![" A", " B", "  "]);
    }

    #[test]
    fn skip_newline_alignment_can_span_whitespace_inside_cjk_tokens() {
        let source = "  甲 乙  丙 丁";
        let boundaries = codepoint_boundaries(source);
        let mut offset = 0;
        assert_eq!(
            align_token(source, " 甲乙 丙", &mut offset, true, &boundaries).unwrap(),
            (2, 8)
        );
        assert_eq!(
            align_token(source, " 丁", &mut offset, true, &boundaries).unwrap(),
            (9, 10)
        );
    }

    #[test]
    fn dictionary_features_stop_only_after_prefix_and_suffix_exhausted() {
        let dictionary = Dictionary::new(vec![
            "甲乙".into(),
            "甲乙丙".into(),
            "乙丙".into(),
            "abc".into(),
        ]);
        assert_eq!(
            dictionary.features(&['甲', '乙', '丙'], 0, 3),
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            dictionary.features(&['甲', '乙', '丙'], 2, 3),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            dictionary.features(&['A', 'B', 'C'], 0, 2),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn masks_use_codepoint_indices_and_preserve_sentence_end_labels() {
        let raw: Vec<_> = "😀 https://www.example.com".chars().collect();
        let mut pred = vec![1; raw.len()];
        *pred.last_mut().unwrap() = 4;
        let regex = MaskRegex::new(r"https?://(?:www\.|(?!www))[a-z]+\.com").unwrap();
        mask_predictions(&regex, &raw, &mut pred).unwrap();
        assert_eq!(pred[0], 1);
        assert!(pred[2..pred.len() - 1].iter().all(|&p| p == 0));
        assert_eq!(pred.last(), Some(&4));
    }
}
