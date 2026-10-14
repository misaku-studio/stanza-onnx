//! Conservative, model-free Bengali tokenization with original-source offsets.
//!
//! Bengali UD describes whitespace-delimited words and separate punctuation:
//! <https://universaldependencies.org/bn/>. Unicode 18 extended grapheme clusters
//! come from the pinned UNIC fork. The word/sentence rules are authored, not a
//! trained Stanza tokenizer or full UAX #29 word/sentence implementation. Periods in
//! decimals and between ASCII letters are protected, but titles/abbreviations
//! followed by whitespace remain ambiguous. Source spelling is never normalized.
//! Fully whitespace clusters are skipped; a mixed whitespace/mark cluster stays
//! intact so neither source marks nor grapheme boundaries are lost.

use crate::types::{Sentence, Token, Word};
use crate::unicode::{
    python_alnum, python_decimal, python_whitespace, unicode_format, unicode_mark,
};
use unic_segment::GraphemeIndices;

// Published UNIC 0.9 has older data; the portable Git revision must provide 18.0.0.
const _: () = assert!(
    unic_segment::UNICODE_VERSION.major == 18
        && unic_segment::UNICODE_VERSION.minor == 0
        && unic_segment::UNICODE_VERSION.micro == 0
);

#[derive(Debug, Default)]
pub struct BengaliTokenizer;

impl BengaliTokenizer {
    /// Tokenize the original text, with half-open Unicode code-point offsets.
    pub fn process(&self, text: &str) -> Vec<Sentence> {
        let characters: Vec<char> = text.chars().collect();
        let cluster_ends = cluster_ends(text, characters.len());
        let mut sentences = Vec::new();
        let mut tokens: Vec<Token> = Vec::new();
        let mut pending_sentence_end = false;
        let mut position = 0;

        while position < characters.len() {
            if whitespace_cluster(&characters[position..cluster_ends[position]]) {
                let start = position;
                while position < characters.len()
                    && whitespace_cluster(&characters[position..cluster_ends[position]])
                {
                    position = cluster_ends[position];
                }
                if paragraph_break(&characters[start..position]) {
                    finish_sentence(&mut tokens, &mut sentences);
                    pending_sentence_end = false;
                }
                continue;
            }

            let start = position;
            let end = token_end(&characters, &cluster_ends, start);
            let terminal = sentence_terminal(&characters, start);
            let contiguous = tokens.last().is_some_and(|token| token.end == start);
            let closes_previous =
                contiguous && (terminal || closing_punctuation(characters[start]));
            if pending_sentence_end && !closes_previous {
                finish_sentence(&mut tokens, &mut sentences);
                pending_sentence_end = false;
            }

            let surface: String = characters[start..end].iter().collect();
            tokens.push(Token {
                text: surface.clone(),
                start,
                end,
                mwt: false,
                words: vec![Word::new(surface)],
            });
            pending_sentence_end |= terminal;
            position = end;
        }

        finish_sentence(&mut tokens, &mut sentences);
        sentences
    }
}

fn finish_sentence(tokens: &mut Vec<Token>, sentences: &mut Vec<Sentence>) {
    if !tokens.is_empty() {
        sentences.push(Sentence {
            tokens: std::mem::take(tokens),
        });
    }
}

fn word_character(character: char) -> bool {
    python_alnum(character) || unicode_mark(character) || unicode_format(character)
}

fn cluster_ends(text: &str, character_count: usize) -> Vec<usize> {
    let mut ends = vec![0; character_count];
    let mut start = 0;
    for (_, cluster) in GraphemeIndices::new(text) {
        let end = start + cluster.chars().count();
        ends[start..end].fill(end);
        start = end;
    }
    ends
}

fn whitespace_cluster(cluster: &[char]) -> bool {
    cluster.iter().copied().all(python_whitespace)
}

fn word_cluster(cluster: &[char]) -> bool {
    // Numeric keycaps are symbols even though their first code point is a digit.
    let keycap = matches!(cluster[0], '0'..='9' | '#' | '*') && cluster.contains(&'\u{20e3}');
    !keycap && cluster.iter().copied().all(word_character)
}

fn token_end(characters: &[char], cluster_ends: &[usize], start: usize) -> usize {
    let mut end = cluster_ends[start];
    if word_cluster(&characters[start..end]) {
        while end < characters.len() {
            let next_end = cluster_ends[end];
            if word_cluster(&characters[end..next_end]) || numeric_separator(characters, end) {
                end = next_end;
            } else {
                break;
            }
        }
        return end;
    }

    end
}

fn numeric_separator(characters: &[char], index: usize) -> bool {
    matches!(characters[index], '.' | ',')
        && index > 0
        && index + 1 < characters.len()
        && python_decimal(characters[index - 1])
        && python_decimal(characters[index + 1])
}

fn sentence_terminal(characters: &[char], index: usize) -> bool {
    match characters[index] {
        '।' | '॥' | '?' | '!' => true,
        '.' => {
            let between_ascii_letters = index > 0
                && index + 1 < characters.len()
                && characters[index - 1].is_ascii_alphabetic()
                && characters[index + 1].is_ascii_alphabetic();
            !numeric_separator(characters, index) && !between_ascii_letters
        }
        _ => false,
    }
}

fn closing_punctuation(character: char) -> bool {
    matches!(
        character,
        ')' | ']'
            | '}'
            | '"'
            | '\''
            | '”'
            | '’'
            | '»'
            | '›'
            | '」'
            | '』'
            | '）'
            | '］'
            | '｝'
            | '〉'
            | '》'
            | '】'
            | '〕'
            | '〗'
            | '〙'
            | '〛'
    )
}

fn paragraph_break(whitespace: &[char]) -> bool {
    let mut line_breaks = 0;
    for (index, &character) in whitespace.iter().enumerate() {
        if character == '\u{2029}' {
            return true;
        }
        if character == '\n' || (character == '\r' && whitespace.get(index + 1) != Some(&'\n')) {
            line_breaks += 1;
        }
    }
    line_breaks >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{convert_offsets, OffsetUnit};
    use unic_segment::Graphemes;
    struct Fixture<'a> {
        id: &'a str,
        text: &'a str,
        expected_tokens: &'a [&'a [&'a str]],
    }

    fn assert_preserved(text: &str, sentences: &[Sentence]) {
        let characters: Vec<char> = text.chars().collect();
        let mut covered = vec![false; characters.len()];
        let mut expected_coverage = vec![false; characters.len()];
        let mut boundaries = vec![false; characters.len() + 1];
        boundaries[0] = true;
        let mut start = 0;
        for cluster in Graphemes::new(text) {
            let end = start + cluster.chars().count();
            boundaries[end] = true;
            if !cluster.chars().all(python_whitespace) {
                expected_coverage[start..end].fill(true);
            }
            start = end;
        }
        for sentence in sentences {
            assert!(!sentence.tokens.is_empty());
            for token in &sentence.tokens {
                assert!(token.start < token.end);
                assert!(boundaries[token.start]);
                assert!(boundaries[token.end]);
                assert_eq!(
                    token.text,
                    characters[token.start..token.end]
                        .iter()
                        .collect::<String>()
                );
                for present in &mut covered[token.start..token.end] {
                    assert!(!*present, "Source code point occurs in two tokens");
                    *present = true;
                }
                assert!(!token.mwt);
                assert_eq!(token.words, vec![Word::new(&token.text)]);
            }
        }
        assert_eq!(covered, expected_coverage);
    }

    #[test]
    fn authored_boundaries_preserve_every_source_code_point() {
        // Embedded fixtures keep the published Rust crate's tests self-contained.
        // The repository fixture file provides additional CLI/adapter gold cases.
        let fixtures = [
            Fixture {
                id: "reading",
                text: "আমি বাংলা শিখছি। তুমি কেমন আছ?",
                expected_tokens: &[&["আমি", "বাংলা", "শিখছি", "।"], &["তুমি", "কেমন", "আছ", "?"]],
            },
            Fixture {
                id: "quotes",
                text: "সে বলল, “আমি আসছি।” তারপর গেল।",
                expected_tokens: &[
                    &["সে", "বলল", ",", "“", "আমি", "আসছি", "।", "”"],
                    &["তারপর", "গেল", "।"],
                ],
            },
            Fixture {
                id: "numbers",
                text: "দাম ১২৩.৪৫ টাকা, 1,234.50 টাকা।",
                expected_tokens: &[&["দাম", "১২৩.৪৫", "টাকা", ",", "1,234.50", "টাকা", "।"]],
            },
            Fixture {
                id: "source_spelling",
                text: "ক্‍ষ ক্‌ষ ক্ষ কো কো Café।",
                expected_tokens: &[&["ক্‍ষ", "ক্‌ষ", "ক্ষ", "কো", "কো", "Café", "।"]],
            },
            Fixture {
                id: "emoji",
                text: "বাংলা😊 👍🏽 👨‍👩‍👧‍👦 🇧🇩🇮🇹 🇦🇧🇨 1️⃣2️⃣ #️⃣ *️⃣।",
                expected_tokens: &[&[
                    "বাংলা",
                    "😊",
                    "👍🏽",
                    "👨‍👩‍👧‍👦",
                    "🇧🇩",
                    "🇮🇹",
                    "🇦🇧",
                    "🇨",
                    "1️⃣",
                    "2️⃣",
                    "#️⃣",
                    "*️⃣",
                    "।",
                ]],
            },
            Fixture {
                id: "paragraphs",
                text: "আমি\r\nবাংলা\nশিখছি\r\n \r\nEnglish\u{2029}শেষ",
                expected_tokens: &[&["আমি", "বাংলা", "শিখছি"], &["English"], &["শেষ"]],
            },
            Fixture {
                id: "period_and_closing_rules",
                text: "example.com e.g ঠিক?! (শেষ.)",
                expected_tokens: &[
                    &["example", ".", "com", "e", ".", "g", "ঠিক", "?", "!"],
                    &["(", "শেষ", ".", ")"],
                ],
            },
            Fixture {
                id: "attached_marks",
                text: "1️⃣́ আমি?́ তুমি।",
                expected_tokens: &[&["1️⃣́", "আমি", "?́"], &["তুমি", "।"]],
            },
            Fixture {
                id: "empty",
                text: "",
                expected_tokens: &[],
            },
            Fixture {
                id: "whitespace",
                text: " \t\r\n  ",
                expected_tokens: &[],
            },
        ];
        for fixture in fixtures {
            let sentences = BengaliTokenizer.process(fixture.text);
            let tokens: Vec<Vec<&str>> = sentences
                .iter()
                .map(|sentence| {
                    sentence
                        .tokens
                        .iter()
                        .map(|token| token.text.as_str())
                        .collect()
                })
                .collect();
            let expected: Vec<Vec<&str>> = fixture
                .expected_tokens
                .iter()
                .map(|sentence| sentence.to_vec())
                .collect();
            assert_eq!(tokens, expected, "{}", fixture.id);
            assert_preserved(fixture.text, &sentences);

            for unit in [OffsetUnit::Utf16, OffsetUnit::Utf8] {
                let mut converted = sentences.clone();
                convert_offsets(fixture.text, &mut converted, unit).unwrap();
                for (source, target) in sentences
                    .iter()
                    .flat_map(|sentence| &sentence.tokens)
                    .zip(converted.iter().flat_map(|sentence| &sentence.tokens))
                {
                    let offset = |index| {
                        fixture
                            .text
                            .chars()
                            .take(index)
                            .map(|character| match unit {
                                OffsetUnit::Utf16 => character.len_utf16(),
                                OffsetUnit::Utf8 => character.len_utf8(),
                                OffsetUnit::CodePoint => unreachable!(),
                            })
                            .sum::<usize>()
                    };
                    assert_eq!(target.start, offset(source.start));
                    assert_eq!(target.end, offset(source.end));
                }
            }
        }
    }

    #[test]
    fn long_input_preserves_boundaries_and_finishes_the_final_word() {
        let text = format!("{}শেষ", "আমি বাংলা শিখছি। ".repeat(1_000));
        let sentences = BengaliTokenizer.process(&text);
        assert_eq!(sentences.len(), 1_001);
        assert_eq!(sentences.last().unwrap().tokens[0].text, "শেষ");
        assert_preserved(&text, &sentences);
    }

    #[test]
    fn unicode_18_clusters_keep_conjuncts_new_marks_and_keycaps_whole() {
        assert_eq!(
            unic_segment::UNICODE_VERSION.to_string(),
            crate::UNICODE_VERSION
        );
        assert_eq!(Graphemes::new("ক্ষ").collect::<Vec<_>>(), vec!["ক্ষ"]);
        assert_eq!(
            Graphemes::new("a\u{11df0}").collect::<Vec<_>>(),
            vec!["a\u{11df0}"]
        );
        let text = "ক্ষ ক\u{11df0} a\u{11df0} a1️⃣b";
        let sentences = BengaliTokenizer.process(text);
        let tokens: Vec<_> = sentences[0]
            .tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect();
        assert_eq!(
            tokens,
            vec!["ক্ষ", "ক\u{11df0}", "a\u{11df0}", "a", "1️⃣", "b"]
        );
        assert_preserved(text, &sentences);
    }

    #[test]
    fn mixed_whitespace_cluster_preserves_its_space_and_mark() {
        let text = "আমি \u{301}খ।";
        let sentences = BengaliTokenizer.process(text);
        let tokens: Vec<_> = sentences[0]
            .tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect();
        assert_eq!(tokens, vec!["আমি", " \u{301}", "খ", "।"]);
        assert_preserved(text, &sentences);
    }
}
