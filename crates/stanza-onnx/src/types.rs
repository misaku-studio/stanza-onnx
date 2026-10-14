use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub upos: Option<String>,
    pub xpos: Option<String>,
    pub feats: Option<String>,
    pub lemma: Option<String>,
}
impl Word {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            upos: None,
            xpos: None,
            feats: None,
            lemma: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub start: usize,
    pub end: usize,
    #[serde(skip)]
    pub mwt: bool,
    pub words: Vec<Word>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentence {
    pub tokens: Vec<Token>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum OffsetUnit {
    #[default]
    #[serde(rename = "unicode-code-point")]
    CodePoint,
    #[serde(rename = "utf16")]
    Utf16,
    #[serde(rename = "utf8")]
    Utf8,
}

pub fn convert_offsets(
    text: &str,
    sentences: &mut [Sentence],
    unit: OffsetUnit,
) -> anyhow::Result<()> {
    if unit == OffsetUnit::CodePoint {
        return Ok(());
    }
    let mut offsets = vec![0];
    for character in text.chars() {
        offsets.push(
            offsets.last().unwrap()
                + if unit == OffsetUnit::Utf16 {
                    character.len_utf16()
                } else {
                    character.len_utf8()
                },
        );
    }
    for sentence in sentences {
        for token in &mut sentence.tokens {
            anyhow::ensure!(
                token.start <= token.end && token.end < offsets.len(),
                "Invalid source token offsets"
            );
            token.start = offsets[token.start];
            token.end = offsets[token.end];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offset_units_preserve_original_combining_and_astral_text() {
        let text = "😊a\u{301}";
        let original = vec![Sentence {
            tokens: vec![Token {
                text: text.into(),
                start: 0,
                end: 3,
                mwt: false,
                words: vec![Word::new(text)],
            }],
        }];
        let mut utf16 = original.clone();
        convert_offsets(text, &mut utf16, OffsetUnit::Utf16).unwrap();
        assert_eq!(utf16[0].tokens[0].end, 4);
        let mut utf8 = original;
        convert_offsets(text, &mut utf8, OffsetUnit::Utf8).unwrap();
        assert_eq!(utf8[0].tokens[0].end, 7);
    }
}
