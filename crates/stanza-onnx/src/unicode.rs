//! Default Unicode 18 casing and Python-style character properties.
//!
//! Generated directly from checksum-pinned Unicode Character Database files.
//! Runtime behavior is independent of Rust's and the host Python's Unicode version.
//! See LICENSE-UNICODE and scripts/generate_unicode.py for source provenance.
use serde::Deserialize;
use std::{collections::HashMap, sync::OnceLock};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Tables {
    unicode_version: String,
    uppercase: Vec<[u32; 2]>,
    lowercase: Vec<[u32; 2]>,
    titlecase: Vec<[u32; 2]>,
    decimal: Vec<[u32; 2]>,
    digit: Vec<[u32; 2]>,
    numeric: Vec<[u32; 2]>,
    alpha: Vec<[u32; 2]>,
    alnum: Vec<[u32; 2]>,
    whitespace: Vec<[u32; 2]>,
    cased: Vec<[u32; 2]>,
    case_ignorable: Vec<[u32; 2]>,
    lower: HashMap<u32, String>,
    upper: HashMap<u32, String>,
}
fn tables() -> &'static Tables {
    static DATA: OnceLock<Tables> = OnceLock::new();
    DATA.get_or_init(|| {
        let tables: Tables = serde_json::from_str(include_str!("data/unicode18.json"))
            .expect("Generated Unicode tables must be valid");
        assert_eq!(
            tables.unicode_version,
            crate::UNICODE_VERSION,
            "Generated Unicode version must match the runtime"
        );
        tables
    })
}
fn in_ranges(character: char, ranges: &[[u32; 2]]) -> bool {
    let code = character as u32;
    let index = ranges.partition_point(|range| range[0] <= code);
    index > 0 && code <= ranges[index - 1][1]
}
pub fn python_whitespace(c: char) -> bool {
    in_ranges(c, &tables().whitespace)
}
pub fn python_uppercase(c: char) -> bool {
    in_ranges(c, &tables().uppercase)
}
pub fn python_lowercase(c: char) -> bool {
    in_ranges(c, &tables().lowercase)
}
pub fn python_titlecase(c: char) -> bool {
    in_ranges(c, &tables().titlecase)
}
pub fn python_decimal(c: char) -> bool {
    in_ranges(c, &tables().decimal)
}
pub fn python_digit(c: char) -> bool {
    in_ranges(c, &tables().digit)
}
pub fn python_numeric(c: char) -> bool {
    in_ranges(c, &tables().numeric)
}
pub fn python_alpha(c: char) -> bool {
    in_ranges(c, &tables().alpha)
}
pub fn python_alnum(c: char) -> bool {
    in_ranges(c, &tables().alnum)
}
pub fn python_cased(c: char) -> bool {
    in_ranges(c, &tables().cased)
}
pub fn python_isupper(text: &str) -> bool {
    let mut has_uppercase = false;
    for c in text.chars() {
        if python_lowercase(c) || python_titlecase(c) {
            return false;
        }
        has_uppercase |= python_uppercase(c);
    }
    has_uppercase
}
pub fn python_islower(text: &str) -> bool {
    let mut has_lowercase = false;
    for c in text.chars() {
        if python_uppercase(c) || python_titlecase(c) {
            return false;
        }
        has_lowercase |= python_lowercase(c);
    }
    has_lowercase
}
fn ignorable(c: char) -> bool {
    in_ranges(c, &tables().case_ignorable)
}

/// A regex class derived from the same whitespace data as classification.
pub(crate) fn whitespace_regex_class() -> String {
    let mut result = String::from("[");
    for &[start, end] in &tables().whitespace {
        result.push_str(&format!(r"\u{{{start:x}}}"));
        if start != end {
            result.push('-');
            result.push_str(&format!(r"\u{{{end:x}}}"));
        }
    }
    result.push(']');
    result
}

pub fn python_lower(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());
    for (index, &character) in chars.iter().enumerate() {
        if character == 'Σ' {
            let preceded = chars[..index]
                .iter()
                .rev()
                .copied()
                .find(|&c| !ignorable(c))
                .is_some_and(python_cased);
            let followed = chars[index + 1..]
                .iter()
                .copied()
                .find(|&c| !ignorable(c))
                .is_some_and(python_cased);
            result.push(if preceded && !followed { 'ς' } else { 'σ' });
        } else if let Some(value) = tables().lower.get(&(character as u32)) {
            result.push_str(value);
        } else {
            result.push(character);
        }
    }
    result
}
pub fn python_upper(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for character in text.chars() {
        if let Some(value) = tables().upper.get(&(character as u32)) {
            result.push_str(value);
        } else {
            result.push(character);
        }
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_casing_preserves_expanding_and_locale_independent_forms() {
        assert_eq!(
            python_lower("ΟΣ ΟΣΑ Σ İ I ı AΣ\u{301}"),
            "ος οσα σ i\u{307} i ı aς\u{301}"
        );
        assert_eq!(python_upper("straße ﬃ i ı"), "STRASSE FFI I I");
        assert!(!python_uppercase('ǅ'));
        assert!(python_titlecase('ǅ'));
        assert!(python_cased('ǅ'));
    }

    #[test]
    fn string_case_predicates_use_properties_not_mapping_equality() {
        assert!(!python_isupper(""));
        assert!(!python_islower("123"));
        assert!(python_isupper("A1!"));
        assert!(python_islower("a1!"));
        assert!(!python_isupper("ǅ"));
        assert!(!python_islower("ǅ"));
        // Modifier small h has no uppercase conversion, but remains lowercase.
        assert_eq!(python_upper("ʰ"), "ʰ");
        assert!(python_islower("ʰ"));
        assert!(!python_isupper("ʰA"));
        // Unicode 17 modifier capital S is lowercase despite its character name.
        assert_eq!(python_lower("\u{a7f1}"), "\u{a7f1}");
        assert!(python_islower("\u{a7f1}"));
        assert!(!python_isupper("\u{a7f1}"));
        // Squared capital B is uppercase but has no lowercase conversion.
        assert_eq!(python_lower("\u{1f131}"), "\u{1f131}");
        assert!(python_isupper("\u{1f131}"));
        assert!(!python_islower("\u{1f131}b"));
        // Unicode 18 modifier Armenian small eh is lowercase and ignorable.
        assert!(python_islower("\u{558}"));
        assert!(!python_isupper("\u{558}A"));
    }

    #[test]
    fn characters_added_in_unicode_16_17_and_18_use_the_latest_tables() {
        // Unicode 16 Cyrillic Tje and Garay digits.
        assert_eq!(python_lower("\u{1c89}"), "\u{1c8a}");
        assert!(python_uppercase('\u{1c89}'));
        assert!(python_lowercase('\u{1c8a}'));
        assert!(python_decimal('\u{10d40}'));
        // Unicode 17 Beria Erfe casing and Tolong Siki digits.
        assert_eq!(python_lower("\u{16ea0}"), "\u{16ebb}");
        assert_eq!(python_upper("\u{16ebb}"), "\u{16ea0}");
        assert!(python_decimal('\u{11de0}'));
        // Unicode 18 closed omega, new CJK/Small Seal letters and numeric signs.
        assert_eq!(python_lower("\u{a7dd}"), "\u{277}");
        assert_eq!(python_upper("\u{277}"), "\u{a7dd}");
        assert!(python_uppercase('\u{a7dd}'));
        assert!(python_alpha('\u{3d000}'));
        assert!(!python_alpha('\u{3fc40}'));
        assert!(python_numeric('\u{1246f}'));
        assert!(!python_digit('\u{1246f}'));
    }

    #[test]
    fn final_sigma_uses_new_ignorable_marks_and_cased_neighbors() {
        assert_eq!(python_lower("AΣ\u{11df0}"), "aς\u{11df0}");
        assert_eq!(python_lower("AΣ\u{11df0}\u{a7dd}"), "aσ\u{11df0}\u{277}");
        assert_eq!(python_lower("AΣ\u{1d127}"), "aς\u{1d127}");
        // New U+0558 is both cased and case-ignorable: skip it for context.
        assert!(python_cased('\u{558}'));
        assert!(ignorable('\u{558}'));
        assert_eq!(python_lower("\u{558}Σ"), "\u{558}σ");
        assert_eq!(python_lower("AΣ\u{558}"), "aς\u{558}");
    }

    #[test]
    fn python_style_properties_distinguish_numbers_letters_and_whitespace() {
        assert!(python_decimal('١'));
        assert!(python_digit('²'));
        assert!(!python_decimal('²'));
        assert!(python_numeric('一'));
        assert!(python_numeric('Ⅸ'));
        assert!(!python_alpha('Ⅸ'));
        assert!(python_alnum('Ⅸ'));
        assert!(python_lowercase('\u{345}'));
        assert!(!python_alpha('\u{345}'));
        assert!(!python_alnum('\u{345}'));
        assert!(python_whitespace('\u{1c}'));
        assert!(python_whitespace('\u{85}'));
        assert!(!python_whitespace('\u{feff}'));
        let regex = regex::Regex::new(&format!("^{}$", whitespace_regex_class())).unwrap();
        for code in 0..=0x7f {
            if let Some(c) = char::from_u32(code) {
                assert_eq!(regex.is_match(&c.to_string()), python_whitespace(c));
            }
        }
        for c in [
            '\u{85}', '\u{a0}', '\u{1680}', '\u{2007}', '\u{2028}', '\u{2029}', '\u{202f}',
            '\u{205f}', '\u{3000}',
        ] {
            assert!(regex.is_match(&c.to_string()));
        }
        assert!(!regex.is_match("\u{feff}"));
        assert!(!regex.is_match("\u{11df0}"));
    }
}
