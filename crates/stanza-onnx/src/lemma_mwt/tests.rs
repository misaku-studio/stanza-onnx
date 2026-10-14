use super::sequence::{reorder, repeat_f32, sequence_scores};
use super::*;
use crate::engine::TensorOutput;
use crate::types::Token;

fn vocab(units: &[&str]) -> Vocab {
    Vocab {
        units: units.iter().map(|s| s.to_string()).collect(),
        ids: units
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i))
            .collect(),
        lower: false,
    }
}

#[test]
fn dynamic_copy_vocab_uses_scalar_sorting_and_special_ids() {
    let base = vocab(&["<PAD>", "<UNK>", "<SOS>", "<EOS>", "a"]);
    let result = expanded(&base, &["😀𐐀é😀a".into()]);
    assert_eq!(
        result.units,
        ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "a", "é", "𐐀", "😀"]
    );
    let sources = Sources::new(&["a😀".into(), "é".into()], &result);
    assert_eq!(sources.lengths, [4, 3]);
    assert_eq!(sources.numbers, [2, 4, 7, 3, 2, 5, 3, 0]);
}

#[test]
fn dictionary_expansion_prioritizes_exact_and_restores_case() {
    let dict = HashMap::from([
        ("du".into(), "de le".into()),
        ("DU".into(), "DE le".into()),
        ("aux".into(), "à les".into()),
    ]);
    assert_eq!(dictionary_expansion("du", &dict).as_deref(), Some("de le"));
    assert_eq!(dictionary_expansion("DU", &dict).as_deref(), Some("DE le"));
    assert_eq!(dictionary_expansion("Du", &dict).as_deref(), Some("De le"));
    assert_eq!(dictionary_expansion("AUX", &dict).as_deref(), Some("À LES"));
    assert_eq!(dictionary_expansion("Aux", &dict).as_deref(), Some("À les"));
    assert_eq!(dictionary_expansion("dU", &dict), None);
    assert_eq!(dictionary_expansion("", &dict), None);
}

#[test]
fn dictionary_expansion_uses_stock_property_based_case_tests() {
    let dict = HashMap::from([
        ("ʰa".into(), "de le".into()),
        ("a\u{1f131}b".into(), "de le".into()),
        ("ǆu".into(), "de le".into()),
        ("\u{558}a".into(), "de le".into()),
        ("du".into(), "\u{1c}de le\u{1f}".into()),
    ]);
    assert_eq!(dictionary_expansion("ʰA", &dict), None);
    assert_eq!(dictionary_expansion("A\u{1f131}b", &dict), None);
    assert_eq!(dictionary_expansion("ǅu", &dict), None);
    assert_eq!(dictionary_expansion("\u{558}A", &dict), None);
    assert_eq!(dictionary_expansion("ʰa", &dict).as_deref(), Some("de le"));
    assert_eq!(
        dictionary_expansion("DU", &dict).as_deref(),
        Some("\u{1c}DE LE\u{1f}")
    );
    assert_eq!("\u{1c}de le\u{1f}".trim_matches(python_whitespace), "de le");
}

#[test]
fn copy_distribution_reproduces_stock_beam_zero_only_copying() {
    let args = Args {
        vocab_size: 4,
        copy: true,
        ..Args::default()
    };
    let inputs = Sources::new(
        &["x".into()],
        &expanded(&vocab(&["<PAD>", "<UNK>", "<SOS>", "<EOS>"]), &["x".into()]),
    );
    let out = HashMap::from([
        (
            "logprobs".into(),
            TensorOutput {
                shape: vec![2, 4],
                data: vec![-20.0; 8],
            },
        ),
        (
            "copygate".into(),
            TensorOutput {
                shape: vec![2],
                data: vec![20.0; 2],
            },
        ),
        (
            "logattn".into(),
            TensorOutput {
                shape: vec![2, 3],
                data: vec![-20.0, 0.0, -20.0, -20.0, 0.0, -20.0],
            },
        ),
    ]);
    let (scores, size) = sequence_scores(&args, &out, &inputs, 1, 2, true).unwrap();
    assert_eq!(size, 5);
    assert!(scores[4] > -0.001);
    assert!(scores[9] < -39.0);
    assert_eq!(scores[1], f32::NEG_INFINITY);
    assert_eq!(scores[6], f32::NEG_INFINITY);
}

#[test]
fn recurrent_state_reordering_preserves_beam_batch_layout() {
    let state = TensorOutput {
        shape: vec![2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let repeated = repeat_f32(state, 2);
    assert_eq!(repeated.shape, [4, 2]);
    let reordered = reorder(repeated, &[3, 0, 2, 1]).unwrap();
    assert_eq!(reordered.data, [3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn underscore_lemma_is_null_except_for_literal_underscore() {
    let mut word = Word::new("word");
    set_lemma(&mut word, "_".into());
    assert_eq!(word.lemma, None);
    let mut underscore = Word::new("_");
    set_lemma(&mut underscore, "_".into());
    assert_eq!(underscore.lemma.as_deref(), Some("_"));
}

fn sentences(texts: &[&str], mwt: bool) -> Vec<Sentence> {
    vec![Sentence {
        tokens: texts
            .iter()
            .enumerate()
            .map(|(i, text)| Token {
                text: (*text).into(),
                start: i * 5,
                end: i * 5 + text.chars().count(),
                mwt,
                words: vec![Word::new(*text)],
            })
            .collect(),
    }]
}

#[test]
fn dictionary_mwt_preserves_tokens_offsets_and_single_piece_source() {
    let meta = MwtMeta {
        kind: "dictionary".into(),
        args: Args::default(),
        vocab: Vocab::default(),
        dict: HashMap::from([
            ("aux".into(), "à les".into()),
            ("alone".into(), "changed".into()),
        ]),
        decode_beam_size: None,
    };
    let mut mwt = Mwt {
        meta,
        session: None,
        sequence: None,
    };
    let mut data = sentences(&["Aux", "alone", "unknown"], true);
    mwt.process(&mut data).unwrap();
    let tokens = &data[0].tokens;
    assert_eq!(tokens[0].text, "Aux");
    assert_eq!((tokens[0].start, tokens[0].end), (0, 3));
    assert_eq!(
        tokens[0]
            .words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>(),
        ["À", "les"]
    );
    assert_eq!(tokens[1].words[0].text, "alone");
    assert_eq!(tokens[2].words[0].text, "unknown");
}

#[test]
fn lemma_dictionary_composite_priority_caseless_and_empty_placeholder() {
    let meta = LemmaMeta {
        kind: "dictionary".into(),
        args: Args {
            caseless: true,
            ..Args::default()
        },
        char: Vocab::default(),
        pos: Vocab::default(),
        composite_dict: vec![],
        word_dict: HashMap::from([
            ("saw".into(), "saw".into()),
            ("missing".into(), String::new()),
        ]),
    };
    let mut lemma = Lemma {
        meta,
        sequence: None,
        composite: HashMap::from([(("saw".into(), "VERB".into()), "see".into())]),
    };
    let mut data = sentences(&["SAW", "Missing", "Unseen", "_"], false);
    data[0].tokens[0].words[0].upos = Some("VERB".into());
    data[0].tokens[0].words[0].xpos = Some("VBD".into());
    lemma.process(&mut data).unwrap();
    assert_eq!(data[0].tokens[0].words[0].lemma.as_deref(), Some("see"));
    assert_eq!(data[0].tokens[0].words[0].xpos.as_deref(), Some("VBD"));
    assert_eq!(data[0].tokens[1].words[0].lemma, None);
    assert_eq!(data[0].tokens[2].words[0].lemma.as_deref(), Some("unseen"));
    assert_eq!(data[0].tokens[3].words[0].lemma.as_deref(), Some("_"));
}

#[test]
fn identity_lemma_keeps_original_unicode_and_annotations() {
    let meta = LemmaMeta {
        kind: "identity".into(),
        args: Args::default(),
        char: Vocab::default(),
        pos: Vocab::default(),
        word_dict: HashMap::new(),
        composite_dict: vec![],
    };
    let mut lemma = Lemma {
        meta,
        sequence: None,
        composite: HashMap::new(),
    };
    let mut data = sentences(&["İ", "a\u{301}", "😊"], false);
    data[0].tokens[0].words[0].upos = Some("NOUN".into());
    lemma.process(&mut data).unwrap();
    for token in &data[0].tokens {
        assert_eq!(token.words[0].lemma.as_deref(), Some(token.text.as_str()));
    }
    assert_eq!(data[0].tokens[0].words[0].upos.as_deref(), Some("NOUN"));
}
