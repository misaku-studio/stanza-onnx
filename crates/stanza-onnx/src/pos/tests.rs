use super::*;
use crate::types::Token;
use serde_json::json;

fn sentence(word_count: usize) -> Sentence {
    Sentence {
        tokens: (0..word_count)
            .map(|index| Token {
                text: format!("word{index}"),
                start: index,
                end: index + 1,
                mwt: false,
                words: vec![Word::new(format!("word{index}"))],
            })
            .collect(),
    }
}

fn vocab(value: serde_json::Value) -> Vocab {
    serde_json::from_value(value).unwrap()
}

#[test]
fn batching_preserves_word_positions_and_bounds_padding() {
    let mut sentences = vec![
        sentence(2),
        sentence(4),
        sentence(0),
        sentence(4),
        sentence(1),
    ];
    sentences[0].tokens[0].words.push(Word::new("expanded"));
    let batches = plan_pos_batches(&sentences, 8).unwrap();
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);
    assert_eq!(batches[0][0].words[0].sentence, 1);
    assert_eq!(batches[0][1].words[0].sentence, 3);
    assert_eq!(
        batches[1][0].words[1],
        WordLocation {
            sentence: 0,
            token: 0,
            word: 1
        }
    );
    batches[1][0].words[1].get_mut(&mut sentences).upos = Some("NOUN".to_string());
    assert_eq!(
        sentences[0].tokens[0].words[1].upos.as_deref(),
        Some("NOUN")
    );
    assert!(plan_pos_batches(&sentences, 0).is_err());
}

#[test]
fn single_long_sentence_remains_a_single_batch() {
    let sentences = vec![sentence(9), sentence(1)];
    let batches = plan_pos_batches(&sentences, 8).unwrap();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0][0].words.len(), 9);
    assert!(plan_pos_batches(&[sentence(0)], 8).unwrap().is_empty());
}

#[test]
fn cpu_sentence_batches_bound_padding_and_preserve_whole_context_rows() {
    let sentences = vec![
        sentence(32),
        sentence(129),
        sentence(16),
        sentence(32),
        sentence(64),
        sentence(1),
        sentence(0),
    ];
    let batches = plan_pos_batches(&sentences, MAX_PADDED_WORDS_CPU).unwrap();
    let lengths: Vec<_> = batches
        .iter()
        .map(|batch| batch.iter().map(|row| row.words.len()).collect::<Vec<_>>())
        .collect();
    assert_eq!(lengths, vec![vec![129], vec![64, 32], vec![32, 16, 1]]);
    let mut source_rows = Vec::new();
    for batch in &batches {
        assert!(batch.len() == 1 || batch.len() * batch[0].words.len() <= MAX_PADDED_WORDS_CPU);
        for row in batch {
            let index = row.words[0].sentence;
            assert!(row.words.iter().all(|location| location.sentence == index));
            assert_eq!(row.words.len(), sentences[index].tokens.len());
            source_rows.push(index);
        }
    }
    source_rows.sort_unstable();
    assert_eq!(source_rows, vec![0, 1, 2, 3, 4, 5]);
    let accelerator_batches = plan_pos_batches(&sentences, MAX_PADDED_WORDS).unwrap();
    assert_eq!(accelerator_batches.len(), 1);
    assert_eq!(accelerator_batches[0].len(), 6);
}

#[test]
fn character_ranges_bound_padding_without_splitting_oversized_words() {
    let lengths = [300, 130, 64, 64, 33, 20, 3, 1];
    let ranges = plan_char_ranges(&lengths, 128).unwrap();
    assert_eq!(
        ranges,
        vec![
            CharRange {
                start: 0,
                end: 1,
                max_chars: 300
            },
            CharRange {
                start: 1,
                end: 2,
                max_chars: 130
            },
            CharRange {
                start: 2,
                end: 4,
                max_chars: 64
            },
            CharRange {
                start: 4,
                end: 7,
                max_chars: 33
            },
            CharRange {
                start: 7,
                end: 8,
                max_chars: 1
            },
        ]
    );
    assert_eq!(
        ranges
            .iter()
            .flat_map(|range| range.start..range.end)
            .collect::<Vec<_>>(),
        (0..lengths.len()).collect::<Vec<_>>()
    );
    for range in ranges {
        assert!(range.end - range.start == 1 || (range.end - range.start) * range.max_chars <= 128);
    }
}

#[test]
fn character_ranges_handle_exact_limits_empty_inputs_and_invalid_lengths() {
    assert_eq!(
        plan_char_ranges(&[8; 17], 128).unwrap(),
        vec![
            CharRange {
                start: 0,
                end: 16,
                max_chars: 8
            },
            CharRange {
                start: 16,
                end: 17,
                max_chars: 8
            },
        ]
    );
    assert!(plan_char_ranges(&[], 128).unwrap().is_empty());
    assert!(plan_char_ranges(&[1], 0).is_err());
    assert!(plan_char_ranges(&[0], 128).is_err());
    assert!(plan_char_ranges(&[3, 0], 128).is_err());
    assert!(plan_char_ranges(&[2, 3], 128).is_err());
    assert_eq!(plan_char_ranges(&[usize::MAX, 1], 128).unwrap()[0].end, 1);
    assert_eq!(
        plan_char_ranges(&[64, 32, 16, 8, 1], usize::MAX).unwrap(),
        vec![CharRange {
            start: 0,
            end: 5,
            max_chars: 64
        }]
    );
}

#[test]
fn character_word_planning_preserves_duplicates_and_unicode_scalar_ids() {
    let vocabulary = vocab(
        json!({"sizes": [10], "_unit2id": {"a": 6, "b": 7, "😀": 8, "A": 9},
        "_id2unit": ["<PAD>", "<UNK>", "<EMPTY>", "<ROOT>", "x", "y", "a", "b", "😀", "A"]}),
    );
    let rows = plan_char_words(&["a", "😀a", "ab", "ab", "😀😀😀", "A"], &vocabulary, true);
    assert_eq!(rows[0].ids, vec![8, 8, 8]);
    assert_eq!(rows[0].positions, vec![4]);
    assert!(rows
        .iter()
        .any(|row| row.ids == [8, 6] && row.positions == [1]));
    let duplicates: Vec<_> = rows
        .iter()
        .filter(|row| row.ids == [6, 7])
        .flat_map(|row| row.positions.iter().copied())
        .collect();
    assert_eq!(duplicates, vec![2, 3]);
    assert!(rows
        .iter()
        .any(|row| row.ids == [9] && row.positions == [5]));
    let mut covered: Vec<_> = rows
        .iter()
        .flat_map(|row| row.positions.iter().copied())
        .collect();
    covered.sort_unstable();
    assert_eq!(covered, vec![0, 1, 2, 3, 4, 5]);
    assert!(plan_char_words(&[], &vocabulary, true).is_empty());
    assert!(plan_pos_batches(&[], MAX_PADDED_WORDS).unwrap().is_empty());
}

#[test]
fn deduplication_uses_exact_text_and_is_disabled_for_accelerator_batches() {
    let vocabulary = vocab(json!({"sizes": [6], "_unit2id": {"a": 4, "A": 5},
        "_id2unit": ["<PAD>", "<UNK>", "<EMPTY>", "<ROOT>", "a", "A"]}));
    let texts = ["a", "a", "A", "á", "é", "a\u{301}"];
    let rows = plan_char_words(&texts, &vocabulary, true);
    assert_eq!(rows.len(), 5);
    assert!(rows
        .iter()
        .any(|row| row.positions == [0, 1] && row.ids == [4]));
    assert!(rows
        .iter()
        .any(|row| row.positions == [2] && row.ids == [5]));
    // Unknown characters can have identical model IDs, but different text
    // stays distinct. No casing or normalization is used as a cache key.
    assert!(rows
        .iter()
        .any(|row| row.positions == [3] && row.ids == [1]));
    assert!(rows
        .iter()
        .any(|row| row.positions == [4] && row.ids == [1]));
    assert!(rows
        .iter()
        .any(|row| row.positions == [5] && row.ids == [4, 1]));
    let original = plan_char_words(&texts, &vocabulary, false);
    assert_eq!(original.len(), texts.len());
    assert!(original.iter().all(|row| row.positions.len() == 1));
}

#[test]
fn character_feature_scatter_restores_source_positions_and_keeps_padding() {
    let rows = vec![
        CharWord {
            ids: vec![8, 8],
            positions: vec![3, 0],
        },
        CharWord {
            ids: vec![6],
            positions: vec![2],
        },
    ];
    let destinations = [0, 2, 8, 6];
    let mut features = vec![0.; 10];
    scatter_char_features(
        &rows,
        &[10., 11., 20., 21.],
        &destinations,
        &mut features,
        2,
    )
    .unwrap();
    assert_eq!(features, vec![10., 11., 0., 0., 0., 0., 10., 11., 20., 21.]);
    assert!(scatter_char_features(&rows, &[10.], &destinations, &mut features, 2).is_err());
    assert!(scatter_char_features(
        &rows,
        &[10., 11., 20., 21.],
        &destinations[..1],
        &mut features,
        2
    )
    .is_err());
    let empty_rows = Vec::new();
    scatter_char_features(&empty_rows, &[], &[], &mut features, 2).unwrap();
}

#[test]
fn flat_vocabulary_ignores_before_lowercasing_and_ties_choose_first() {
    let value = vocab(json!({"lower": true, "ignore": ["SKIP"], "sizes": [5],
        "_unit2id": {"skip": 4, "ος": 8}, "_id2unit": ["<PAD>", "<UNK>", "<EMPTY>", "<ROOT>", "NOUN"]}));
    assert_eq!(value.id("SKIP"), 2);
    assert_eq!(value.id("Skip"), 4);
    assert_eq!(value.id("ΟΣ"), 8);
    assert_eq!(value.id("missing"), 1);
    assert_eq!(value.decode(&[0., 0., 3., 0., 3.]).unwrap(), "_");
    assert_eq!(value.decode(&[0., 0., 0., 0., 3.]).unwrap(), "NOUN");
}

#[test]
fn composite_features_skip_empty_heads_and_keep_key_order() {
    let value = vocab(json!({"composite": true, "keyed": true, "sep": "|",
        "keys": ["Case", "Number"], "sizes": [4, 4],
        "_unit2id": {"Case": {"<PAD>": 0, "<UNK>": 1, "<EMPTY>": 2, "Nom": 3},
            "Number": {"<PAD>": 0, "<UNK>": 1, "<EMPTY>": 2, "Sing": 3}},
        "_id2unit": {"Case": ["<PAD>", "<UNK>", "<EMPTY>", "Nom"],
            "Number": ["<PAD>", "<UNK>", "<EMPTY>", "Sing"]}}));
    assert_eq!(
        value.decode(&[0., 0., 1., 0., 0., 0., 0., 1.]).unwrap(),
        "Number=Sing"
    );
    assert_eq!(
        value.decode(&[0., 0., 0., 1., 0., 0., 0., 1.]).unwrap(),
        "Case=Nom|Number=Sing"
    );
    assert_eq!(
        value.decode(&[0., 0., 1., 0., 0., 0., 1., 0.]).unwrap(),
        "_"
    );
}

#[test]
fn composite_xpos_accepts_numeric_keys_and_unkeyed_separator() {
    let value = vocab(json!({"composite": true, "keyed": false, "sep": "",
        "keys": [0, 1], "sizes": [4, 4],
        "_unit2id": {"0": {"<PAD>": 0, "<UNK>": 1, "<EMPTY>": 2, "N": 3},
            "1": {"<PAD>": 0, "<UNK>": 1, "<EMPTY>": 2, "C": 3}},
        "_id2unit": {"0": ["<PAD>", "<UNK>", "<EMPTY>", "N"],
            "1": ["<PAD>", "<UNK>", "<EMPTY>", "C"]}}));
    assert_eq!(
        value.decode(&[0., 0., 0., 1., 0., 0., 0., 1.]).unwrap(),
        "NC"
    );
}

#[test]
fn input_only_pretrained_vocabulary_needs_no_output_head_sizes() {
    let value = vocab(json!({"lower": true, "_unit2id": {"hello": 4},
        "_id2unit": ["<PAD>", "<UNK>", "<EMPTY>", "<ROOT>", "hello"]}));
    assert_eq!(value.id("Hello"), 4);
    assert_eq!(value.id("unseen"), 1);
    assert!(value.decode(&[0., 0., 0., 0., 1.]).is_err());
}

#[test]
#[ignore = "Requires exported language bundles in STANZA_ONNX_POS_BUNDLE_ROOT"]
fn exported_pos_metadata_schema_audit() {
    let root = std::env::var("STANZA_ONNX_POS_BUNDLE_ROOT")
        .expect("Set STANZA_ONNX_POS_BUNDLE_ROOT to a directory of language bundles");
    let mut checked = 0;
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path().join("pos-vocab.json");
        if !path.is_file() {
            continue;
        }
        let metadata: Metadata = serde_json::from_slice(&fs::read(&path).unwrap())
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        for name in ["word", "upos", "xpos", "feats"] {
            metadata.vocab(name).unwrap();
        }
        for name in ["upos", "xpos", "feats"] {
            let vocab = metadata.vocab(name).unwrap();
            let width = vocab.sizes.iter().sum();
            vocab
                .decode(&vec![0.; width])
                .unwrap_or_else(|error| panic!("{} {name}: {error}", path.display()));
        }
        if metadata.has_char {
            metadata.vocab("char").unwrap();
        }
        if metadata.has_pretrain {
            assert!(metadata.pretrained_vocab.is_some(), "{}", path.display());
        }
        checked += 1;
    }
    assert!(checked > 0, "No POS metadata files found");
    println!("Parsed and validated output vocabularies for {checked} POS metadata files");
}

#[test]
fn backward_context_uses_code_points_and_reverses_word_offsets() {
    let vocabulary = vocab(
        json!({"sizes": [10], "_unit2id": {"\n": 4, " ": 5, "a": 6, "b": 7, "😀": 8},
        "_id2unit": ["<PAD>", "<UNK>", "<EMPTY>", "<ROOT>", "\n", " ", "a", "b", "😀", "x"]}),
    );
    let mut metadata = ContextMetadata {
        filename: "unused".to_string(),
        backwards: true,
        vocab: vocabulary,
        feature_dim: 2,
    };
    let row = context_row(&["ab", "😀"], 7, &metadata);
    assert_eq!(row.chars, vec![4, 8, 5, 7, 6, 5]);
    assert_eq!(row.offsets, vec![5, 2]);
    assert_eq!(row.original_index, 7);
    metadata.backwards = false;
    let row = context_row(&["ab", "😀"], 7, &metadata);
    assert_eq!(row.chars, vec![4, 6, 7, 5, 8, 5]);
    assert_eq!(row.offsets, vec![3, 5]);
}

#[test]
fn pretrained_vectors_are_little_endian_and_validate_shape() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vectors.f32");
    let bytes: Vec<_> = [1.25f32, -2.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    fs::write(&path, bytes).unwrap();
    assert_eq!(read_pretrained(&path, [1, 2]).unwrap(), vec![1.25, -2.5]);
    assert!(read_pretrained(&path, [2, 2]).is_err());
}
