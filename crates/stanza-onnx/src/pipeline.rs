use crate::{
    bengali::BengaliTokenizer,
    engine::{initialize_library, SessionOptions},
    lemma_mwt::{Lemma, Mwt},
    manifest::{load_bundle, BundleManifest, TokenizerKind},
    pos::Pos,
    tokenizer::Tokenizer,
    types::{convert_offsets, OffsetUnit, Sentence},
};
use anyhow::Result;
use std::path::Path;

/// Native pipeline; sessions and their buffers are released automatically on Drop.
/// Mutably borrowing the pipeline serializes calls without a background JS runtime.
pub struct Pipeline {
    pub manifest: BundleManifest,
    tokenizer: PipelineTokenizer,
    mwt: Option<Mwt>,
    pos: Option<Pos>,
    lemma: Option<Lemma>,
}
enum PipelineTokenizer {
    Stanza(Box<Tokenizer>),
    BengaliRule(BengaliTokenizer),
}
impl Pipeline {
    pub fn load(
        model_dir: &Path,
        options: &SessionOptions,
        verify_integrity: bool,
    ) -> Result<Self> {
        Self::load_with_library(model_dir, options, verify_integrity, None)
    }

    /// Choose an explicit ONNX Runtime library only when the bundle needs it.
    /// Rule-only bundles load without touching the process-global ONNX runtime.
    pub fn load_with_library(
        model_dir: &Path,
        options: &SessionOptions,
        verify_integrity: bool,
        library: Option<&Path>,
    ) -> Result<Self> {
        anyhow::ensure!(options.threads > 0, "Thread count must be positive");
        anyhow::ensure!(
            matches!(options.provider.as_str(), "cpu" | "coreml" | "directml"),
            "Unsupported execution provider {}",
            options.provider
        );
        let (dir, manifest) = load_bundle(model_dir, verify_integrity)?;
        let tokenizer = match manifest.tokenizer_kind {
            TokenizerKind::Stanza => {
                initialize_library(library)?;
                PipelineTokenizer::Stanza(Box::new(Tokenizer::load(&dir, options)?))
            }
            TokenizerKind::BengaliRule => PipelineTokenizer::BengaliRule(BengaliTokenizer),
        };
        let mwt = if manifest.processors.contains_key("mwt") {
            Some(Mwt::load(&dir, options)?)
        } else {
            None
        };
        let pos = if manifest.processors.contains_key("pos") {
            Some(Pos::load(&dir, options)?)
        } else {
            None
        };
        let lemma = if manifest.processors.contains_key("lemma") {
            Some(Lemma::load(&dir, options)?)
        } else {
            None
        };
        Ok(Self {
            manifest,
            tokenizer,
            mwt,
            pos,
            lemma,
        })
    }
    pub fn analyze(&mut self, text: &str, offset_unit: OffsetUnit) -> Result<Vec<Sentence>> {
        let mut sentences = match &mut self.tokenizer {
            PipelineTokenizer::Stanza(tokenizer) => tokenizer.process(text)?,
            PipelineTokenizer::BengaliRule(tokenizer) => tokenizer.process(text),
        };
        if let Some(mwt) = &mut self.mwt {
            mwt.process(&mut sentences)?;
        }
        if let Some(pos) = &mut self.pos {
            pos.process(&mut sentences)?;
        }
        if let Some(lemma) = &mut self.lemma {
            lemma.process(&mut sentences)?;
        }
        convert_offsets(text, &mut sentences, offset_unit)?;
        Ok(sentences)
    }
}
