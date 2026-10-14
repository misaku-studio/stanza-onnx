use crate::{
    engine::SessionOptions,
    lemma_mwt::{Lemma, Mwt},
    manifest::{load_bundle, BundleManifest},
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
    tokenizer: Tokenizer,
    mwt: Option<Mwt>,
    pos: Option<Pos>,
    lemma: Option<Lemma>,
}
impl Pipeline {
    pub fn load(
        model_dir: &Path,
        options: &SessionOptions,
        verify_integrity: bool,
    ) -> Result<Self> {
        let (dir, manifest) = load_bundle(model_dir, verify_integrity)?;
        let tokenizer = Tokenizer::load(&dir, options)?;
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
        let mut sentences = self.tokenizer.process(text)?;
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
