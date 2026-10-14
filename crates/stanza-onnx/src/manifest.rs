use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    pub tokenize: bool,
    pub sentence_split: bool,
    pub multiword_expansion: bool,
    pub pos: bool,
    pub morphology: bool,
    pub lemma: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub processor: String,
    pub package: String,
    pub url: String,
    pub md5: String,
    pub license: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundleManifest {
    pub schema_version: u32,
    pub language: String,
    pub language_name: String,
    pub profile: String,
    pub stanza_version: String,
    pub resource_version: String,
    pub runtime_version: String,
    pub unicode_version: String,
    pub offset_unit: String,
    pub processors: HashMap<String, String>,
    pub capabilities: Capabilities,
    pub files: Vec<Asset>,
    pub sources: Vec<Source>,
}
pub fn validate_asset_path(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.as_bytes()[0].is_ascii_alphanumeric()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            && !name.ends_with('.'),
        "Unsafe bundle asset path {name}"
    );
    let stem = name.split('.').next().unwrap().to_ascii_lowercase();
    ensure!(
        !matches!(stem.as_str(), "con" | "prn" | "aux" | "nul")
            && !((stem.starts_with("com") || stem.starts_with("lpt"))
                && stem.len() == 4
                && matches!(stem.as_bytes()[3], b'1'..=b'9')),
        "Reserved Windows asset path {name}"
    );
    Ok(())
}
fn hexadecimal(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}
impl BundleManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "Unsupported bundle schemaVersion");
        for field in [
            &self.language,
            &self.language_name,
            &self.profile,
            &self.stanza_version,
            &self.resource_version,
            &self.runtime_version,
            &self.unicode_version,
        ] {
            ensure!(!field.trim().is_empty(), "Empty bundle metadata field");
        }
        ensure!(
            self.language.as_bytes()[0].is_ascii_lowercase()
                && self
                    .language
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-'),
            "Invalid language code"
        );
        let patch = self
            .runtime_version
            .strip_prefix("0.2.")
            .context("Incompatible bundle runtimeVersion; expected 0.2.x")?;
        ensure!(
            !patch.is_empty() && patch.bytes().all(|c| c.is_ascii_digit()),
            "Incompatible bundle runtimeVersion"
        );
        ensure!(
            self.offset_unit == "unicode-code-point",
            "Bundle source offsets must count Unicode code points"
        );
        ensure!(
            self.unicode_version == crate::UNICODE_VERSION,
            "Incompatible model preprocessing Unicode version; expected {}",
            crate::UNICODE_VERSION
        );
        ensure!(
            self.processors
                .get("tokenize")
                .is_some_and(|value| !value.is_empty()),
            "Bundle has no tokenizer pipeline"
        );
        for (name, package) in &self.processors {
            ensure!(
                matches!(name.as_str(), "tokenize" | "mwt" | "pos" | "lemma")
                    && !package.trim().is_empty(),
                "Invalid processor {name}"
            );
        }
        let c = &self.capabilities;
        ensure!(
            c.tokenize
                && c.sentence_split
                && c.multiword_expansion == self.processors.contains_key("mwt")
                && c.pos == self.processors.contains_key("pos")
                && c.morphology == c.pos,
            "Capabilities do not match selected processors"
        );
        ensure!(
            matches!(c.lemma.as_str(), "none" | "trained" | "identity")
                && (c.lemma != "none") == self.processors.contains_key("lemma"),
            "Invalid lemma capability"
        );
        ensure!(!self.files.is_empty(), "Bundle lists no runtime assets");
        let mut paths = HashSet::new();
        let mut portable = HashSet::new();
        for file in &self.files {
            validate_asset_path(&file.path)?;
            ensure!(
                portable.insert(file.path.to_ascii_lowercase()),
                "Duplicate bundle asset {}",
                file.path
            );
            ensure!(hexadecimal(&file.sha256, 64), "Invalid asset SHA256");
            paths.insert(file.path.as_str());
        }
        let mut required = vec!["tokenizer.json", "tokenizer.onnx"];
        if self.processors.contains_key("pos") {
            required.extend(["pos-vocab.json", "pos-tagger.onnx"]);
        }
        if self.processors.contains_key("mwt") {
            required.push("mwt.json");
        }
        if self.processors.contains_key("lemma") {
            required.push("lemma.json");
        }
        for name in required {
            ensure!(
                paths.contains(name),
                "Bundle is missing required asset {name}"
            );
        }
        ensure!(
            !self.sources.is_empty(),
            "Bundle records no resource provenance"
        );
        for source in &self.sources {
            ensure!(
                !source.processor.trim().is_empty()
                    && !source.package.trim().is_empty()
                    && !source.license.trim().is_empty()
                    && hexadecimal(&source.md5, 32),
                "Invalid source metadata"
            );
            let url = source
                .url
                .strip_prefix("https://")
                .context("Source URL must use HTTPS")?;
            let authority = url.split('/').next().unwrap_or("");
            ensure!(
                !authority.is_empty()
                    && !authority.contains('@')
                    && !url.chars().any(char::is_whitespace),
                "Invalid source URL"
            );
        }
        Ok(())
    }
}
pub fn load_bundle(dir: &Path, verify_integrity: bool) -> Result<(PathBuf, BundleManifest)> {
    let dir = dir
        .canonicalize()
        .with_context(|| format!("Bundle directory {} is missing", dir.display()))?;
    let manifest_path = dir.join("bundle.json").canonicalize()?;
    ensure!(
        manifest_path.parent() == Some(dir.as_path()),
        "Bundle manifest resolves outside model directory"
    );
    let manifest: BundleManifest =
        serde_json::from_reader(BufReader::new(File::open(manifest_path)?))
            .context("Invalid bundle manifest")?;
    manifest.validate()?;
    for asset in &manifest.files {
        let path = dir.join(&asset.path).canonicalize()?;
        ensure!(
            path.parent() == Some(dir.as_path()),
            "Asset resolves outside bundle: {}",
            asset.path
        );
        let metadata = fs::metadata(&path)?;
        ensure!(
            metadata.is_file() && metadata.len() == asset.size,
            "Asset size mismatch: {}",
            asset.path
        );
        if verify_integrity {
            let mut file = File::open(&path)?;
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            ensure!(
                format!("{:x}", hash.finalize()) == asset.sha256,
                "Asset SHA256 mismatch: {}",
                asset.path
            );
        }
    }
    validate_processor_assets(&dir, &manifest)?;
    Ok((dir, manifest))
}
fn read_meta(dir: &Path, name: &str) -> Result<Value> {
    serde_json::from_reader(BufReader::new(File::open(dir.join(name))?))
        .with_context(|| format!("Invalid processor metadata {name}"))
}
fn positive(value: &Value, name: &str) -> Result<()> {
    ensure!(
        value
            .as_u64()
            .is_some_and(|n| n > 0 && n <= 9007199254740991),
        "Invalid positive integer {name}"
    );
    Ok(())
}
fn validate_processor_assets(dir: &Path, manifest: &BundleManifest) -> Result<()> {
    let files: HashSet<&str> = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect();
    let require = |name: &str| -> Result<()> {
        validate_asset_path(name)?;
        ensure!(
            files.contains(name),
            "Processor references unlisted asset {name}"
        );
        Ok(())
    };
    let token = read_meta(dir, "tokenizer.json")?;
    ensure!(
        token["config"].is_object()
            && token["vocab"]["_unit2id"].is_object()
            && token["mask_regex"].is_string()
            && token["config"]["lang"].is_string()
            && token["config"]["feat_funcs"].is_array(),
        "Invalid tokenizer metadata"
    );
    positive(&token["config"]["feat_dim"], "tokenizer feature width")?;
    if manifest.processors.contains_key("pos") {
        let pos = read_meta(dir, "pos-vocab.json")?;
        ensure!(
            pos["vocab"].is_object()
                && pos["has_char"].is_boolean()
                && pos["has_pretrain"].is_boolean(),
            "Invalid POS metadata"
        );
        if pos["has_char"] == true {
            require("pos-char.onnx")?;
        }
        if pos["has_pretrain"] == true {
            require("pretrained.f32")?;
            let shape = pos["pretrained_shape"]
                .as_array()
                .context("Missing embedding shape")?;
            ensure!(shape.len() == 2, "Embedding shape must have two dimensions");
            let bytes = shape[0]
                .as_u64()
                .context("Invalid embedding rows")?
                .checked_mul(shape[1].as_u64().context("Invalid embedding columns")?)
                .and_then(|n| n.checked_mul(4))
                .context("Embedding size overflow")?;
            ensure!(
                manifest
                    .files
                    .iter()
                    .find(|asset| asset.path == "pretrained.f32")
                    .unwrap()
                    .size
                    == bytes,
                "Embedding shape does not match asset size"
            );
        }
        if let Some(contexts) = pos.get("contexts") {
            for context in contexts.as_array().context("Invalid POS contexts")? {
                ensure!(
                    context["backwards"].is_boolean() && context["vocab"].is_object(),
                    "Invalid character-language-model metadata"
                );
                positive(&context["feature_dim"], "charLM width")?;
                require(
                    context["filename"]
                        .as_str()
                        .context("Missing charLM graph name")?,
                )?;
            }
        }
    }
    for processor in ["mwt", "lemma"] {
        if !manifest.processors.contains_key(processor) {
            continue;
        }
        let meta = read_meta(dir, &format!("{processor}.json"))?;
        let kind = meta["kind"].as_str().context("Missing processor kind")?;
        ensure!(
            matches!(kind, "identity" | "dictionary" | "classifier" | "seq2seq"),
            "Unsupported {processor} implementation {kind}"
        );
        if processor == "mwt" {
            ensure!(
                kind != "identity",
                "Declared MWT has no expansion implementation"
            );
        }
        if processor == "lemma" {
            ensure!(
                (kind == "identity") == (manifest.capabilities.lemma == "identity"),
                "Lemma kind does not match declared capability"
            );
            ensure!(
                kind != "classifier",
                "Invalid lemma classifier implementation"
            );
        }
        if kind != "identity" {
            let args = meta["args"]
                .as_object()
                .context("Missing decoder arguments")?;
            for name in ["batch_size", "beam_size"] {
                if let Some(value) = args.get(name) {
                    positive(value, name)?;
                }
            }
            if let Some(value) = meta.get("decode_beam_size") {
                positive(value, "decode_beam_size")?;
            }
            for name in ["copy", "pos", "caseless", "ensemble_dict", "edit"] {
                if let Some(value) = args.get(name) {
                    ensure!(
                        value.is_boolean(),
                        "Invalid boolean decoder argument {name}"
                    );
                }
            }
        }
        if kind == "classifier" {
            require("mwt.onnx")?;
        }
        if kind == "seq2seq" {
            positive(&meta["args"]["max_dec_len"], "max_dec_len")?;
            positive(&meta["args"]["vocab_size"], "vocab_size")?;
            require(&format!("{processor}_encoder.onnx"))?;
            require(&format!("{processor}_step.onnx"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundles_must_match_runtime_and_preprocessing_versions() {
        let mut manifest = BundleManifest {
            schema_version: 1,
            language: "en".into(),
            language_name: "English".into(),
            profile: "default".into(),
            stanza_version: "1.11.0".into(),
            resource_version: "1.11.0".into(),
            runtime_version: crate::RUNTIME_VERSION.into(),
            unicode_version: crate::UNICODE_VERSION.into(),
            offset_unit: "unicode-code-point".into(),
            processors: HashMap::from([("tokenize".into(), "combined".into())]),
            capabilities: Capabilities {
                tokenize: true,
                sentence_split: true,
                multiword_expansion: false,
                pos: false,
                morphology: false,
                lemma: "none".into(),
            },
            files: ["tokenizer.json", "tokenizer.onnx"]
                .into_iter()
                .map(|path| Asset {
                    path: path.into(),
                    size: 1,
                    sha256: "0".repeat(64),
                })
                .collect(),
            sources: vec![Source {
                processor: "tokenize".into(),
                package: "combined".into(),
                url: "https://example.com/tokenizer.pt".into(),
                md5: "0".repeat(32),
                license: "test".into(),
            }],
        };
        manifest.validate().unwrap();
        manifest.unicode_version = "15.0.0".into();
        assert!(manifest
            .validate()
            .unwrap_err()
            .to_string()
            .contains("Unicode"));
        manifest.unicode_version = crate::UNICODE_VERSION.into();
        manifest.runtime_version = "0.1.0".into();
        assert!(manifest
            .validate()
            .unwrap_err()
            .to_string()
            .contains("runtimeVersion"));
    }
    #[test]
    fn portable_asset_paths_reject_traversal_and_windows_devices() {
        for name in [
            "../model.onnx",
            "/tmp/model.onnx",
            "C:\\model.onnx",
            "model/sub.onnx",
            "NUL.onnx",
            "model.",
        ] {
            assert!(validate_asset_path(name).is_err());
        }
        assert!(validate_asset_path("lemma_step.onnx").is_ok());
    }
    #[test]
    fn numeric_loop_parameters_cannot_be_zero_or_negative() {
        for value in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(1.5),
        ] {
            assert!(positive(&value, "batch_size").is_err());
        }
    }
}
