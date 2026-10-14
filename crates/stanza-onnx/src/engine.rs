use anyhow::{bail, Context, Result};
use ort::{
    execution_providers::{CoreMLExecutionProvider, DirectMLExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
    value::Tensor,
};
use std::{collections::HashMap, path::Path, sync::OnceLock};

static LIBRARY_INITIALIZED: OnceLock<()> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub threads: usize,
    pub provider: String,
}
impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            provider: "cpu".into(),
        }
    }
}

pub enum TensorData {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I64 { shape: Vec<usize>, data: Vec<i64> },
    Bool { shape: Vec<usize>, data: Vec<bool> },
}
impl TensorData {
    pub fn f32(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Self {
        Self::F32 {
            shape: shape.into(),
            data,
        }
    }
    pub fn i64(shape: impl Into<Vec<usize>>, data: Vec<i64>) -> Self {
        Self::I64 {
            shape: shape.into(),
            data,
        }
    }
    pub fn bool(shape: impl Into<Vec<usize>>, data: Vec<bool>) -> Self {
        Self::Bool {
            shape: shape.into(),
            data,
        }
    }
}
#[derive(Debug, Clone)]
pub struct TensorOutput {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}
pub struct ModelSession {
    session: Session,
}
impl ModelSession {
    pub fn load(path: &Path, options: &SessionOptions) -> Result<Self> {
        anyhow::ensure!(options.threads > 0, "Thread count must be positive");
        initialize_library(None)?;
        let mut builder = Session::builder()?
            .with_intra_threads(options.threads)?
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;
        match options.provider.as_str() {
            "cpu" => {}
            "coreml" => {
                anyhow::ensure!(
                    cfg!(target_os = "macos"),
                    "CoreML is available only on macOS"
                );
                builder = builder.with_execution_providers([CoreMLExecutionProvider::default()
                    .build()
                    .error_on_failure()])?;
            }
            "directml" => {
                anyhow::ensure!(
                    cfg!(target_os = "windows"),
                    "DirectML is available only on Windows"
                );
                builder = builder
                    .with_memory_pattern(false)?
                    .with_parallel_execution(false)?
                    .with_execution_providers([DirectMLExecutionProvider::default()
                        .build()
                        .error_on_failure()])?;
            }
            other => bail!("Unsupported execution provider {other}"),
        }
        let session = builder
            .commit_from_file(path)
            .with_context(|| format!("Loading graph {}", path.display()))?;
        Ok(Self { session })
    }
    pub fn has_input(&self, name: &str) -> bool {
        self.session.inputs.iter().any(|input| input.name == name)
    }
    pub fn run(
        &mut self,
        inputs: Vec<(&str, TensorData)>,
    ) -> Result<HashMap<String, TensorOutput>> {
        let feeds: Vec<(String, SessionInputValue<'static>)> = inputs
            .into_iter()
            .map(|(name, value)| {
                let tensor: SessionInputValue<'static> = match value {
                    TensorData::F32 { shape, data } => {
                        Tensor::from_array((shape, data.into_boxed_slice()))?.into()
                    }
                    TensorData::I64 { shape, data } => {
                        Tensor::from_array((shape, data.into_boxed_slice()))?.into()
                    }
                    TensorData::Bool { shape, data } => {
                        Tensor::from_array((shape, data.into_boxed_slice()))?.into()
                    }
                };
                Ok((name.to_owned(), tensor))
            })
            .collect::<Result<_>>()?;
        let outputs = self.session.run(feeds)?;
        outputs
            .iter()
            .map(|(name, value)| {
                let (shape, data) = value.try_extract_tensor::<f32>()?;
                let shape = shape
                    .iter()
                    .map(|dimension| usize::try_from(*dimension))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok((
                    name.to_owned(),
                    TensorOutput {
                        shape,
                        data: data.to_vec(),
                    },
                ))
            })
            .collect()
    }
}

/// Optional explicit path avoids relying on a host's global library search path.
pub fn initialize_library(path: Option<&Path>) -> Result<()> {
    if LIBRARY_INITIALIZED.get().is_some() {
        return Ok(());
    }
    let configured = path
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(std::path::PathBuf::from));
    let adjacent = std::env::current_exe()
        .ok()
        .and_then(|executable| executable.parent().map(Path::to_path_buf))
        .and_then(|directory| {
            [
                "libonnxruntime.1.23.2.dylib",
                "libonnxruntime.dylib",
                "libonnxruntime.so.1.23.2",
                "libonnxruntime.so",
                "onnxruntime.dll",
            ]
            .iter()
            .map(|name| directory.join(name))
            .find(|file| file.is_file())
        });
    let path=configured.or(adjacent).context("ONNX Runtime library is not configured. Set --ort-library /path/to/library or ORT_DYLIB_PATH; use scripts/setup_ort.py to obtain the pinned native runtime")?;
    anyhow::ensure!(
        path.is_file(),
        "ONNX Runtime shared library is missing: {}",
        path.display()
    );
    let path = path
        .canonicalize()
        .with_context(|| format!("Cannot resolve ONNX Runtime library {}", path.display()))?;
    // ort's loader panics on dlopen/symbol/API failures. Preflight the ABI first
    // so normal installation errors remain fallible library/CLI errors.
    // SAFETY: callers choose a trusted native runtime library. Its exported C
    // ABI is checked before use; the guard stays loaded through initialization.
    unsafe {
        let guard = libloading::Library::new(&path)
            .with_context(|| format!("Cannot load ONNX Runtime library {}", path.display()))?;
        let get_base: libloading::Symbol<
            unsafe extern "system" fn() -> *const ort::sys::OrtApiBase,
        > = guard
            .get(b"OrtGetApiBase\0")
            .context("Native library is missing OrtGetApiBase")?;
        let base = get_base();
        anyhow::ensure!(!base.is_null(), "ONNX Runtime returned a null API base");
        anyhow::ensure!(
            !((*base).GetApi)(ort::sys::ORT_API_VERSION).is_null(),
            "ONNX Runtime does not support required C API {}",
            ort::sys::ORT_API_VERSION
        );
        let initialized = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ort::init_from(path.to_string_lossy())
                .with_name("stanza-onnx")
                .commit()
        }));
        match initialized {
            Ok(result) => {
                result?;
            }
            Err(_) => bail!("ONNX Runtime initialization failed for {}", path.display()),
        }
    }
    let _ = LIBRARY_INITIALIZED.set(());
    Ok(())
}
