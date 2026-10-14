pub mod engine;
pub mod lemma_mwt;
pub mod manifest;
pub mod pipeline;
pub mod pos;
pub mod tokenizer;
pub mod types;
pub mod unicode;

pub use engine::{initialize_library, SessionOptions};
pub use pipeline::Pipeline;
pub use types::{OffsetUnit, Sentence, Token, Word};
pub const RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Unicode data version used for preprocessing and bundle compatibility.
pub const UNICODE_VERSION: &str = "18.0.0";
