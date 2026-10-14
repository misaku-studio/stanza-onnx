use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::{json, Value};
use stanza_onnx::{OffsetUnit, Pipeline, SessionOptions, RUNTIME_VERSION};
use std::{
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(version, about = "Experimental native Stanza ONNX pipeline runtime")]
struct Arguments {
    /// Path to a versioned language bundle.
    #[arg(long)]
    model_dir: PathBuf,
    /// Explicit ONNX Runtime shared library; alternatively set ORT_DYLIB_PATH.
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "coreml", "directml"])]
    provider: String,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Skip content hashes for a trusted cache. Asset/path validation still runs.
    #[arg(long)]
    skip_integrity: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Analyze one text and print normalized sentence/token annotations.
    Analyze {
        #[arg(long)]
        text: String,
        #[arg(
            long,
            default_value = "unicode-code-point",
            value_parser = ["unicode-code-point", "utf16", "utf8"]
        )]
        offset_unit: String,
    },
    /// Keep sessions loaded and exchange one JSON request/response per line.
    Serve,
    /// Analyze a fixture file (texts:[{id,text}]) through the same native API.
    Batch {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Print the loaded pipeline's capabilities and version information.
    Info,
}

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    id: Value,
    method: String,
    #[serde(default)]
    params: Value,
}

fn info(pipeline: &Pipeline) -> Value {
    json!({
        "language": pipeline.manifest.language,
        "capabilities": pipeline.manifest.capabilities,
        "tokenizerKind": pipeline.manifest.tokenizer_kind,
        "runtimeVersion": RUNTIME_VERSION,
        "stanzaVersion": pipeline.manifest.stanza_version,
        "resourceVersion": pipeline.manifest.resource_version,
        "unicodeVersion": pipeline.manifest.unicode_version,
    })
}

fn analyze(pipeline: &mut Pipeline, text: &str, unit: OffsetUnit) -> Result<Value> {
    Ok(json!({
        "language": pipeline.manifest.language,
        "offsetUnit": unit,
        "sentences": pipeline.analyze(text, unit)?,
    }))
}

/// The second return value requests shutdown after its response has been flushed.
fn process_request(pipeline: &mut Pipeline, line: &str) -> (Value, bool) {
    let request = match serde_json::from_str::<Request>(line) {
        Ok(value) => value,
        Err(error) => {
            return (
                json!({
                    "id": null,
                    "error": {
                        "code": "invalid_request",
                        "message": error.to_string(),
                    },
                }),
                false,
            )
        }
    };
    let result: Result<Value> = (|| match request.method.as_str() {
        "info" => Ok(info(pipeline)),
        "analyze" => {
            let text = request.params["text"]
                .as_str()
                .context("params.text must be a string")?;
            let unit = if let Some(value) = request.params.get("offsetUnit") {
                serde_json::from_value::<OffsetUnit>(value.clone()).context("Invalid offsetUnit")?
            } else {
                OffsetUnit::CodePoint
            };
            analyze(pipeline, text, unit)
        }
        "shutdown" => Ok(json!({"closed": true})),
        method => bail!("Unknown method {method}"),
    })();
    let shutdown = request.method == "shutdown" && result.is_ok();
    (
        match result {
            Ok(value) => json!({"id": request.id, "result": value}),
            Err(error) => json!({
                "id": request.id,
                "error": {
                    "code": "request_failed",
                    "message": format!("{error:#}"),
                },
            }),
        },
        shutdown,
    )
}

fn serve(pipeline: &mut Pipeline) -> Result<()> {
    let input = io::stdin();
    let mut output = io::stdout().lock();
    for line in input.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let (response, shutdown) = process_request(pipeline, &line);
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
        if shutdown {
            break;
        }
    }
    Ok(())
}

fn analyze_batch(pipeline: &mut Pipeline, input: Value) -> Result<Value> {
    let texts = input["texts"]
        .as_array()
        .context("Input must contain a texts array")?;
    let mut seen = std::collections::HashSet::new();
    let mut cases = Vec::new();
    for entry in texts {
        let id = entry["id"].as_str().context("Input id must be a string")?;
        ensure!(seen.insert(id), "Duplicate input id {id}");
        let text = entry["text"]
            .as_str()
            .context("Input text must be a string")?;
        cases.push(json!({
            "id": id,
            "text": text,
            "sentences": pipeline.analyze(text, OffsetUnit::CodePoint)?,
        }));
    }
    Ok(json!({
        "language": pipeline.manifest.language,
        "stanzaVersion": pipeline.manifest.stanza_version,
        "processors": pipeline.manifest.processors,
        "unicodeVersion": pipeline.manifest.unicode_version,
        "parity": cases,
    }))
}

fn write_batch_result(result: Value, output: Option<PathBuf>) -> Result<()> {
    let serialized = serde_json::to_vec_pretty(&result)?;
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, serialized)?;
    } else {
        io::stdout().write_all(&serialized)?;
        println!();
    }
    Ok(())
}

fn execute() -> Result<()> {
    let arguments = Arguments::parse();
    ensure!(arguments.threads > 0, "Thread count must be positive");
    let library = arguments
        .ort_library
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from));
    let options = SessionOptions {
        threads: arguments.threads,
        provider: arguments.provider,
    };
    let mut pipeline = Pipeline::load_with_library(
        &arguments.model_dir,
        &options,
        !arguments.skip_integrity,
        library.as_deref(),
    )?;
    match arguments.command {
        Command::Info => println!("{}", info(&pipeline)),
        Command::Serve => serve(&mut pipeline)?,
        Command::Analyze { text, offset_unit } => {
            let unit: OffsetUnit = serde_json::from_value(json!(offset_unit))?;
            println!("{}", analyze(&mut pipeline, &text, unit)?);
        }
        Command::Batch { input, output } => {
            let input: Value = serde_json::from_slice(&fs::read(input)?)?;
            let result = analyze_batch(&mut pipeline, input)?;
            write_batch_result(result, output)?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("stanza-onnx: {error:#}");
        std::process::exit(1);
    }
}
