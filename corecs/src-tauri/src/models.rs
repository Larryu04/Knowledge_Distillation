//! Model discovery. CoreCS is model-agnostic: any valid `.gguf` in the models folder can be
//! loaded. A checkpoint may ship an optional profile next to it (`<stem>.corecs.json`, written
//! by `scripts/export_gguf.py`) describing how it was trained to be prompted.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};

pub const DEFAULT_CONTEXT: u32 = 4096;
pub const DEFAULT_MAX_TOKENS: u64 = 1024;

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct ModelProfile {
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    /// Text prepended to the first user turn. Used instead of a system message because
    /// templates like Mistral's reject the `system` role.
    #[serde(default)]
    pub prompt: Option<PromptProfile>,
    /// Passed through verbatim into the llama-server request body
    /// (temperature, top_p, repeat_penalty, max_tokens, dry_multiplier, ...).
    #[serde(default)]
    pub sampling: Map<String, Value>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct PromptProfile {
    pub preamble: String,
    #[serde(default = "default_separator")]
    pub separator: String,
}

fn default_separator() -> String {
    "\n\n".into()
}

impl ModelProfile {
    pub fn context_length(&self) -> u32 {
        self.context_length.unwrap_or(DEFAULT_CONTEXT)
    }

    pub fn max_tokens(&self) -> u64 {
        self.sampling.get("max_tokens").and_then(Value::as_u64).unwrap_or(DEFAULT_MAX_TOKENS)
    }
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// File name; stable identifier used by the UI and stored in settings.
    pub id: String,
    pub path: PathBuf,
    pub display_name: String,
    pub size_bytes: u64,
    pub has_profile: bool,
    #[serde(skip)]
    pub profile: ModelProfile,
    #[serde(skip)]
    modified: std::time::SystemTime,
}

fn is_gguf(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}

fn profile_path(gguf: &Path) -> PathBuf {
    gguf.with_extension("corecs.json")
}

/// Loads a single model file (and its profile, if any). Fails if it is not a GGUF file.
pub fn load(path: &Path) -> Result<ModelInfo> {
    // Absolute because llama-server runs with its own folder as working directory.
    // (Not canonicalize(): its \\?\ prefix on Windows is not understood by llama.cpp.)
    let path = &std::path::absolute(path)?;
    if !is_gguf(path) {
        return Err(Error::msg(format!("{} is not a valid GGUF model", path.display())));
    }
    let meta = fs::metadata(path)?;
    let id = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();

    let profile_file = profile_path(path);
    let profile: Option<ModelProfile> = match fs::read_to_string(&profile_file) {
        Ok(text) => match serde_json::from_str(&text) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("[models] ignoring invalid profile {}: {e}", profile_file.display());
                None
            }
        },
        Err(_) => None,
    };
    let has_profile = profile.is_some();
    let profile = profile.unwrap_or_default();
    let display_name = profile
        .display_name
        .clone()
        .unwrap_or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());

    Ok(ModelInfo {
        id,
        path: path.to_path_buf(),
        display_name,
        size_bytes: meta.len(),
        has_profile,
        profile,
        modified: meta.modified()?,
    })
}

/// All valid models in `dir`, newest first.
pub fn scan(dir: &Path) -> Result<Vec<ModelInfo>> {
    fs::create_dir_all(dir)?;
    let mut models: Vec<ModelInfo> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")))
        .filter_map(|p| load(&p).ok())
        .collect();
    models.sort_by(|a, b| b.modified.cmp(&a.modified));
    Ok(models)
}
