//! First-launch model provisioning: which model to pull from the Hugging Face Hub, and pulling it.
//!
//! The installer is the same generic build for everyone. What differs is a small
//! `corecs-model.json` manifest that CoreCS Global hands out next to it (both inside the
//! "student app" ZIP a lecturer downloads after training). The NSIS installer copies the manifest
//! from its own folder into the install folder (`windows/hooks.nsh`), and the app reads it here:
//!
//!   1. a lecturer manifest next to the app executable or in the app data folder, if its model was
//!      not pulled yet -> pull that model;
//!   2. otherwise, if the models folder is empty -> pull the default model (CoreCS v8);
//!   3. otherwise -> nothing to do, the existing models are used.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::{Error, Result};

pub const MANIFEST_FILE: &str = "corecs-model.json";
const HUB: &str = "https://huggingface.co";
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

/// The default model when no lecturer manifest is present. Override at build time with
/// `CORECS_DEFAULT_REPO` / `CORECS_DEFAULT_GGUF` to ship a different default.
const DEFAULT_REPO: &str = match option_env!("CORECS_DEFAULT_REPO") {
    Some(r) => r,
    None => "Larryu04/corecs-v8-gguf",
};
const DEFAULT_GGUF: &str = match option_env!("CORECS_DEFAULT_GGUF") {
    Some(g) => g,
    None => "corecs-v8-q4_k_m.gguf",
};
/// Checksum and size of the published `Larryu04/corecs-v8-gguf/corecs-v8-q4_k_m.gguf`. Only
/// applied to the built-in default, not to an overridden one.
const DEFAULT_IS_V8: bool =
    option_env!("CORECS_DEFAULT_REPO").is_none() && option_env!("CORECS_DEFAULT_GGUF").is_none();
const V8_SHA256: &str = "ad3828ac0b9d2f778839f00ecf4e5cbabef7687ce79e63f7dc42da9e5d785b9b";
const V8_BYTES: u64 = 4_374_992_096;

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct ModelManifest {
    /// Hugging Face model repository, `owner/name`.
    pub repo: String,
    #[serde(default = "default_revision")]
    pub revision: String,
    /// File name of the GGUF inside the repository.
    pub gguf: String,
    /// Optional `<stem>.corecs.json` profile inside the repository.
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    /// Checked after download when present (lowercase hex).
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

fn default_revision() -> String {
    "main".into()
}

pub fn default_manifest() -> ModelManifest {
    ModelManifest {
        repo: DEFAULT_REPO.into(),
        revision: default_revision(),
        gguf: DEFAULT_GGUF.into(),
        profile: Some(DEFAULT_GGUF.trim_end_matches(".gguf").to_string() + ".corecs.json"),
        display_name: Some("CoreCS v8".into()),
        sha256: DEFAULT_IS_V8.then(|| V8_SHA256.into()),
        size_bytes: DEFAULT_IS_V8.then_some(V8_BYTES),
    }
}

/// A plain file name: no directories, no `..`, so a manifest can't write outside the models folder.
fn is_plain_file_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', ':'])
        && Path::new(name).file_name().is_some_and(|n| n == name)
}

fn is_repo_id(repo: &str) -> bool {
    let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
    matches!(repo.split_once('/'), Some((owner, name)) if ok(owner) && ok(name) && !name.contains('/'))
}

impl ModelManifest {
    pub fn validate(&self) -> Result<()> {
        if !is_repo_id(&self.repo) {
            return Err(Error::msg(format!("invalid Hugging Face repository \"{}\"", self.repo)));
        }
        if !is_plain_file_name(&self.gguf) || !self.gguf.to_ascii_lowercase().ends_with(".gguf") {
            return Err(Error::msg(format!("invalid model file name \"{}\"", self.gguf)));
        }
        if self.profile.as_deref().is_some_and(|p| !is_plain_file_name(p)) {
            return Err(Error::msg("invalid profile file name"));
        }
        if self.revision.is_empty() || self.revision.contains(['/', '\\', '?', '#']) {
            return Err(Error::msg(format!("invalid revision \"{}\"", self.revision)));
        }
        Ok(())
    }

    /// Identifies what was pulled, so a manifest is acted on once, not on every launch.
    pub fn key(&self) -> String {
        format!("{}@{}/{}", self.repo, self.revision, self.gguf)
    }

    pub fn label(&self) -> String {
        self.display_name.clone().unwrap_or_else(|| self.gguf.trim_end_matches(".gguf").to_string())
    }

    fn url(&self, file: &str) -> String {
        format!("{HUB}/{}/resolve/{}/{file}?download=true", self.repo, self.revision)
    }
}

/// The first valid lecturer manifest in `dirs`. An unreadable or invalid one is reported and skipped.
pub fn find_custom(dirs: &[PathBuf]) -> Option<ModelManifest> {
    dirs.iter().map(|d| d.join(MANIFEST_FILE)).find_map(|path| {
        let text = std::fs::read_to_string(&path).ok()?;
        match serde_json::from_str::<ModelManifest>(&text).map_err(|e| Error::msg(e.to_string())).and_then(|m| {
            m.validate()?;
            Ok(m)
        }) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("[provision] ignoring {}: {e}", path.display());
                None
            }
        }
    })
}

fn hub_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent(concat!("CoreCS/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(60))
        .build()?)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Downloads the manifest's GGUF (and profile) into `models_dir`, resuming a previous partial
/// download. `progress(received, total)` is called at most every 250 ms. Returns the GGUF path.
pub async fn download(
    m: &ModelManifest,
    models_dir: &Path,
    progress: impl Fn(u64, Option<u64>),
) -> Result<PathBuf> {
    m.validate()?;
    tokio::fs::create_dir_all(models_dir).await?;
    let http = hub_client()?;
    let target = models_dir.join(&m.gguf);
    let part = models_dir.join(format!("{}.part", m.gguf));

    // The profile first: it is tiny, and once the GGUF appears the app may scan the folder.
    if let Some(profile) = &m.profile {
        match http.get(m.url(profile)).send().await.and_then(|r| r.error_for_status()) {
            Ok(resp) => tokio::fs::write(models_dir.join(profile), resp.bytes().await?).await?,
            Err(e) => eprintln!("[provision] no profile for {}: {e}", m.repo), // optional
        }
    }

    // Resume: re-hash what is already on disk, then ask only for the rest.
    let mut hasher = Sha256::new();
    let mut have = 0u64;
    if let Ok(mut f) = tokio::fs::File::open(&part).await {
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            have += n as u64;
        }
    }

    let mut req = http.get(m.url(&m.gguf));
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let resp = req.send().await?;
    let status = resp.status();
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && have > 0 {
        // Everything was already downloaded; only the verification below is left.
    } else if !status.is_success() {
        return Err(Error::msg(match status.as_u16() {
            401 | 403 | 404 => format!(
                "{} was not found on Hugging Face (HTTP {status}). The repository may be private or not published yet.",
                m.url(&m.gguf).trim_end_matches("?download=true")
            ),
            _ => format!("Hugging Face answered HTTP {status}"),
        }));
    }

    let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if status.is_success() && !resumed {
        // Full response (server ignored the range, or a fresh start): begin again.
        hasher = Sha256::new();
        have = 0;
    }
    let total = m.size_bytes.or_else(|| resp.content_length().map(|n| n + have));

    if status.is_success() {
        let mut file = if resumed {
            tokio::fs::OpenOptions::new().append(true).open(&part).await?
        } else {
            tokio::fs::File::create(&part).await?
        };
        let mut stream = resp.bytes_stream();
        let mut last = Instant::now();
        progress(have, total);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            hasher.update(&chunk);
            file.write_all(&chunk).await?;
            have += chunk.len() as u64;
            if last.elapsed() >= PROGRESS_INTERVAL {
                last = Instant::now();
                progress(have, total);
            }
        }
        file.flush().await?;
        progress(have, total);
    }

    if let Some(expected) = m.size_bytes {
        if have != expected {
            return Err(Error::msg(format!("download incomplete ({have} of {expected} bytes); try again to resume")));
        }
    }
    if let Some(expected) = &m.sha256 {
        if !hex(&hasher.finalize()).eq_ignore_ascii_case(expected) {
            let _ = tokio::fs::remove_file(&part).await; // corrupt: don't resume from it
            return Err(Error::msg("the downloaded model is corrupt (checksum mismatch); try again"));
        }
    }
    tokio::fs::rename(&part, &target).await?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_manifest_is_valid() {
        let m = default_manifest();
        m.validate().unwrap();
        assert_eq!(m.profile.as_deref(), Some("corecs-v8-q4_k_m.corecs.json"));
        if DEFAULT_IS_V8 {
            assert_eq!(
                m.url(&m.gguf),
                "https://huggingface.co/Larryu04/corecs-v8-gguf/resolve/main/corecs-v8-q4_k_m.gguf?download=true"
            );
            assert_eq!(m.size_bytes, Some(4_374_992_096));
        }
    }

    #[test]
    fn rejects_paths_and_bad_repos() {
        let ok = default_manifest();
        for bad in [
            ModelManifest { gguf: "../evil.gguf".into(), ..ok.clone() },
            ModelManifest { gguf: "dir\\x.gguf".into(), ..ok.clone() },
            ModelManifest { gguf: "model.bin".into(), ..ok.clone() },
            ModelManifest { repo: "no-slash".into(), ..ok.clone() },
            ModelManifest { repo: "a/b/c".into(), ..ok.clone() },
            ModelManifest { profile: Some("../x.json".into()), ..ok.clone() },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    /// Network: `cargo test --lib -- --ignored hub_download`. Pulls a ~19 MB public GGUF, then
    /// resumes it from a half-written `.part` and checks the result against the first checksum.
    #[tokio::test]
    #[ignore]
    async fn hub_download_resumes_and_verifies() {
        let dir = std::env::temp_dir().join(format!("corecs-hub-{}", std::process::id()));
        let m = ModelManifest {
            repo: "ggml-org/tiny-llamas".into(),
            revision: "main".into(),
            gguf: "stories15M-q4_0.gguf".into(),
            profile: None,
            display_name: None,
            sha256: None,
            size_bytes: None,
        };
        let path = download(&m, &dir, |_, _| {}).await.unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        let sha = hex(&Sha256::digest(&bytes));

        std::fs::remove_file(&path).unwrap();
        std::fs::write(dir.join("stories15M-q4_0.gguf.part"), &bytes[..bytes.len() / 2]).unwrap();
        let m = ModelManifest { sha256: Some(sha), size_bytes: Some(bytes.len() as u64), ..m };
        let seen = std::sync::Mutex::new(Vec::new());
        download(&m, &dir, |r, _| seen.lock().unwrap().push(r)).await.unwrap();
        assert!(seen.lock().unwrap()[0] >= bytes.len() as u64 / 2, "did not resume");
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reads_web_manifest_and_finds_it() {
        let dir = std::env::temp_dir().join(format!("corecs-provision-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // The shape CoreCS Global writes (extra keys are ignored).
        std::fs::write(
            dir.join(MANIFEST_FILE),
            r#"{"version": 1, "source": "corecs-global", "repo": "prof/cs-tutor-gguf", "revision": "main",
                "gguf": "cs-q4_k_m.gguf", "profile": "cs-q4_k_m.corecs.json", "display_name": "CS Tutor",
                "sha256": "ab", "size_bytes": 10}"#,
        )
        .unwrap();
        let m = find_custom(&[dir.join("missing"), dir.clone()]).unwrap();
        assert_eq!(m.repo, "prof/cs-tutor-gguf");
        assert_eq!(m.key(), "prof/cs-tutor-gguf@main/cs-q4_k_m.gguf");
        assert_eq!(m.url("cs-q4_k_m.gguf"), "https://huggingface.co/prof/cs-tutor-gguf/resolve/main/cs-q4_k_m.gguf?download=true");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
