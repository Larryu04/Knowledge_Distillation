//! SidecarManager: owns the bundled `llama-server` process.
//!
//! The server listens on 127.0.0.1 only, on a random free port, protected by a random
//! per-launch API key. Only the Rust core knows the port and key; the UI never talks to it.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;

use crate::error::{Error, Result};
use crate::models::ModelInfo;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const LOG_TAIL_LINES: usize = 40;

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "state", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum EngineStatus {
    NoModel,
    Starting { model: String, model_id: String },
    Ready { model: String, model_id: String, device: Option<String> },
    Stopped,
    Failed { message: String },
}

/// Where a running server can be reached.
#[derive(Clone)]
pub struct Endpoint {
    pub base_url: String,
    pub api_key: String,
    pub model: ModelInfo,
}

struct Running {
    endpoint: Endpoint,
    kill: oneshot::Sender<()>,
    monitor: JoinHandle<()>,
}

pub type StatusListener = Arc<dyn Fn(&EngineStatus) + Send + Sync>;

pub struct Engine {
    server_exe: PathBuf,
    running: AsyncMutex<Option<Running>>,
    status: Arc<Mutex<EngineStatus>>,
    listener: StatusListener,
    http: reqwest::Client,
    #[cfg(windows)]
    job: Option<job::Job>,
}

impl Engine {
    pub fn new(server_exe: PathBuf, listener: StatusListener) -> Self {
        Engine {
            server_exe,
            running: AsyncMutex::new(None),
            status: Arc::new(Mutex::new(EngineStatus::NoModel)),
            listener,
            http: loopback_client(),
            #[cfg(windows)]
            job: job::Job::new(),
        }
    }

    /// `llama-server` sits next to the app executable (Tauri copies the sidecar there,
    /// in dev and in installed builds alike).
    pub fn default_server_exe() -> Result<PathBuf> {
        let exe = std::env::current_exe()?;
        let name = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
        Ok(exe.parent().map(|d| d.join(name)).unwrap_or_else(|| PathBuf::from(name)))
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn status(&self) -> EngineStatus {
        self.status.lock().unwrap().clone()
    }

    fn set_status(&self, s: EngineStatus) {
        publish(&self.status, &self.listener, s);
    }

    /// The server endpoint, only while the engine is ready.
    pub async fn endpoint(&self) -> Option<Endpoint> {
        if !matches!(self.status(), EngineStatus::Ready { .. }) {
            return None;
        }
        self.running.lock().await.as_ref().map(|r| r.endpoint.clone())
    }

    /// Stops any running server, then launches one for `model` and waits until it is ready.
    /// `device`: a llama.cpp device name ("Vulkan0"), "none" for CPU only, or None to pick
    /// automatically (prefers a discrete GPU over integrated graphics).
    pub async fn start(&self, model: ModelInfo, device: Option<String>) -> Result<()> {
        let mut running = self.running.lock().await;
        if let Some(r) = running.take() {
            shutdown(r).await;
        }
        self.set_status(EngineStatus::Starting { model: model.display_name.clone(), model_id: model.id.clone() });

        let result = self.launch(&model, device).await;
        match result {
            Ok((r, device)) => {
                *running = Some(r);
                self.set_status(EngineStatus::Ready { model: model.display_name.clone(), model_id: model.id.clone(), device });
                Ok(())
            }
            Err(e) => {
                self.set_status(EngineStatus::Failed { message: e.to_string() });
                Err(e)
            }
        }
    }

    pub async fn stop(&self) {
        if let Some(r) = self.running.lock().await.take() {
            shutdown(r).await;
            self.set_status(EngineStatus::Stopped);
        }
    }

    async fn launch(&self, model: &ModelInfo, device: Option<String>) -> Result<(Running, Option<String>)> {
        if !self.server_exe.exists() {
            return Err(Error::msg(format!("inference engine not found at {}", self.server_exe.display())));
        }
        let device = match device {
            Some(d) => Some(d),
            None => pick_device(&list_devices(&self.server_exe).await),
        };
        let port = free_port()?;
        let api_key = random_key();

        let mut cmd = command(&self.server_exe);
        cmd.arg("--model").arg(&model.path)
            .args(["--host", "127.0.0.1", "--port", &port.to_string()])
            .args(["--api-key", &api_key])
            .args(["--ctx-size", &model.profile.context_length().to_string()])
            .args(["--parallel", "1"])
            .arg("--jinja")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(d) = &device {
            cmd.args(["--device", d]);
        }
        let mut child = cmd.spawn().map_err(|e| Error::msg(format!("failed to start inference engine: {e}")))?;
        #[cfg(windows)]
        if let Some(job) = &self.job {
            job.assign(&child);
        }

        let log_tail = Arc::new(Mutex::new(VecDeque::with_capacity(LOG_TAIL_LINES)));
        if let Some(stderr) = child.stderr.take() {
            let tail = log_tail.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let mut t = tail.lock().unwrap();
                    if t.len() == LOG_TAIL_LINES {
                        t.pop_front();
                    }
                    t.push_back(line);
                }
            });
        }

        let (kill_tx, kill_rx) = oneshot::channel();
        let (exit_tx, mut exit_rx) = oneshot::channel::<String>();
        let monitor = tokio::spawn(monitor(child, kill_rx, exit_tx, log_tail.clone(), self.status.clone(), self.listener.clone()));

        let base_url = format!("http://127.0.0.1:{port}");
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if let Ok(message) = exit_rx.try_recv() {
                return Err(Error::msg(message));
            }
            if Instant::now() > deadline {
                let _ = kill_tx.send(());
                let _ = monitor.await;
                return Err(Error::msg("inference engine did not become ready in time"));
            }
            // /health is unauthenticated: 503 while loading, 200 once the model is ready.
            if let Ok(resp) = self.http.get(format!("{base_url}/health")).send().await {
                if resp.status().is_success() {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        let endpoint = Endpoint { base_url, api_key, model: model.clone() };
        Ok((Running { endpoint, kill: kill_tx, monitor }, device))
    }
}

fn publish(status: &Mutex<EngineStatus>, listener: &StatusListener, s: EngineStatus) {
    *status.lock().unwrap() = s.clone();
    listener(&s);
}

async fn shutdown(r: Running) {
    let _ = r.kill.send(());
    let _ = r.monitor.await;
}

/// Owns the child process: kills it on request, or reports an unexpected exit.
async fn monitor(
    mut child: Child,
    kill_rx: oneshot::Receiver<()>,
    exit_tx: oneshot::Sender<String>,
    log_tail: Arc<Mutex<VecDeque<String>>>,
    status: Arc<Mutex<EngineStatus>>,
    listener: StatusListener,
) {
    tokio::select! {
        _ = kill_rx => {
            let _ = child.kill().await;
        }
        exit = child.wait() => {
            let code = exit.ok().and_then(|s| s.code()).map_or("unknown".into(), |c| c.to_string());
            let tail = log_tail.lock().unwrap().iter().rev().take(8).rev().cloned().collect::<Vec<_>>().join("\n");
            let message = format!("inference engine exited (code {code})\n{tail}");
            eprintln!("[engine] {message}");
            // During startup the launcher reports the failure; afterwards, flag the crash.
            if exit_tx.send(message.clone()).is_err() || matches!(*status.lock().unwrap(), EngineStatus::Ready { .. }) {
                publish(&status, &listener, EngineStatus::Failed { message });
            }
        }
    }
}

fn command(exe: &Path) -> Command {
    let mut cmd = Command::new(exe);
    if let Some(dir) = exe.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// A client that never goes through a system proxy — the engine is always on loopback.
fn loopback_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(5))
        .build()
        .expect("http client")
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn random_key() -> String {
    format!("{:032x}", rand::random::<u128>())
}

#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub total_mib: u64,
}

pub async fn list_devices(server_exe: &Path) -> Vec<Device> {
    let out = match command(server_exe).arg("--list-devices").stdin(Stdio::null()).output().await {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    parse_devices(&text)
}

/// Parses lines like `  Vulkan0: NVIDIA GeForce RTX 4070 SUPER (11997 MiB, 11229 MiB free)`.
fn parse_devices(text: &str) -> Vec<Device> {
    text.lines()
        .filter_map(|line| {
            let (id, rest) = line.trim().split_once(": ")?;
            if id.contains(' ') || !id.chars().last()?.is_ascii_digit() {
                return None;
            }
            let (name, mem) = rest.rsplit_once(" (")?;
            let total_mib = mem.split_whitespace().next()?.parse().ok()?;
            Some(Device { id: id.into(), name: name.trim().into(), total_mib })
        })
        .collect()
}

/// With several GPUs, llama.cpp splits the model across all of them — including integrated
/// graphics sharing system RAM, which is slower than the discrete card alone. Prefer the
/// discrete GPU. With a single device (or none), let llama.cpp decide.
fn pick_device(devices: &[Device]) -> Option<String> {
    if devices.len() < 2 {
        return None;
    }
    let score = |d: &Device| {
        let n = d.name.to_lowercase();
        let integrated = ["(tm) graphics", "uhd graphics", "iris", "vega", "radeon graphics"];
        if integrated.iter().any(|m| n.contains(m)) {
            0
        } else if n.contains("nvidia") || n.contains("radeon rx") || n.contains("arc") {
            2
        } else {
            1
        }
    };
    devices.iter().max_by_key(|d| (score(d), d.total_mib)).map(|d| d.id.clone())
}

/// Windows: a job object that kills llama-server if CoreCS dies without a clean shutdown
/// (crash, Task Manager, `tauri dev` restarting on a Rust change) — no orphaned server
/// holding gigabytes of VRAM.
#[cfg(windows)]
mod job {
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    pub struct Job(HANDLE);
    // The handle is only used for AssignProcessToJobObject, which is thread-safe.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn new() -> Option<Self> {
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if handle.is_null() {
                    return None;
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    std::mem::size_of_val(&info) as u32,
                );
                // Never closed: the OS closes it when CoreCS exits, killing the children.
                (ok != 0).then_some(Job(handle))
            }
        }

        pub fn assign(&self, child: &tokio::process::Child) {
            if let Some(process) = child.raw_handle() {
                unsafe { AssignProcessToJobObject(self.0, process as HANDLE) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "ggml_vulkan: Found 2 Vulkan devices:\n\
        Available devices:\n  \
        Vulkan0: NVIDIA GeForce RTX 4070 SUPER (11997 MiB, 11229 MiB free)\n  \
        Vulkan1: AMD Radeon(TM) Graphics (16166 MiB, 15357 MiB free)\n";

    #[test]
    fn parses_and_prefers_discrete_gpu() {
        let devs = parse_devices(SAMPLE);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[1].name, "AMD Radeon(TM) Graphics");
        assert_eq!(devs[1].total_mib, 16166);
        // The iGPU reports more (shared) memory, but the RTX must win.
        assert_eq!(pick_device(&devs).as_deref(), Some("Vulkan0"));
    }

    #[test]
    fn single_device_left_to_llama_cpp() {
        let devs = parse_devices("  MTL0: Apple M2 (10922 MiB, 10922 MiB free)");
        assert_eq!(devs.len(), 1);
        assert_eq!(pick_device(&devs), None);
    }
}
