//! Smoke tests for the ai00_server.
//!
//! These tests verify that the server can start, load a model, and respond to prompts.
//! They are marked `#[ignore]` because they require:
//! - A model file to be present (configured in assets/configs/)
//! - GPU hardware (WebGPU or HIP)
//!
//! Run with: `cargo test --test smoke -- --ignored`
//!
//! Environment variables:
//! - `AI00_TEST_CONFIG`: Path to config file (default: assets/configs/Config.toml)
//! - `AI00_TEST_TIMEOUT`: Server startup timeout in seconds (default: 120)

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_TIMEOUT_SECS: u64 = 120;

struct ServerProcess {
    child: Child,
    port: u16,
    stderr_log: Arc<Mutex<Vec<String>>>,
}

impl ServerProcess {
    fn spawn(config_path: &str, port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        // Build the server binary path
        let binary = env!("CARGO_BIN_EXE_ai00-server");
        eprintln!("Using binary: {}", binary);

        // The server expects to run from the workspace root (where assets/ is located)
        // CARGO_MANIFEST_DIR points to crates/ai00-server, so we go up two levels
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .expect("Failed to find workspace root");
        eprintln!("Working directory: {}", workspace_root.display());

        let mut child = Command::new(binary)
            .args(["--config", config_path, "--port", &port.to_string()])
            .current_dir(workspace_root)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;

        // Capture stderr in a background thread to prevent buffer blocking
        let stderr = child.stderr.take().expect("Failed to capture stderr");
        let stderr_log = Arc::new(Mutex::new(Vec::new()));
        let stderr_log_clone = stderr_log.clone();

        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    eprintln!("[server] {}", line);
                    if let Ok(mut log) = stderr_log_clone.lock() {
                        log.push(line);
                    }
                }
            }
        });

        Ok(Self {
            child,
            port,
            stderr_log,
        })
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn get_logs(&self) -> Vec<String> {
        self.stderr_log
            .lock()
            .map(|l| l.clone())
            .unwrap_or_default()
    }

    async fn wait_ready(&self, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let url = format!("{}/api/models/info", self.base_url());
        let start = Instant::now();

        loop {
            if start.elapsed() > timeout {
                let logs = self.get_logs();
                let log_snippet = logs
                    .iter()
                    .rev()
                    .take(20)
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n");
                return Err(format!(
                    "Server startup timeout after {:?}. Last logs:\n{}",
                    timeout, log_snippet
                )
                .into());
            }

            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    // Check if model is loaded by inspecting the response
                    let text = resp.text().await?;
                    if text.contains("model_path") && !text.contains("\"model_path\":null") {
                        return Ok(());
                    }
                }
                _ => {}
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    async fn complete(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let url = format!("{}/api/oai/completions", self.base_url());

        let request_body = serde_json::json!({
            "prompt": [prompt],
            "max_tokens": 32,
            "stop": ["\n", "."],
            "stream": false,
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 1
        });

        let resp = client
            .post(&url)
            .json(&request_body)
            .send()
            .await?
            .error_for_status()?;

        let json: serde_json::Value = resp.json().await?;
        let text = json["choices"][0]["text"]
            .as_str()
            .ok_or("Missing text in response")?
            .to_string();

        Ok(text)
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn get_timeout() -> Duration {
    let secs = std::env::var("AI00_TEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Smoke test for WebGPU backend.
///
/// Requires:
/// - Config file at `assets/configs/Config.toml` (or set AI00_TEST_CONFIG)
/// - Model file specified in config
/// - WebGPU-capable GPU
///
/// Run with: `cargo test smoke_webgpu -- --ignored`
#[tokio::test]
#[ignore]
async fn smoke_webgpu() {
    let config = std::env::var("AI00_TEST_CONFIG")
        .unwrap_or_else(|_| "assets/configs/Config.toml".to_string());

    // Use a unique port to avoid conflicts
    let port = 19530;

    eprintln!("Starting server with config: {}", config);
    let server = ServerProcess::spawn(&config, port).expect("Failed to spawn server");

    eprintln!("Waiting for server to be ready...");
    let timeout = get_timeout();
    server
        .wait_ready(timeout)
        .await
        .expect("Server failed to start");

    eprintln!("Server ready, sending test prompt...");
    let response = server
        .complete("Q: What is the capital of France?\nA:")
        .await
        .expect("Completion request failed");

    eprintln!("Response: {:?}", response);

    // The response should contain "Paris" (case-insensitive)
    assert!(
        response.to_lowercase().contains("paris"),
        "Expected response to contain 'Paris', got: {}",
        response
    );
}

/// Smoke test for HIP backend.
///
/// Requires:
/// - Config file at `assets/configs/Config.hip.toml` (or set AI00_TEST_CONFIG_HIP)
/// - Model file specified in config
/// - AMD GPU with ROCm/HIP support
/// - Binary built with `--features hip`
///
/// Run with: `cargo test smoke_hip --features hip -- --ignored`
#[tokio::test]
#[ignore]
async fn smoke_hip() {
    let config = std::env::var("AI00_TEST_CONFIG_HIP")
        .unwrap_or_else(|_| "assets/configs/Config.hip.toml".to_string());

    // Use a unique port to avoid conflicts
    let port = 19531;

    eprintln!("Starting server with config: {}", config);
    let server = ServerProcess::spawn(&config, port).expect("Failed to spawn server");

    eprintln!("Waiting for server to be ready...");
    let timeout = get_timeout();
    server
        .wait_ready(timeout)
        .await
        .expect("Server failed to start");

    eprintln!("Server ready, sending test prompt...");
    let response = server
        .complete("Q: What is the capital of France?\nA:")
        .await
        .expect("Completion request failed");

    eprintln!("Response: {:?}", response);

    // The response should contain "Paris" (case-insensitive)
    assert!(
        response.to_lowercase().contains("paris"),
        "Expected response to contain 'Paris', got: {}",
        response
    );
}

/// Test that the server returns consistent results with temperature=0.
#[tokio::test]
#[ignore]
async fn smoke_deterministic() {
    let config = std::env::var("AI00_TEST_CONFIG")
        .unwrap_or_else(|_| "assets/configs/Config.toml".to_string());

    let port = 19532;

    let server = ServerProcess::spawn(&config, port).expect("Failed to spawn server");
    let timeout = get_timeout();
    server
        .wait_ready(timeout)
        .await
        .expect("Server failed to start");

    // With temperature=0 and top_k=1, results should be deterministic
    let prompt = "The quick brown fox";
    let response1 = server.complete(prompt).await.expect("First request failed");
    let response2 = server
        .complete(prompt)
        .await
        .expect("Second request failed");

    assert_eq!(
        response1, response2,
        "Responses should be identical with temperature=0"
    );
}
