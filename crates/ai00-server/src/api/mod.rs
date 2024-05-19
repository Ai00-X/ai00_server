use std::time::Duration;

use ai00_core::{RuntimeInfo, ThreadRequest};
use anyhow::Result;
use flume::Sender;

pub mod adapter;
pub mod auth;
pub mod file;
pub mod model;
pub mod oai;

pub use adapter::adapters;
pub use file::{dir, load_config, models, save_config, unzip};
pub use model::{info, load, save, state, unload};

pub async fn try_request_info(sender: Sender<ThreadRequest>) -> Result<RuntimeInfo> {
    let (info_sender, info_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Info(info_sender));
    let _info = info_receiver.recv_async().await?;
    Ok(_info)
}

pub async fn request_info(sender: Sender<ThreadRequest>, sleep: Duration) -> RuntimeInfo {
    loop {
        if let Ok(_info) = try_request_info(sender.clone()).await {
            break _info;
        }
        tokio::time::sleep(sleep).await;
    }
}

pub async fn request_info_stream(
    sender: Sender<ThreadRequest>,
    info_sender: Sender<RuntimeInfo>,
    sleep: Duration,
) {
    loop {
        if let Ok(_info) = try_request_info(sender.clone()).await {
            if info_sender.send(_info).is_err() {
                break;
            }
        }
        tokio::time::sleep(sleep).await;
    }
}
