use std::time::Duration;

use anyhow::Result;
use flume::Sender;

pub mod adapter;
pub mod file;
pub mod load;
pub mod oai;

pub use adapter::{salvo_adapters};
pub use file::{salvo_dir, salvo_models, salvo_load_config, salvo_save_config, salvo_unzip};
pub use load::{salvo_info, salvo_load, salvo_state, salvo_unload};
