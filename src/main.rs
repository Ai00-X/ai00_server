use std::{
    fs::File,
    io::{BufReader, Cursor, Read},
    net::IpAddr,
    path::{Path, PathBuf},
};
use anyhow::Result;
use clap::Parser;
use memmap2::Mmap;
use crate::middleware::ThreadState;

mod api;
mod config;
mod middleware;
mod run;
mod sampler;

#[cfg(feature = "axum-api")]
mod axum_main;
#[cfg(feature = "salvo-api")]
mod salvo_main;

pub fn load_web(path: impl AsRef<Path>, target: &Path) -> Result<()> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    zip_extract::extract(Cursor::new(&map), target, false)?;
    Ok(())
}

pub fn load_plugin(path: impl AsRef<Path>, target: &Path, name: &String) -> Result<()> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    let root = target.join("plugins");
    if !root.exists() {
        std::fs::create_dir(&root)?;
    }
    let dir = root.join(name);
    std::fs::create_dir(&dir)?;
    zip_extract::extract(Cursor::new(&map), &dir, false)?;
    Ok(())
}

pub fn load_config(path: impl AsRef<Path>) -> Result<config::Config> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(toml::from_str(&contents)?)
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long, short, value_name = "FILE")]
    config: Option<PathBuf>,
    #[arg(long, short)]
    ip: Option<IpAddr>,
    #[arg(long, short, default_value_t = 65530)]
    port: u16,
}

#[tokio::main]
async fn main() {
    if cfg!(feature = "axum-api") {
        #[cfg(feature = "axum-api")]
        axum_main::axum_main().await
    } else {
        #[cfg(feature = "salvo-api")]
        salvo_main::salvo_main().await
    }
}
