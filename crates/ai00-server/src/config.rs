use std::{
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
};

use ai00_core::{
    reload::{AdapterOption, BnfOption, Lora, Model, State, Tokenizer},
    ReloadRequest,
};
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::build_path;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: Model,
    pub lora: Vec<Lora>,
    pub state: Vec<State>,
    pub tokenizer: Tokenizer,
    pub bnf: BnfOption,
    pub adapter: AdapterOption,
    pub listen: ListenerOption,
    pub web: Option<WebOption>,
}

impl TryFrom<Config> for ReloadRequest {
    type Error = anyhow::Error;

    fn try_from(value: Config) -> Result<Self, Self::Error> {
        let Config {
            model:
                Model {
                    name,
                    path,
                    quant,
                    quant_type,
                    precision,
                    token_chunk_size,
                    max_batch,
                    embed_device,
                },
            mut lora,
            mut state,
            tokenizer: Tokenizer {
                path: tokenizer_path,
            },
            bnf,
            adapter,
            ..
        } = value;

        let model_path = build_path(&path, name)?;
        for lora in lora.iter_mut() {
            lora.path = build_path(&path, &lora.path)?;
        }
        for state in state.iter_mut() {
            state.path = build_path(&path, &state.path)?;
        }

        Ok(Self {
            model_path,
            lora,
            state,
            quant,
            quant_type,
            precision,
            token_chunk_size,
            max_batch,
            embed_device,
            tokenizer_path,
            bnf,
            adapter,
        })
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AppKey {
    pub app_id: String,
    pub secret_key: String,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
pub struct ListenerOption {
    /// Ip to listen to.
    #[derivative(Default(value = "IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))"))]
    pub ip: IpAddr,
    /// Binding port.
    #[derivative(Default(value = "65530u16"))]
    pub port: u16,
    /// Domain for certs. Certs are stored in `assets/certs/`.
    #[derivative(Default(value = "String::from(\"local\")"))]
    pub domain: String,
    /// Using acme to issue the certs if the domain is not local.
    #[derivative(Default(value = "false"))]
    pub acme: bool,
    /// Force to enable https. When acme is true, tls must be true.
    #[derivative(Default(value = "false"))]
    pub tls: bool,
    /// For JWT Token encoding and decoding.    
    #[derivative(Default(value = "String::from(\"ai00s\")"))]
    pub slot: String,
    /// Whether the identifier is forced to pass even if JWT Token verification fails
    #[derivative(Default(value = "Some(false)"))]
    pub force_pass: Option<bool>,
    /// Token expiration time by second
    #[derivative(Default(value = "Some(86400u32)"))]
    pub expire_sec: Option<u32>,
    /// AppId with SecretKey pairs
    pub app_keys: Vec<AppKey>,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
#[serde(default)]
pub struct WebOption {
    #[derivative(Default(value = "\"assets/www/index.zip\".into()"))]
    pub path: PathBuf,
}
