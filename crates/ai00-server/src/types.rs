use std::path::PathBuf;

use ai00_core::ThreadRequest;
use flume::Sender;
use salvo::oapi::ToSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum Array<T> {
    #[default]
    None,
    Item(T),
    Vec(Vec<T>),
}

impl<T> From<Array<T>> for Vec<T> {
    fn from(value: Array<T>) -> Self {
        match value {
            Array::None => vec![],
            Array::Item(item) => vec![item],
            Array::Vec(vec) => vec,
        }
    }
}

#[derive(Clone)]
pub struct ThreadState {
    pub sender: Sender<ThreadRequest>,
    pub path: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JwtClaims {
    pub sid: String,
    pub exp: i64,
}
