use std::time::Duration;

use salvo::oapi::{ToResponse, ToSchema};
use serde::Serialize;

pub use private::models;

#[derive(Debug, Serialize, ToSchema)]
struct ModelChoice {
    object: String,
    id: String,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct ModelResponse {
    data: Vec<ModelChoice>,
}

#[cfg(feature = "axum-api")]
mod private {
    use axum::{extract::State, Json};

    use super::*;
    use crate::{api::request_info, ThreadState};

    /// `/api/oai/models`, `/api/oai/v1/models`.
    pub async fn models(State(ThreadState(sender)): State<ThreadState>) -> Json<ModelResponse> {
        let info = request_info(sender, Duration::from_secs(1)).await;
        let model_name = info
            .reload
            .model_path
            .file_stem()
            .map(|stem| stem.to_string_lossy())
            .unwrap_or_default();

        Json(ModelResponse {
            data: vec![ModelChoice {
                object: "models".into(),
                id: model_name.into(),
            }],
        })
    }
}

#[cfg(feature = "salvo-api")]
mod private {
    use salvo::{oapi::endpoint, prelude::*};

    use super::*;
    use crate::{api::request_info, ThreadState};

    /// Model name and id of the current choice.
    #[endpoint]
    pub async fn models(depot: &mut Depot) -> Json<ModelResponse> {
        let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
        let info = request_info(sender.to_owned(), Duration::from_secs(1)).await;
        let model_name = info
            .reload
            .model_path
            .file_stem()
            .map(|stem| stem.to_string_lossy())
            .unwrap_or_default();

        Json(ModelResponse {
            data: vec![ModelChoice {
                object: "models".into(),
                id: model_name.into(),
            }],
        })
    }
}
