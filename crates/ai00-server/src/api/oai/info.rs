use salvo::{
    oapi::{ToResponse, ToSchema},
    prelude::*,
};
use serde::Serialize;

use crate::{api::request_info, types::ThreadSender, SLEEP};

#[derive(Debug, Serialize, ToSchema)]
struct ModelChoice {
    object: String,
    id: String,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct ModelResponse {
    data: Vec<ModelChoice>,
}

/// Model name and id of the current choice.
#[endpoint(responses((status_code = 200, body = ModelResponse)))]
pub async fn models(depot: &mut Depot) -> Json<ModelResponse> {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.to_owned(), SLEEP).await;
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
