use salvo::{
    macros::handler,
    oapi::{endpoint, ToSchema},
    Depot,
};
use serde::Serialize;
use std::time::Duration;

use crate::{api::request_info, ThreadState};

#[derive(Debug, Serialize, ToSchema)]
struct ModelChoice {
    object: String,
    id: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelResponse {
    data: Vec<ModelChoice>,
}

/// This method is getting the current model's information 
///
/// The responses the model name and id by current choice.
#[endpoint]
pub async fn salvo_oai_models(depot: &mut Depot) -> salvo::prelude::Json<ModelResponse> {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
    let info = request_info(sender.to_owned(), Duration::from_secs(1)).await;
    let model_name = info
        .reload
        .model_path
        .file_stem()
        .map(|stem| stem.to_string_lossy())
        .unwrap_or_default();

    log::info!("Process this.");
    salvo::prelude::Json(ModelResponse {
        data: vec![ModelChoice {
            object: "models".into(),
            id: model_name.into(),
        }],
    })
}
