use ai00_core::{GenerateKind, GenerateRequest, ThreadRequest, Token};
use futures_util::StreamExt;
use salvo::{
    oapi::{extract::JsonBody, ToParameters, ToResponse, ToSchema},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{
    api::request_info,
    types::{Array, ThreadState},
    SLEEP,
};

#[derive(Debug, Default, Clone, Deserialize, ToSchema, ToParameters)]
#[serde(default)]
pub struct ChooseRequest {
    input: Array<String>,
    choices: Vec<String>,
}

impl From<ChooseRequest> for GenerateRequest {
    fn from(value: ChooseRequest) -> Self {
        let ChooseRequest { input, choices } = value;
        Self {
            prompt: Vec::from(input).join(""),
            max_tokens: 1,
            kind: GenerateKind::Choose { choices },
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct ChooseResponse {
    object: String,
    model: String,
    perplexities: Vec<f32>,
}

#[endpoint]
pub async fn chooses(depot: &mut Depot, req: JsonBody<ChooseRequest>) -> Json<ChooseResponse> {
    let request = req.to_owned();
    let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Generate {
        request: Box::new(request.into()),
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let mut perplexities = Vec::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        if let Token::Choose(ppl) = token {
            perplexities = ppl;
            break;
        }
    }

    Json(ChooseResponse {
        object: "list".into(),
        model: model_name,
        perplexities,
    })
}
