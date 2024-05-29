use ai00_core::{GenerateKind, GenerateRequest, ThreadRequest, Token};
use futures_util::StreamExt;
use itertools::Itertools;
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
pub struct ChooseData {
    object: String,
    index: usize,
    rank: usize,
    choice: String,
    perplexity: f32,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct ChooseResponse {
    object: String,
    model: String,
    data: Vec<ChooseData>,
}

#[endpoint]
pub async fn chooses(depot: &mut Depot, req: JsonBody<ChooseRequest>) -> Json<ChooseResponse> {
    let request = req.to_owned();
    let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let choices = request.choices.clone();

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

    let data = perplexities
        .into_iter()
        .zip(choices.into_iter())
        .enumerate()
        .sorted_by(|(_, (x, _)), (_, (y, _))| x.total_cmp(y))
        .enumerate()
        .map(|(rank, (index, (perplexity, choice)))| ChooseData {
            object: "choice".into(),
            rank,
            index,
            choice,
            perplexity,
        })
        .collect();

    Json(ChooseResponse {
        object: "list".into(),
        model: model_name,
        data,
    })
}
