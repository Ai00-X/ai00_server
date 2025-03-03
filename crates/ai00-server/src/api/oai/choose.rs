use ai00_core::{GenerateKind, GenerateRequest, InputState, ThreadRequest, Token};
use futures_util::StreamExt;
use itertools::Itertools;
use salvo::{
    oapi::{extract::JsonBody, ToParameters, ToResponse, ToSchema},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{
    api::request_info,
    types::{Array, ThreadSender},
    SLEEP,
};

#[derive(Debug, Default, Clone, Deserialize, ToSchema, ToParameters)]
#[serde(default)]
#[salvo(schema(
    example = json!({
        "input": [
            "The Eiffel Tower is located in the city of"
        ],
        "choices": [
            " Paris",
            " Seattle",
            " San Francisco",
            " Shanghai"
        ],
        "calibrate": false,
        "state": "00000000-0000-0000-0000-000000000000"
    })
))]
struct ChooseRequest {
    input: Array<String>,
    choices: Vec<String>,
    calibrate: bool,
    state: InputState,
}

impl From<ChooseRequest> for GenerateRequest {
    fn from(value: ChooseRequest) -> Self {
        let ChooseRequest {
            input,
            choices,
            calibrate,
            state,
        } = value;
        Self {
            prompt: Vec::from(input).join(""),
            max_tokens: 1,
            kind: GenerateKind::Choose { choices, calibrate },
            state: state.into(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct ChooseData {
    object: String,
    index: usize,
    rank: usize,
    choice: String,
    perplexity: f32,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
#[salvo(schema(
    example = json!({
        "object": "list",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "data": [
            {
                "object": "choice",
                "index": 0,
                "rank": 0,
                "choice": " Paris",
                "perplexity": 0.031040953
            },
            {
                "object": "choice",
                "index": 2,
                "rank": 1,
                "choice": " San Francisco",
                "perplexity": 6.299065
            },
            {
                "object": "choice",
                "index": 3,
                "rank": 2,
                "choice": " Shanghai",
                "perplexity": 12.735298
            },
            {
                "object": "choice",
                "index": 1,
                "rank": 3,
                "choice": " Seattle",
                "perplexity": 14.686427
            }
        ]
    })
))]
struct ChooseResponse {
    object: String,
    model: String,
    data: Vec<ChooseData>,
}

/// Let the model choose from several options given a prompt.
#[endpoint(responses((status_code = 200, body = ChooseResponse)))]
pub async fn chooses(depot: &mut Depot, req: JsonBody<ChooseRequest>) -> Json<ChooseResponse> {
    let request = req.to_owned();
    let sender = depot.obtain::<ThreadSender>().unwrap();
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
