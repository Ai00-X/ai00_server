use ai00_core::{GenerateKind, GenerateRequest, InputState, ThreadRequest, Token, TokenCounter};
use futures_util::StreamExt;
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
        ]
    })
))]
struct StateRequest {
    input: Array<String>,
    state: InputState,
}

impl From<StateRequest> for GenerateRequest {
    fn from(value: StateRequest) -> Self {
        let StateRequest { input, state } = value;
        Self {
            prompt: Vec::from(input).join(""),
            max_tokens: 1,
            kind: GenerateKind::State,
            state: state.into(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct StateData {
    object: String,
    index: usize,
    data: Vec<f32>,
    shape: [usize; 4],
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct StateResponse {
    object: String,
    model: String,
    data: Vec<StateData>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

/// Generate the model state for the given text.
#[endpoint(responses((status_code = 200, body = StateResponse)))]
pub async fn states(depot: &mut Depot, req: JsonBody<StateRequest>) -> Json<StateResponse> {
    let request = req.to_owned();
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Generate {
        request: Box::new(request.into()),
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let mut token_counter = TokenCounter::default();
    let mut data = Vec::new();
    let mut shape = [0, 0, 0, 0];
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Stop(_, counter) => token_counter = counter,
            Token::Embed(_data, _shape) => {
                data = _data;
                shape = _shape;
                break;
            }
            _ => {}
        }
    }

    Json(StateResponse {
        object: "list".into(),
        model: model_name,
        data: vec![StateData {
            object: "states".into(),
            index: 0,
            data,
            shape,
        }],
        counter: token_counter,
    })
}
