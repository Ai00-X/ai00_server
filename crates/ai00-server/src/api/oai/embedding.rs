use ai00_core::{GenerateRequest, ThreadRequest, Token, TokenCounter};
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
pub struct EmbeddingRequest {
    input: Array<String>,
    embed_layer: usize,
}

impl From<EmbeddingRequest> for GenerateRequest {
    fn from(value: EmbeddingRequest) -> Self {
        let EmbeddingRequest { input, embed_layer } = value;
        Self {
            prompt: Vec::from(input).join(""),
            max_tokens: 1,
            embed: true,
            embed_layer,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct EmbeddingData {
    object: String,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct EmbeddingResponse {
    object: String,
    model: String,
    data: Vec<EmbeddingData>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

/// Generate a embedding vector for the given text, with layer number specified for producing the embedding.
#[endpoint]
pub async fn embeddings(
    depot: &mut Depot,
    req: JsonBody<EmbeddingRequest>,
) -> Json<EmbeddingResponse> {
    let request = req.to_owned(); // req.parse_json::<EmbeddingRequest>().await.unwrap();
    let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let (token_sender, token_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Generate {
        request: Box::new(request.into()),
        tokenizer: info.tokenizer,
        sender: token_sender,
    });

    let mut token_counter = TokenCounter::default();
    let mut embedding = Vec::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Stop(_, counter) => token_counter = counter,
            Token::Embed(emb) => {
                embedding = emb;
                break;
            }
            _ => {}
        }
    }

    Json(EmbeddingResponse {
        object: "list".into(),
        model: model_name,
        data: vec![EmbeddingData {
            object: "embedding".into(),
            index: 0,
            embedding,
        }],
        counter: token_counter,
    })
}
