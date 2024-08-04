use std::{path::PathBuf, sync::Arc};

use derivative::Derivative;
use salvo::{
    oapi::{extract::JsonBody, ToParameters, ToResponse, ToSchema},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use text_splitter::{ChunkConfig, TextSplitter};

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct ChunkData {
    chunk: String,
    embeds: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct EmbedData {
    object: String,
    index: usize,
    chunks: Vec<ChunkData>,
}

#[derive(Debug, Derivative, Clone, Deserialize, ToSchema, ToParameters)]
#[derivative(Default)]
#[serde(default)]
pub struct EmbedRequest {
    #[derivative(Default(value = "\"Ai00 is all your need!\".into()"))]
    input: String,
    #[derivative(Default(value = "510"))]
    max_tokens: usize,
    #[derivative(Default(value = "\"query:\".into()"))]
    prefix: String,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct EmbedResponse {
    object: String,
    model: String,
    data: Vec<EmbedData>,
}

/// Generate a embedding vector for the given text, with layer number specified for producing the embedding.
#[endpoint(responses((status_code = 200, body = EmbedResponse)))]
pub async fn embeds(
    depot: &mut Depot,
    req: JsonBody<EmbedRequest>,
) -> Result<Json<EmbedResponse>, StatusCode> {
    let embed = depot
        .get::<Option<Arc<crate::TextEmbed>>>("embed")
        .unwrap()
        .clone();

    let Some(embed) = embed else {
        return Err(StatusCode::BAD_REQUEST);
    };

    if req.input.as_str().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let task = move || {
        let max_tokens = req.max_tokens.clamp(1, 510);
    };

    let handle = tokio::task::spawn_blocking(task);

    // let future = async move {
    //     let mut input = req.input.clone();
    //     let mut max_tokens = req.max_tokens;
    //     max_tokens = max_tokens.clamp(1, 510);

    //     if input.is_empty() {
    //         input = "Ai00 is all your need!".into();
    //     }

    //     let mut result = vec![];

    //     let tokenizer = EMBEDTOKENIZERS.clone();

    //     let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));

    //     let chunks = splitter.chunks(&input);
    //     for chunk in chunks {
    //         let chunk_b = format!("{}{}", req.prefix, chunk);

    //         let pp = [chunk_b];

    //         let embedding_result = EMBEDMODEL
    //             .embed(Vec::from(pp), None)
    //             .expect("Failed to get embedding");

    //         let embeds_data = ChunkData {
    //             chunk: chunk.to_owned(),
    //             embeds: embedding_result.clone(),
    //         };
    //         result.push(embeds_data);
    //     }

    //     result
    // };

    // let embedding_result = tokio::spawn(future).await.expect("spawn failed");

    // Json(EmbedResponse {
    //     object: "embeds".into(),
    //     model: "multilingual-e5-small".into(),
    //     data: vec![EmbedData {
    //         object: "embed".into(),
    //         index: 0,
    //         chunks: embedding_result,
    //     }],
    // })

    todo!()
}
