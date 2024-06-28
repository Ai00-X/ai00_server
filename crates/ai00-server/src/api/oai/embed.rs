use derivative::Derivative;
use salvo::{
    oapi::{extract::JsonBody, ToParameters, ToResponse, ToSchema},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{EMBEDMODEL, EMBEDTOKENIZERS};
use text_splitter::{ChunkConfig, TextSplitter};

#[derive(Debug, Default, Derivative, Clone, Deserialize, ToSchema, ToParameters)]
#[serde(default)]
pub struct EmbedRequest {
    #[derivative(Default(value = "Ai00 is all your need!"))]
    input: String,
    #[derivative(Default(value = "510"))]
    max_tokens: usize,
    #[derivative(Default(value = "query:"))]
    prefix: String,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct EmbedsData {
    chunk: String,
    embeds: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct EmbedData {
    object: String,
    index: usize,
    chunks: Vec<EmbedsData>,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct EmbedResponse {
    object: String,
    model: String,
    data: Vec<EmbedData>,
}

/// Generate a embedding vector for the given text, with layer number specified for producing the embedding.
#[endpoint(responses((status_code = 200, body = EmbedResponse)))]
pub async fn embeds(_depot: &mut Depot, req: JsonBody<EmbedRequest>) -> Json<EmbedResponse> {
    let future = async move {
        let mut input = req.input.clone();
        let mut max_tokens = req.max_tokens;
        max_tokens = max_tokens.clamp(1, 510);

        if input.is_empty() {
            input = "Ai00 is all your need!".into();
        }

        let mut embeddings_result: Vec<EmbedsData> = Vec::new();

        let tokenizer = EMBEDTOKENIZERS.clone();

        let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));

        let chunks = splitter.chunks(&input);
        for chunk in chunks {
            //在每个 chunk 之前加入 前缀
            let chunk_b = format!("{}{}", req.prefix, chunk);

            let pp = [chunk_b];

            let embedding_result = EMBEDMODEL
                .embed(Vec::from(pp), None)
                .expect("Failed to get embedding");

            let embeds_data = EmbedsData {
                chunk: chunk.to_owned(),
                embeds: embedding_result.clone(),
            };
            embeddings_result.push(embeds_data);
        }

        embeddings_result
    };

    let embedding_result = tokio::spawn(future).await.expect("spawn failed");

    Json(EmbedResponse {
        object: "embeds".into(),
        model: "multilingual-e5-small".into(),
        data: vec![EmbedData {
            object: "embed".into(),
            index: 0,
            chunks: embedding_result,
        }],
    })
}
