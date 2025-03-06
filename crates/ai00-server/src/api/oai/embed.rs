use std::sync::Arc;

use anyhow::Result;
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
    embed: Vec<Vec<f32>>,
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
#[salvo(schema(
    example = json!({
        "input": "The Eiffel Tower is located in the city of",
        "max_tokens": 510,
        "prefix": "query:"
    })
))]
struct EmbedRequest {
    input: String,
    #[derivative(Default(value = "510"))]
    max_tokens: usize,
    #[derivative(Default(value = "\"query:\".into()"))]
    prefix: String,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct EmbedResponse {
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

    if req.input.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let model = embed.info.model_code.clone();

    let task = move || -> Result<_> {
        let max_tokens = req.max_tokens.clamp(1, 510);
        let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(&embed.tokenizer));

        let mut data = vec![];
        let chunks = splitter.chunks(&req.input);
        for chunk in chunks {
            let text = format!("{}{}", req.prefix, chunk);
            let embed = embed.model.embed(vec![text], None)?;
            let chunk = chunk.to_string();
            data.push(ChunkData { chunk, embed });
        }

        Ok(data)
    };

    let handle = tokio::task::spawn_blocking(task);
    match handle.await {
        Ok(Ok(data)) => Ok(Json(EmbedResponse {
            object: "embeds".into(),
            model,
            data: vec![EmbedData {
                object: "embed".into(),
                index: 0,
                chunks: data,
            }],
        })),
        _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}
