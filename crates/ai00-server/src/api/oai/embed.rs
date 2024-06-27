use salvo::{
    oapi::{extract::JsonBody, ToParameters, ToResponse, ToSchema},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::Tokenizer;
use fastembed::EmbeddingModel;
use log::info;
use crate::{EMBEDCONFIG, EMBEDMODEL};
use std::{env,fs};

#[derive(Debug, Default, Clone, Deserialize, ToSchema, ToParameters)]
#[serde(default)]
pub struct EmbedRequest {
    input: String,
    mode_name: String,
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

pub fn models_list() -> Vec<ModelInfo> {
    let models_list = vec![
        ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2,
            model_code: String::from("Qdrant/all-MiniLM-L6-v2-onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2Q,
            model_code: String::from("Xenova/all-MiniLM-L6-v2"),
        },
        ModelInfo {
            model: EmbeddingModel::BGEBaseENV15,
            model_code: String::from("Xenova/bge-base-en-v1.5"),

        },
        ModelInfo {
            model: EmbeddingModel::BGEBaseENV15Q,
            model_code: String::from("Qdrant/bge-base-en-v1.5-onnx-Q"),
        },
        ModelInfo {
            model: EmbeddingModel::BGELargeENV15,
            model_code: String::from("Xenova/bge-large-en-v1.5"),
        },
        ModelInfo {
            model: EmbeddingModel::BGELargeENV15Q,
            model_code: String::from("Qdrant/bge-large-en-v1.5-onnx-Q"),
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallENV15,
            model_code: String::from("Xenova/bge-small-en-v1.5"),
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallENV15Q,
            model_code: String::from("Qdrant/bge-small-en-v1.5-onnx-Q"),
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV1,
            model_code: String::from("nomic-ai/nomic-embed-text-v1"),
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV15,
            model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV15Q,
            model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMiniLML12V2Q,
            model_code: String::from("Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"),
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMiniLML12V2,
            model_code: String::from("Xenova/paraphrase-multilingual-MiniLM-L12-v2"),
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMpnetBaseV2,
            model_code: String::from("Xenova/paraphrase-multilingual-mpnet-base-v2"),
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallZHV15,
            model_code: String::from("Xenova/bge-small-zh-v1.5"),
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Small,
            model_code: String::from("intfloat/multilingual-e5-small"),
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Base,
            model_code: String::from("intfloat/multilingual-e5-base"),
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Large,
            model_code: String::from("Qdrant/multilingual-e5-large-onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::MxbaiEmbedLargeV1,
            model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
        },
        ModelInfo {
            model: EmbeddingModel::MxbaiEmbedLargeV1Q,
             model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
        },
    ];

    // TODO: Use when out in stable
    // assert_eq!(
    //     std::mem::variant_count::<EmbeddingModel>(),
    //     models_list.len(),
    //     "models::models() is not exhaustive"
    // );

    models_list
}
/// Data struct about the available models
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model: EmbeddingModel,
    pub model_code: String,
}

trait EmbeddingModelExt {
    fn from_name(name: &str) -> Self;
}

impl EmbeddingModelExt for EmbeddingModel {
    fn from_name(name: &str) -> Self {
        match name {
            "AllMiniLML6V2" => EmbeddingModel::AllMiniLML6V2,
            "AllMiniLML6V2Q" => EmbeddingModel::AllMiniLML6V2Q,
            "BGEBaseENV15" => EmbeddingModel::BGEBaseENV15,
            "BGEBaseENV15Q" => EmbeddingModel::BGEBaseENV15Q,
            "BGELargeENV15" => EmbeddingModel::BGELargeENV15,
            "BGELargeENV15Q" => EmbeddingModel::BGELargeENV15Q,
            "BGESmallENV15" => EmbeddingModel::BGESmallENV15,
            "BGESmallENV15Q" => EmbeddingModel::BGESmallENV15Q,
            "NomicEmbedTextV1" => EmbeddingModel::NomicEmbedTextV1,
            "NomicEmbedTextV15" => EmbeddingModel::NomicEmbedTextV15,
            "NomicEmbedTextV15Q" => EmbeddingModel::NomicEmbedTextV15Q,
            "ParaphraseMLMiniLML12V2" => EmbeddingModel::ParaphraseMLMiniLML12V2,
            "ParaphraseMLMiniLML12V2Q" => EmbeddingModel::ParaphraseMLMiniLML12V2,
            "ParaphraseMLMpnetBaseV2" => EmbeddingModel::ParaphraseMLMpnetBaseV2,
            "BGESmallZHV15" => EmbeddingModel::BGESmallZHV15,
            "MultilingualE5Small" => EmbeddingModel::MultilingualE5Small,
            "MultilingualE5Base" => EmbeddingModel::MultilingualE5Base,
            "MultilingualE5Large" => EmbeddingModel::MultilingualE5Large,
            "MxbaiEmbedLargeV1" => EmbeddingModel::MxbaiEmbedLargeV1,
            "MxbaiEmbedLargeV1Q" => EmbeddingModel::MxbaiEmbedLargeV1Q,
            _ => panic!("Unsupported model name"),
        }
        //没有match
    }
}

/// Generate a embedding vector for the given text, with layer number specified for producing the embedding.
#[endpoint(responses((status_code = 200, body = EmbedResponse)))]
pub async fn embeds(_depot: &mut Depot, req: JsonBody<EmbedRequest>) -> Json<EmbedResponse> {


    let future = async move {

        let models_list = models_list();
        //从模型列表models_list 中获取 models_list[].model 和 model 匹配的 models_list[].model_code
        let identifier = models_list
            .iter()
            .find(|m| m.model == EmbeddingModel::from_name(&EMBEDCONFIG.model_name))
            .map(|m| m.model_code.clone())
            .unwrap();

        env::set_var("HF_ENDPOINT", EMBEDCONFIG.endpoint.clone());
        env::set_var("HF_HOME", EMBEDCONFIG.home_path.clone());
        let api = hf_hub::api::sync::Api::new().unwrap();
        info!("identifier:{}", identifier);
        let filename = api
                .model(identifier)
                .get("tokenizer.json")
                .unwrap();


        let tokenizers = Tokenizer::from_file(filename);
        let input = req.input.clone();
        let max_tokens = 10;

        let mut embeddings_result: Vec<EmbedsData> = Vec::new();
        match tokenizers {
            Ok(tokenizer) => {
                let splitter =
                    TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));
                // 使用splitter
                let chunks = splitter.chunks(&input);
                for chunk in chunks {
                    let pp = [chunk];
                    let embedding_result = EMBEDMODEL
                        .embed(Vec::from(pp), None)
                        .expect("Failed to get embedding");

                    let embeds_data = EmbedsData {
                        chunk: chunk.to_owned(),
                        embeds: embedding_result.clone(),
                    };
                    embeddings_result.push(embeds_data);
                }
            }
            Err(error) => {
                // 处理错误
                println!("Error initializing tokenizer: {}", error);
            }
        }

        

 // Use expect to handle error
      //  print!("Embedding result: {:?}", embedding_result);

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
