use std::time::Duration;

use anyhow::Result;
use futures_util::{Stream, StreamExt};
use salvo::oapi::{endpoint, ToSchema};
use salvo::sse::{self, SseEvent};
use salvo::{handler, Depot, Request};
use serde::Serialize;
use web_rwkv::model::ModelInfo;

use crate::api::{request_info, request_info_stream, try_request_info};
use crate::middleware::{ReloadRequest, RuntimeInfo, ThreadRequest, ThreadState};

#[derive(Debug, Clone, Serialize)]
pub struct InfoResponse {
    reload: ReloadRequest,
    model: ModelInfo,
}

#[handler]
pub async fn salvo_info(depot: &mut Depot) -> salvo::prelude::Json<InfoResponse> {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
    let RuntimeInfo { reload, model, .. } =
        request_info(sender.to_owned(), Duration::from_millis(500)).await;
    salvo::prelude::Json(InfoResponse { reload, model })
}

/// `/api/models/state`.
#[handler]
pub async fn salvo_state(depot: &mut Depot, res: &mut salvo::http::Response) {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
    let (info_sender, info_receiver) = flume::unbounded();
    let task = request_info_stream(sender.to_owned(), info_sender, Duration::from_millis(500));
    tokio::task::spawn(task);

    let stream = info_receiver.into_stream().map(|info| {
        let RuntimeInfo { reload, model, .. } = info;
        match serde_json::to_string(&InfoResponse { reload, model }) {
            Ok(json_text) => {
                log::info!("Present Json: {}", json_text.clone());
                SseEvent::default().json(json_text)
            }
            Err(err) => Err(err),
        }
    });

    salvo::sse::stream(res, stream);
}

/// `/api/models/load`.
#[handler]
pub async fn salvo_load<'a>(depot: &mut Depot, req: &mut Request) -> salvo::http::StatusCode {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
    let (result_sender, result_receiver) = flume::unbounded();
    let reload: ReloadRequest = req.parse_body().await.unwrap();
    let _ = sender.send(ThreadRequest::Reload {
        request: reload,
        sender: Some(result_sender),
    });
    match result_receiver.recv_async().await.unwrap() {
        true => salvo::http::StatusCode::OK,
        false => salvo::http::StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// `/api/models/unload`.
#[handler]
pub async fn salvo_unload(depot: &mut Depot) -> salvo::http::StatusCode {
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
    let _ = sender.send(ThreadRequest::Unload);
    while try_request_info(sender.clone()).await.is_ok() {}
    salvo::http::StatusCode::OK
}
