use futures_util::StreamExt;
use serde::Serialize;
use web_rwkv::model::ModelInfo;

use super::{request_info, request_info_stream, try_request_info};
use crate::{build_path, middleware::ReloadRequest};

pub use private::{info, load, state, unload};

#[derive(Debug, Clone, Serialize)]
pub struct InfoResponse {
    reload: ReloadRequest,
    model: ModelInfo,
}

#[cfg(feature = "axum-api")]
mod private {
    use std::time::Duration;

    use anyhow::Result;
    use axum::{
        extract::State,
        http::StatusCode,
        response::{sse::Event, Sse},
        Json,
    };
    use futures_util::Stream;

    use super::*;
    use crate::middleware::{ReloadRequest, RuntimeInfo, ThreadRequest, ThreadState};

    /// `/api/models/info`.
    pub async fn info(State(ThreadState { sender, .. }): State<ThreadState>) -> Json<InfoResponse> {
        let RuntimeInfo { reload, model, .. } =
            request_info(sender, Duration::from_millis(500)).await;
        Json(InfoResponse { reload, model })
    }

    /// `/api/models/state`.
    pub async fn state(
        State(ThreadState { sender, .. }): State<ThreadState>,
    ) -> Sse<impl Stream<Item = Result<Event>>> {
        let (info_sender, info_receiver) = flume::unbounded();
        let task = request_info_stream(sender, info_sender, Duration::from_millis(500));
        tokio::task::spawn(task);

        let stream = info_receiver.into_stream().map(|info| {
            let RuntimeInfo { reload, model, .. } = info;
            let json = serde_json::to_string(&InfoResponse { reload, model })?;
            Ok(Event::default().data(json))
        });
        Sse::new(stream)
    }

    /// `/api/models/load`.
    pub async fn load(
        State(ThreadState { sender, model_path }): State<ThreadState>,
        Json(mut request): Json<ReloadRequest>,
    ) -> StatusCode {
        let (result_sender, result_receiver) = flume::unbounded();

        // make sure that we are not visiting un-permitted path.
        request.model_path = match build_path(model_path, request.model_path) {
            Ok(path) => path,
            Err(_) => return StatusCode::NOT_FOUND,
        };

        let _ = sender.send(ThreadRequest::Reload {
            request: Box::new(request),
            sender: Some(result_sender),
        });
        match result_receiver.recv_async().await.unwrap() {
            true => StatusCode::OK,
            false => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// `/api/models/unload`.
    pub async fn unload(State(ThreadState { sender, .. }): State<ThreadState>) -> StatusCode {
        let _ = sender.send(ThreadRequest::Unload);
        while try_request_info(sender.clone()).await.is_ok() {}
        StatusCode::OK
    }
}

#[cfg(feature = "salvo-api")]
mod private {
    use std::time::Duration;

    use salvo::prelude::*;

    use super::*;
    use crate::middleware::{RuntimeInfo, ThreadRequest, ThreadState};

    #[handler]
    pub async fn info(depot: &mut Depot) -> Json<InfoResponse> {
        let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
        let RuntimeInfo { reload, model, .. } =
            request_info(sender.to_owned(), Duration::from_millis(500)).await;
        Json(InfoResponse { reload, model })
    }

    /// `/api/models/state`.
    #[handler]
    pub async fn state(depot: &mut Depot, res: &mut Response) {
        let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
        let (info_sender, info_receiver) = flume::unbounded();
        let task = request_info_stream(sender.to_owned(), info_sender, Duration::from_millis(500));
        tokio::task::spawn(task);

        let stream = info_receiver.into_stream().map(|_info| {
            let RuntimeInfo { reload, model, .. } = _info;
            match serde_json::to_string(&InfoResponse { reload, model }) {
                Ok(json) => SseEvent::default().json(json),
                Err(err) => Err(err),
            }
        });

        salvo::sse::stream(res, stream);
    }

    /// `/api/models/load`.
    #[handler]
    pub async fn load(depot: &mut Depot, req: &mut Request) -> StatusCode {
        let ThreadState { sender, model_path } = depot.obtain::<ThreadState>().unwrap();
        let (result_sender, result_receiver) = flume::unbounded();
        let mut request: ReloadRequest = req.parse_body().await.unwrap();

        // make sure that we are not visiting un-permitted path.
        request.model_path = match build_path(model_path, &request.model_path) {
            Ok(path) => path,
            Err(_) => return StatusCode::NOT_FOUND,
        };
        for lora in request.lora.iter_mut() {
            lora.path = match build_path(model_path, &lora.path) {
                Ok(path) => path,
                Err(_) => return StatusCode::NOT_FOUND,
            }
        }

        let _ = sender.send(ThreadRequest::Reload {
            request: Box::new(request),
            sender: Some(result_sender),
        });
        match result_receiver.recv_async().await.unwrap() {
            true => StatusCode::OK,
            false => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// `/api/models/unload`.
    #[handler]
    pub async fn unload(depot: &mut Depot) -> StatusCode {
        let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
        let _ = sender.send(ThreadRequest::Unload);
        while try_request_info(sender.clone()).await.is_ok() {}
        StatusCode::OK
    }
}
