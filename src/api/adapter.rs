use axum::{extract::State, Json};
use serde::Serialize;

use crate::{AdapterList, ThreadRequest, ThreadState};

#[derive(Debug, Default, Clone, Serialize)]
pub struct AdapterResponse(Vec<String>);

/// `/api/adapters`.
pub async fn adapters(State(ThreadState(sender)): State<ThreadState>) -> Json<AdapterResponse> {
    let (list_sender, list_receiver) = flume::unbounded();
    let _ = sender.send(ThreadRequest::Adapter(list_sender));
    let AdapterList(list) = list_receiver.recv().unwrap_or_default();
    Json(AdapterResponse(list))
}
