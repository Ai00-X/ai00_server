use salvo::prelude::*;

use crate::middleware::{AdapterList, ThreadRequest, ThreadState};

/// `/api/adapters`.
#[endpoint]
pub async fn adapters(depot: &mut Depot) -> Json<Vec<String>> {
    let (list_sender, list_receiver) = flume::unbounded();
    let ThreadState { sender, .. } = depot.obtain::<ThreadState>().unwrap();
    let _ = sender.send(ThreadRequest::Adapter(list_sender));
    let AdapterList(list) = list_receiver.recv_async().await.unwrap_or_default();
    Json(list)
}
