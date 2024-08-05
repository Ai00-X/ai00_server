use ai00_core::{AdapterList, ThreadRequest};
use salvo::prelude::*;

use crate::types::ThreadSender;

/// `/api/adapters`.
#[endpoint]
pub async fn adapters(depot: &mut Depot) -> Json<Vec<String>> {
    let (list_sender, list_receiver) = flume::unbounded();
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let _ = sender.send(ThreadRequest::Adapter(list_sender));
    let AdapterList(list) = list_receiver.recv_async().await.unwrap_or_default();
    Json(list)
}
