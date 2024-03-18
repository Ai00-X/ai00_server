use salvo::{handler, Depot};

use crate::middleware::{AdapterList, ThreadRequest, ThreadState};

/// `/api/adapters`.
#[handler]
pub async fn salvo_adapters(depot: &mut Depot) -> salvo::prelude::Json<Vec<String>> {
    let (list_sender, list_receiver) = flume::unbounded();
    let ThreadState(sender) = depot.obtain::<ThreadState>().unwrap();
    let _ = sender.send(ThreadRequest::Adapter(list_sender));
    let AdapterList(list) = list_receiver.recv_async().await.unwrap_or_default();
    salvo::prelude::Json(list)
}
