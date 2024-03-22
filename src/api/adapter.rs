pub use private::adapters;

#[cfg(feature = "axum-api")]
mod private {
    use axum::{extract::State, Json};

    use crate::middleware::{AdapterList, ThreadRequest, ThreadState};

    /// `/api/adapters`.
    pub async fn adapters(
        State(ThreadState { sender, .. }): State<ThreadState>,
    ) -> Json<Vec<String>> {
        let (list_sender, list_receiver) = flume::unbounded();
        let _ = sender.send(ThreadRequest::Adapter(list_sender));
        let AdapterList(list) = list_receiver.recv_async().await.unwrap_or_default();
        Json(list)
    }
}

#[cfg(feature = "salvo-api")]
mod private {
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
}
