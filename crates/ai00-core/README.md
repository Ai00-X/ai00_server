# Ai00-Core

This is the core library of the [Ai00](https://github.com/Ai00-X/ai00_server) server. It provides the following functionalities:

- Model/LoRA/initial state loading with auto version detecting;
- Samplers;
- BNF integration;
- State caching;
- Session management.

The purpose of this crate is to expose a state-less native inference API.

## Guide

The first thing is to start the runtime, an async task that serves all background stuff (e.g., model runtime, caches, session queue):

```rust
use ai00_core::model_route;

let (sender, receiver) = flume::unbounded::<ThreadRequest>();
tokio::spawn(model_route(receiver));
```

Then users can communicate with the runtime by sending `ThreadRequest`s, which are commands that request the runtime to do all kinds of stuff.
Check its definition:

```rust
pub enum ThreadRequest {
    /// Acquire a list of current available adapters.
    Adapter(Sender<AdapterList>),
    /// Get the current runtime info.
    Info(Sender<RuntimeInfo>),
    /// Request the runtime to complement a prompt.
    Generate {
        request: Box<GenerateRequest>,
        tokenizer: Arc<Tokenizer>,
        sender: Sender<Token>,
    },
    /// Reload the runtime with custom config.
    Reload {
        request: Box<ReloadRequest>,
        sender: Option<Sender<bool>>,
    },
    /// Unload the runtime.
    Unload,
    /// Additionally load an initial state.
    StateLoad {
        request: reload::State,
        sender: Option<Sender<bool>>,
    },
    /// Unload an initial state given its id.
    StateUnload(StateId),
    /// Save the current model with config.
    Save {
        request: SaveRequest,
        sender: Sender<bool>,
    },
}
```