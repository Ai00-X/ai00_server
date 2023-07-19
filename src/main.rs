use anyhow::{anyhow, Result};
use itertools::Itertools;
use memmap2::Mmap;
use std::{
    env,
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
    time::Instant,
};
use warp::Filter;
use warp::sse::ServerSentEvent;
use futures::channel::mpsc;
use futures::Stream;
use futures::SinkExt;
use web_rwkv::{Environment, Model, Tokenizer};

// The code that defines softmax and sample remains unchanged...

async fn create_environment() -> Result<Environment> {
    // The code remains unchanged...
}

async fn load_tokenizer(path: &str) -> Result<Tokenizer> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(Tokenizer::new(&contents)?)
}

async fn load_model(env: Arc<Environment>, path: &str) -> Result<Model> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    let model = Model::from_bytes(&map, env)?;
    println!("{:#?}", model.info);
    #[cfg(target_arch = "wasm32")]
    log::info!("{:#?}", model.info);
    Ok(model)
}

async fn run_model(
    prompt: String,
    model: Arc<Model>,
    tokenizer: Arc<Tokenizer>,
    state: Arc<web_rwkv::StateBuffer>,
    num_tokens: usize,
    mut tx: mpsc::Sender<ServerSentEvent>,
) -> Result<()> {
  let mut tokens = tokenizer.encode(prompt.as_bytes())?;
  
  let mut start = Instant::now();
  for index in 0..=num_tokens {
      let buffer = model.create_buffer(&tokens);
  
      #[cfg(not(target_arch = "wasm32"))]
      let logits = model.run(&buffer, &state)?;
  
      #[cfg(target_arch = "wasm32")]
      let logits = model.run_async(&buffer, &state).await?;

      let probs = softmax(&logits);
  
      let token = sample(&probs, 0.5);
      let word = String::from_utf8(tokenizer.decode(&[token])?)?;
      tokens = vec![token];

      tx.send(ServerSentEvent::data(word.clone())).await.unwrap();
  
      if index == 0 {
          start = Instant::now();
      }
  }

  Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        panic!("Please provide two command-line arguments for model and tokenizer files respectively.")
    }
    let (model_path, tokenizer_path) = (&args[1], &args[2]);
    let (tx, rx) = mpsc::unbounded();
  
    #[cfg(not(target_arch = "wasm32"))]
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let env = Arc::new(runtime.block_on(create_environment().unwrap()));
    let tokenizer = Arc::new(runtime.block_on(load_tokenizer(tokenizer_path)).unwrap());
    let model = Arc::new(runtime.block_on(load_model(env.clone(), model_path)).unwrap());
  
    let post_chat = warp::post()
        .and(warp::path!("chat" / "completion"))
        .and(warp::body::json())
        .and(warp::sse())
        .map(move |prompt: String, _sse| {
            let tx = tx.clone();
            let model = model.clone();
            let tokenizer = tokenizer.clone();
            let state = model.create_state();
            tokio::spawn(run_model(prompt, model, tokenizer, state, 100, tx));
            warp::sse::reply(warp::sse::keep(rx, None))
        });

    let routes = post_chat;

    warp::serve(routes).run(([127, 0, 0, 1], 3030));
}
