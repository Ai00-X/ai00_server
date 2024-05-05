# 💯AI00 RWKV Server
<p align='center'>
<image src="docs/public/logo.gif" />
    
</p>

<div align="center"> 
    
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
[![Rust Version](https://img.shields.io/badge/Rust-1.75.0+-blue)](https://releases.rs/docs/1.75.0)
![PRs welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen)     
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->



[English](README.md) | [中文](README_zh.md) 

---

<div align="left">
    
`AI00 RWKV Server` is an inference API server for the [`RWKV` language model](https://github.com/BlinkDL/ChatRWKV) based upon the [`web-rwkv`](https://github.com/cryscan/web-rwkv) inference engine.

It supports `VULKAN` parallel and concurrent batched inference and can run on all GPUs that support `VULKAN`. No need for Nvidia cards!!! AMD cards and even integrated graphics can be accelerated!!!

No need for bulky `pytorch`, `CUDA` and other runtime environments, it's compact and ready to use out of the box!

Compatible with OpenAI's ChatGPT API interface.

100% open source and commercially usable, under the MIT license.

If you are looking for a fast, efficient, and easy-to-use LLM API server, then `AI00 RWKV Server` is your best choice. It can be used for various tasks, including chatbots, text generation, translation, and Q&A.

Join the `AI00 RWKV Server` community now and experience the charm of AI!

QQ Group for communication: 30920262

### 💥Features

*   Based on the `RWKV` model, it has high performance and accuracy
*   Supports `VULKAN` inference acceleration, you can enjoy GPU acceleration without the need for `CUDA`! Supports AMD cards, integrated graphics, and all GPUs that support `VULKAN`
*   No need for bulky `pytorch`, `CUDA` and other runtime environments, it's compact and ready to use out of the box!
*   Compatible with OpenAI's ChatGPT API interface

### ⭕Usages

*   Chatbots
*   Text generation
*   Translation
*   Q&A
*   Any other tasks that LLM can do

### 👻Other

*   Based on the [web-rwkv](https://github.com/cryscan/web-rwkv) project
*   Model download: [V5](https://huggingface.co/cgisky/AI00_RWKV_V5) or [V6](https://huggingface.co/cgisky/ai00_rwkv_x060)

## Installation, Compilation, and Usage

### 📦Download Pre-built Executables

1.  Directly download the latest version from [Release](https://github.com/cgisky1980/ai00_rwkv_server/releases)
    
2.  After [downloading the model](#👻other), place the model in the `assets/models/` path, for example, `assets/models/RWKV-x060-World-3B-v2-20240228-ctx4096.st`

3.  Optionally modify [`assets/Config.toml`](./assets/Config.toml) for model configurations like model path, quantization layers, etc.
    
4.  Run in the command line
    
    ```bash
    $ ./ai00_rwkv_server
    ```
    
5.  Open the browser and visit the WebUI [`https://localhost:65530`](https://localhost:65530)
    

### 📜(Optional) Build from Source

1.  [Install Rust](https://www.rust-lang.org/)
    
2.  Clone this repository
    
    ```bash
    $ git clone https://github.com/cgisky1980/ai00_rwkv_server.git
    $ cd ai00_rwkv_server
    ```
    
3.  After [downloading the model](#👻other), place the model in the `assets/models/` path, for example, `assets/models/RWKV-x060-World-3B-v2-20240228-ctx4096.st`
    
4.  Compile
    
    ```bash
    $ cargo build --release
    ```
    
5.  After compilation, run
    
    ```bash
    $ cargo run --release
    ```
    
6.  Open the browser and visit the WebUI [`https://localhost:65530`](https://localhost:65530)

### 📒Convert the Model

It only supports Safetensors models with the `.st` extension now. Models saved with the `.pth` extension using torch need to be converted before use.

1. [Download the `.pth` model](https://huggingface.co/BlinkDL)

2. In the [Release](https://github.com/cgisky1980/ai00_rwkv_server/releases) you could find an executable called `converter`. Run
  
  ```bash
  $ ./converter --input /path/to/model.pth
  ```
  
3. If you are building from source, run
  
  ```bash
  $ cargo run --release --bin converter -- --input /path/to/model.pth
  ```

4. Just like the steps mentioned above, place the model in the `.st` model in the `assets/models/` path and modify the model path in [`assets/Config.toml`](./assets/Config.toml)
    

## 📝Supported Arguments

*   `--config`: Configure file path (default: `assets/Config.toml`)
*   `--ip`: The IP address the server is bound to
*   `--port`: Running port


## 📙Currently Available APIs

The API service starts at port 65530, and the data input and output format follow the Openai API specification.

*   `/api/oai/v1/models`
*   `/api/oai/models`
*   `/api/oai/v1/chat/completions`
*   `/api/oai/chat/completions`
*   `/api/oai/v1/completions`
*   `/api/oai/completions`
*   `/api/oai/v1/embeddings`
*   `/api/oai/embeddings`

The following is an example of ai00 invocation based on Python and an out of the box tool class implementation

```python
import openai

class Ai00:
    def __init__(self,model="model",port=65530,api_key="JUSTSECRET_KEY") :
        openai.api_base = f"http://127.0.0.1:{port}/api/oai"
        openai.api_key = api_key
        self.ctx = []
        self.ctx_str = ""
        self.params = {
            "system_name": "System",
            "user_name": "User", 
            "assistant_name": "Assistant",
            "model": model,
            "max_tokens": 4096,
            "top_p": 0.6,
            "temperature": 1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "half_life": 400,
            "stop": ['\x00','\n\n']
        }
        
    def set_params(self,**kwargs):
        self.params.update(kwargs)
        
    def clear_ctx(self):
        self.ctx = []
        self.ctx_str = ""
        
    def get_ctx(self):
        return self.ctx
    
    def send_message(self, message,role="user"):
        if role == "user":
            self.ctx.append({
                "role": self.params['user_name'],
                "content": message
            })
            self.ctx_str += f"{self.params['user_name']}: {message}\n\n{self.params['assistant_name']}: "
        elif role == "assistant":
            self.ctx.append({
                "role": self.params['assistant_name'],
                "content": message
            })
            self.ctx_str += f"{self.params['assistant_name']}: {message}\n\n{self.params['assistant_name']}: "
        elif role == "system":
            self.ctx.append({
                "role": self.params['system_name'],
                "content": message
            })
            self.ctx_str += f"{self.params['system_name']}: {message}"
            return ""
        else:
            raise ValueError("role must be 'user' or 'assistant' or 'system'")
        response = openai.Completion.create(
            model=self.params['model'],
            prompt=self.ctx_str,
            max_tokens=self.params['max_tokens'],
            half_life=self.params['half_life'],
            top_p=self.params['top_p'],
            temperature=self.params['temperature'],
            presence_penalty=self.params['presence_penalty'],
            frequency_penalty=self.params['frequency_penalty'],
            stop=self.params['stop']
        )
        result = response.choices[0].text
        self.ctx.append({
            "role": self.params['assistant_name'],
            "content": result
        })
        return result
    
ai00 = Ai00()
ai00.set_params(
    max_tokens = 4096,
    top_p = 0.55,
    temperature = 2,
    presence_penalty = 0.3,
    frequency_penalty = 0.8,
    half_life = 400,
    stop = ['\x00','\n\n']
)
print(ai00.send_message("how are you?"))
print(ai00.get_ctx())
ai00.clear_ctx()
print(ai00.send_message("me too!"))
print(ai00.get_ctx())
```

## 📙WebUI Screenshots

### Chat Feature  

<image src="img/chat_en.gif" />

### Continuation Feature

<image src="img/continuation_en.gif" />

### Paper Writing Feature

<image src="img/paper_en.gif" />

## 📝TODO List

*   [x] Support for `text_completions` and `chat_completions`
*   [x] Support for sse push
*   [x] Add `embeddings`
*   [x] Integrate basic front-end
*   [x] Parallel inference via `batch serve`
*   [x] Support for `int8` quantization
*   [x] Support for `NF4` quantization
*   [x] Support for `LoRA` model
*   [ ] Hot loading and switching of `LoRA` model

## 👥Join Us

We are always looking for people interested in helping us improve the project. If you are interested in any of the following, please join us!

*   💀Writing code
*   💬Providing feedback
*   🔆Proposing ideas or needs
*   🔍Testing new features
*   ✏Translating documentation
*   📣Promoting the project
*   🏅Anything else that would be helpful to us

No matter your skill level, we welcome you to join us. You can join us in the following ways:

*   Join our Discord channel
*   Join our QQ group
*   Submit issues or pull requests on GitHub
*   Leave feedback on our website

We can't wait to work with you to make this project better! We hope the project is helpful to you!


## Thank you to these awesome individuals who are insightful and outstanding for their support and selfless dedication to the project
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cgisky1980"><img src="https://avatars.githubusercontent.com/u/82481660?v=4?s=100" width="100px;" alt="顾真牛"/><br /><sub><b>顾真牛</b></sub></a><br /><a href="https://github.com/Ai00-X/ai00_server/commits?author=cgisky1980" title="Documentation">📖</a> <a href="https://github.com/Ai00-X/ai00_server/commits?author=cgisky1980" title="Code">💻</a> <a href="#content-cgisky1980" title="Content">🖋</a> <a href="#design-cgisky1980" title="Design">🎨</a> <a href="#mentoring-cgisky1980" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://cryscan.github.io/profile"><img src="https://avatars.githubusercontent.com/u/16053640?v=4?s=100" width="100px;" alt="研究社交"/><br /><sub><b>研究社交</b></sub></a><br /><a href="https://github.com/Ai00-X/ai00_server/commits?author=cryscan" title="Code">💻</a> <a href="#example-cryscan" title="Examples">💡</a> <a href="#ideas-cryscan" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-cryscan" title="Maintenance">🚧</a> <a href="https://github.com/Ai00-X/ai00_server/pulls?q=is%3Apr+reviewed-by%3Acryscan" title="Reviewed Pull Requests">👀</a> <a href="#platform-cryscan" title="Packaging/porting to new platform">📦</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/josStorer"><img src="https://avatars.githubusercontent.com/u/13366013?v=4?s=100" width="100px;" alt="josc146"/><br /><sub><b>josc146</b></sub></a><br /><a href="https://github.com/Ai00-X/ai00_server/issues?q=author%3AjosStorer" title="Bug reports">🐛</a> <a href="https://github.com/Ai00-X/ai00_server/commits?author=josStorer" title="Code">💻</a> <a href="#ideas-josStorer" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-josStorer" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/l15y"><img src="https://avatars.githubusercontent.com/u/11372524?v=4?s=100" width="100px;" alt="l15y"/><br /><sub><b>l15y</b></sub></a><br /><a href="#tool-l15y" title="Tools">🔧</a> <a href="#plugin-l15y" title="Plugin/utility libraries">🔌</a> <a href="https://github.com/Ai00-X/ai00_server/commits?author=l15y" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/cahyawirawan/"><img src="https://avatars.githubusercontent.com/u/7669893?v=4?s=100" width="100px;" alt="Cahya Wirawan"/><br /><sub><b>Cahya Wirawan</b></sub></a><br /><a href="https://github.com/Ai00-X/ai00_server/issues?q=author%3Acahya-wirawan" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yuunnn-w"><img src="https://avatars.githubusercontent.com/u/91336323?v=4?s=100" width="100px;" alt="yuunnn_w"/><br /><sub><b>yuunnn_w</b></sub></a><br /><a href="https://github.com/Ai00-X/ai00_server/commits?author=yuunnn-w" title="Documentation">📖</a> <a href="https://github.com/Ai00-X/ai00_server/commits?author=yuunnn-w" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/longzou"><img src="https://avatars.githubusercontent.com/u/59821454?v=4?s=100" width="100px;" alt="longzou"/><br /><sub><b>longzou</b></sub></a><br /><a href="https://github.com/Ai00-X/ai00_server/commits?author=longzou" title="Code">💻</a> <a href="#security-longzou" title="Security">🛡️</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## Stargazers over time

[![Stargazers over time](https://starchart.cc/cgisky1980/ai00_rwkv_server.svg)](https://starchart.cc/cgisky1980/ai00_rwkv_server)
