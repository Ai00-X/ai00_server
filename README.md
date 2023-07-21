# 💯AI00 RWKV Server

`AI00 RWKV Server`是一个基于[`RWKV`模型](https://github.com/BlinkDL/ChatRWKV)的推理API服务器。

支持`VULKAN`推理加速，可以在所有支持`VULKAN`的GPU上运行。不用N卡！！！A卡甚至集成显卡都可加速！！！

无需臃肿的`pytorch`、`CUDA`等运行环境，小巧身材，开箱即用！

兼容OpenAI的ChatGPT API接口。

100% 开源可商用，采用MIT协议。

如果您正在寻找一个快速、高效、易于使用的LLM API服务器，那么`AI00 RWKV Server`是您的最佳选择。它可以用于各种任务，包括聊天机器人、文本生成、翻译和问答。

立即加入`AI00 RWKV Server`社区，体验AI的魅力！

交流QQ群：30920262

### 💥特色

- 基于`RWKV`模型，具有高性能和准确性
- 支持`VULKAN`推理加速，不用该死的`CUDA`也能享受GPU加速！支持A卡、集成显卡等一切支持`VULKAN`的GPU
- 无需臃肿的`pytorch`、`CUDA`等运行环境，小巧身材，开箱即用！
- 兼容OpenAI的ChatGPT API接口

### ⭕用途

- 聊天机器人
- 文本生成
- 翻译
- 问答
- 其他所有你能想到的LLM能干的事

### 👻其他

- 基于 [web-rwkv](https://github.com/cryscan/web-rwkv) 项目
- [模型下载](https://huggingface.co/cgisky/RWKV-safetensors-fp16)

## 安装、编译和使用

### 📦直接下载安装

1. 直接从 [Release](https://github.com/cgisky1980/ai00_rwkv_server/releases) 下载最新版本

2. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在`/assets/models/`路径，例如`/assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st`

3. 在命令行运行
    ```bash
    $ ./ai00_rwkv_server --model /assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st
    ```

### 📜从源码编译

1. [安装Rust](https://www.rust-lang.org/)

2. 克隆本仓库
    ```bash
    $ git clone https://github.com/cgisky1980/ai00_rwkv_serve.git
    $ cd ai00_rwkv_serve
    ```

3. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在
`/assets/models/`路径下，例如`/assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st`

4. 编译
    ```bash
    $ cargo build --release
    ```

5. 编译完成后运行
    ```bash
    $ cargo run --release -- --model /assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st
    ```

## 目前可用的API

API 服务开启于 3000 端口

- `/v1/chat/completions`
- `/chat/completions`
- `/v1/completions`
- `/completions`