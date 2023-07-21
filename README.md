# 💯AI00 RWKV Server

AI00 RWKV Server 是一个基于RWKV模型的推理API服务器。

支持VULKAN推理加速，可以在所有支持VULKAN的GPU上运行。

无需臃肿的pytorch CUDA等运行环境，小巧身材，开箱即用！

兼容Openai的ChatGPT API接口。

如果您正在寻找一个快速、高效、易于使用的API服务器，那么RWKV API Server是您的最佳选择。它可以用于各种任务，包括聊天机器人、文本生成、翻译和问答。

立即加入RWKV API Server社区，体验AI的魅力！

交流QQ群： 30920262

## 💥特色

- 基于RWKV模型，具有高性能和准确性

- 支持VULKAN推理加速，不用该死的CUDA也能享受GPU加速！
- 无需臃肿的pytorch CUDA等运行环境，小巧身材，开箱即用！
- 兼容Openai的ChatGPT API接口

## ⭕用途

- 聊天机器人
- 文本生成
- 翻译
- 问答
- 其他所有你能想到的LLM能干的事

## 👻其他

基于 [web-rwkv](https://github.com/cryscan/web-rwkv) 项目

[模型下载](https://huggingface.co/cgisky/RWKV-safetensors-fp16)







# 📜**安装**

安装了cargo 编译环境

```bash
git clone https://github.com/cgisky1980/ai00_rwkv_serve.git

cd ai00_rwkv_serve
```
[下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)
把模型放在  \assets\models\RWKV-4-World-0.4B-v1-20230529-ctx4096.st
目前模型路径和名称写死，后面可以在启动参数指定

```bash
cargo b -r

./target/release/ai00_server.exe

```

API 服务开启于 3000 端口

目前可用api

/v1/chat/completions

/chat/completions

/v1/completions

/completions