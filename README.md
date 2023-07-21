# 💯AI00 RWKV Server

AI00 RWKV Server 是一个基于[RWKV模型](https://github.com/BlinkDL/ChatRWKV)的推理API服务器。

支持VULKAN推理加速，可以在所有支持VULKAN的GPU上运行。

无需臃肿的pytorch CUDA等运行环境，小巧身材，开箱即用！

兼容Openai的ChatGPT API接口。

如果您正在寻找一个快速、高效、易于使用的API服务器，那么RWKV API Server是您的最佳选择。它可以用于各种任务，包括聊天机器人、文本生成、翻译和问答。

立即加入RWKV API Server社区，体验AI的魅力！

交流QQ群： 30920262



AI00 RWKV Server is based on the [RWKV model]（ https://github.com/BlinkDL/ChatRWKV ）Inference API server for.

Supports VULKAN inference acceleration and can run on all GPUs that support VULKAN.

No need for bulky Pytorch CUDA and other running environments, compact body, ready to use out of the box!

Compatible with Openai's ChatGPT API interface.

If you are looking for a fast, efficient, and easy-to-use API server, then RWKV API Server is your best choice. It can be used for various tasks, including Chatbot, text generation, translation and question answering.

Join the RWKV API Server community now and experience the charm of AI!

## 💥特色 Characteristic

- 基于RWKV模型，具有高性能和准确性(Based on the RWKV model, with high performance and accuracy)

- 支持VULKAN推理加速，不用该死的CUDA也能享受GPU加速！(Support VULKAN inference acceleration, and enjoy GPU acceleration without the need for damn CUDA!)
- 无需臃肿的pytorch CUDA等运行环境，小巧身材，开箱即用！(No need for bulky Pytorch CUDA and other running environments, compact body, ready to use out of the box!)
- 兼容Openai的ChatGPT API接口(Openai ChatGPT API interface compatible)

## ⭕用途 Use

- 聊天机器人   chatbot
- 文本生成   text generation
- 翻译  translate
- 问答 Q&A
- 其他所有你能想到的LLM能干的事 All the other things LLM can do that you can think of

## 👻其他 Other

基于 [web-rwkv](https://github.com/cryscan/web-rwkv) 项目

[下载模型 （download models）](https://huggingface.co/cgisky/RWKV-safetensors-fp16)



------

[^1]: haha



# 📜**安装** Install

安装了cargo 编译环境 (Installed the cargo compilation environment)

```bash
git clone https://github.com/cgisky1980/ai00_rwkv_serve.git

cd ai00_rwkv_serve
```
[下载模型 （download models）](https://huggingface.co/cgisky/RWKV-safetensors-fp16)
把模型放在（put model file as）  \assets\models\RWKV-4-World-0.4B-v1-20230529-ctx4096.st
目前模型路径和名称写死，后面可以在启动参数指定(At present, the model path and name are written dead, and can be specified in the startup parameters later on)

```bash
cargo b -r

./target/release/ai00_server.exe

```

API 服务开启于 3000 端口(API service is enabled on port 3000)

目前可用APIs (Currently available APIs)

/v1/chat/completions

/chat/completions

/v1/completions

/completions