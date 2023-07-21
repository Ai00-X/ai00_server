# AI00 RWKV Server
中文  |  English  |  日本語 | 


AI00 RWKV Server 是一个基于RWKV模型的API服务器，它支持VULKAN推理加速，可以在所有支持VULKAN的GPU上运行。它无需臃肿的pytorch CUDA等运行环境，小巧身材，开箱即用！它兼容Openai的ChatGPT API接口。

如果您正在寻找一个快速、高效、易于使用的API服务器，那么RWKV API Server是您的最佳选择。它可以用于各种任务，包括聊天机器人、文本生成、翻译和问答。

立即加入RWKV API Server社区，体验AI的魅力！

##特色

基于RWKV模型，具有高性能和准确性
支持VULKAN推理加速，可以提高性能
无需臃肿的pytorch CUDA等运行环境，小巧身材，开箱即用！
兼容Openai的ChatGPT API接口

##用途

聊天机器人
文本生成
翻译
问答

##其他

基于 [web-rwkv](https://github.com/cryscan/web-rwkv) 项目

[模型下载](https://huggingface.co/cgisky/RWKV-safetensors-fp16)

交流QQ群： 30920262

# 安装

安装了cargo 编译环境

```
git clone https://github.com/cgisky1980/ai00_rwkv_serve.git

cd ai00_rwkv_serve

// put model in \assets\models\RWKV-4-World-0.4B-v1-20230529-ctx4096.st
// 目前模型路径和名称写死，后面可以在启动参数指定

cargo b -r

./target/release/ai00_server.exe

```

API 服务开启于 3000 端口

目前可用api

/v1/chat/completions

/chat/completions

/v1/completions

/completions

------

An API server based on the RWKV model.

Supports VULKAN inference acceleration. So can run RWKV on all GPUs that support VULKAN.

API interface compatible with Openai's API .

Based on  [web-rwkv](https://github.com/cryscan/web-rwkv)  project

[download models](https://huggingface.co/cgisky/RWKV-safetensors-fp16)

------

RWKV モデルに基づく API サーバー。

VULKAN推論アクセラレーションをサポートします。そのため、VULKANをサポートするすべてのGPUでRWKVを実行できます。

OpenaiのAPIと互換性のあるAPIインターフェース。

プロジェクトに基づく [web-rwkv](https://github.com/cryscan/web-rwkv) 

[モデルのダウンロード](https://huggingface.co/cgisky/RWKV-safetensors-fp16)

