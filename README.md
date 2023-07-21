# AI00 RWKV Server
中文  |  English  |  日本語 | 



一个基于RWKV模型的 API server。

支持VULKAN推理加速，可以在所有支持VULKAN的GPU上运行。

兼容Openai的 ChatGPT API 接口。

基于 [web-rwkv](https://github.com/cryscan/web-rwkv) 项目

[模型下载](https://huggingface.co/cgisky/RWKV-safetensors-fp16)


# 用法
```
git clone https://github.com/cgisky1980/ai00_rwkv_serve.git

cd ai00_rwkv_serve

// put model in \assets\models

cargo b -r

./target/release/ai00_server.exe

```

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

