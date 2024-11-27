# Debug API

以下 API 用于调试或配置 Ai00。本地启动 Ai00 后，可以通过 `http://localhost:65530/api-docs/` 打开 API 文档，查看以下 API 。

要查看兼容 OpenAI 的 API 文档，请参考 [OpenAI API](./openai.md)。
## admin/models/load

**API 功能**：该 API 用于重新加载模型、 LoRA 、初始状态等运行时环境。

**API 地址**：（post）`http://localhost:65530/admin/models/load`

**参数列表**：

| 参数名称         | 是否可选 | 类型    | 参数解释                                                       |
| ---------------- | -------- | ------- | -------------------------------------------------------------- |
| adapter          | 可选     | string  | 指定使用的 GPU 和驱动后端，可使用 API `api/adapters` 查看，默认为 "Auto" |
| bnf              | 可选     | object  | BNF 选项配置，用于配置 BNF 解析相关参数，默认为空                |
| embed_device     | 可选     | string  | 在 GPU 还是 CPU 上放模型的 Embed 矩阵，默认为 "CPU"             |
| lora             | 可选     | array   | 是否加载 LoRA 文件，填 LoRA 文件的路径和 alpha 值等属性，默认为空       |
| max_batch        | 可选     | integer | 缓存在 GPU 上的最大批次，默认为 8                                 |
| model_path       | 必选     | string  | 模型文件路径                                                   |
| precision        | 可选     | string  | 中间张量的精度，默认为 "Fp16"                                   |
| quant_type       | 可选     | string  | 量化类型 ("Int8" or "NF4")，Int8 效果比 NF4 好，但需要更多显存   |
| quant            | 可选     | integer | 量化层数，调高会提升效率，但可能损失精度使模型效果变差，默认为 0  |
| state            | 可选     | array   | state 文件列表，每个元素包含 path 路径属性，默认为空               |
| token_chunk_size | 可选     | integer | 一次并行处理的最大 token 数，默认为 128                           |
| tokenizer_path   | 必选     | string  | 指定使用的 RWKV 分词器路径                                     |

**参考的 API 请求主体**

``` json
{
    "model_path": "RWKV-x060-ChnNovel-3B-20240807-ctx4096.st",
    "adapter": {
        "Manual": 0
    },
    "quant_type": "NF4",
    "quant": 31,
    "token_chunk_size": 128,
    "max_batch": 16,
    "tokenizer_path": "assets/tokenizer/rwkv_vocab_v20230424.json",
    "embed_device": "Cpu",
    "precision": "Fp16",
    "lora": [
        {
            "alpha": 192,
            "path": "test-x606-3B.lora"
        }
    ],
    "state": [
        {
            "path": "rwkv-x060-chn_single_round_qa-3B-20240516-ctx2048.state"
        }
    ]
}
```

返回结果说明：

- 该 API 没有直接的返回值
- 响应状态码 200 表示请求成功
- 如果加载失败，会返回相应的错误状态码和错误信息
- 加载状态和进度可在部署 Ai00 服务的终端查看

![admin/models/load-API](./imgs/admin-models-load-terminal.png)

## admin/models/save

**API 功能**：该 API 能够以 `.prefab` 格式导出**带有量化方法和量化层数两项配置**的 RWKV 模型。

> `.prefab` 预制件通常指游戏开发中使用的一种资源格式，允许开发者创建、保存和复用游戏对象（GameObject）及其**组件的配置**。
> 
> Ai00 支持导入和导出 `.prefab` 格式的 RWKV 模型，通过预量化 RWKV 模型节省加载时间。

**API 地址**：（post）`http://localhost:65530/admin/models/save`

**参数列表**：

| 参数名称 | 是否可选 | 类型   | 参数解释             |
| -------- | -------- | ------ | -------------------- |
| path     | 必选     | string | 导出 `.prefab` RWKV 模型的路径 |

**参考的 API 请求主体**

``` json
{
  "path": "/assets/models/temp.st"
}
```
返回结果说明：

- 该 API 没有直接的返回值`
- 响应状态码 200 表示请求成功
- 导出进度和结果可在部署服务的终端查看
- 导出成功后会在指定路径生成 `.prefab` 格式的模型文件

![admin/model/save](./imgs/admin-models-save-terminal.png)

## admin/models/state/load

**API 功能**：挂载指定路径的 state 文件。

> 作为 RNN 模型，RWKV 模型拥有固定大小的隐藏状态（State）。可通过加载 State 文件初始化 RWKV 模型的隐藏状态，以强化 RWKV 模型在特定任务的表现（类似于模型增强插件）。

**API 地址**：（post）`http://localhost:65530/admin/models/state/load`

**参数列表**：

| 参数名称 | 是否可选 | 类型           | 参数解释                                |
| -------- | -------- | -------------- | --------------------------------------- |
| default  | 可选     | boolean        | 是否立即挂载此 state                          |
| id       | 可选     | string         | state 文件的 UUID，不指定则随机分配     |
| name     | 可选     | string \| null | 为 state 文件命名     |
| path     | 必选     | string         |  `assets\models`目录下的某个 state 文件名 |

**参考的 API 请求主体**

``` json
{
  "default": false,
  "id": "00000000-0000-0000-0000-000000000000",
  "name": "中文单轮对话",
  "path": "rwkv-x060-chn_single_round_qa-1B6-20240516-ctx2048.state"
}
```

返回结果说明：

- 该 API 没有直接的返回值
- 响应状态码 200 表示请求成功
- 如果指定了 default 为 true，该 state 会立即生效
- state 文件的加载状态可在部署 Ai00 服务的终端查看

![admin/model/save](./imgs/admin-model-state-load-terminal.png)

## admin/models/unload

**API 功能**：关闭当前已加载的模型和 Ai00 服务。

**API 地址**：（get）`http://localhost:65530/admin/models/unload`

返回结果说明：

- 该 API 没有直接的返回值
- 响应状态码 200 表示请求成功
- 卸载成功后，需要重新加载模型才能继续使用服务
- 卸载状态可在部署 Ai00 服务的终端查看
  
![admin/models/unload-API](./imgs/admin-models-unload-terminal.png)

## api/adapters

**API 功能**：返回当前设备所有的 GPU 和驱动，以用于加载模型时指定 GPU。

**API 地址**：（get）`http://localhost:65530/api/adapters`

**参考的 API 返回结果**

``` json
[
    "AMD Radeon 780M Graphics (Vulkan)",
    "AMD Radeon 780M Graphics (Vulkan)",
    "AMD Radeon 780M Graphics (Dx12)",
    "AMD Radeon 780M Graphics (Dx12)",
    "AMD Radeon 780M Graphics (Gl)"
]
```
返回结果说明：

- 返回一个字符串数组，每个元素代表一个可用的 GPU 设备和其驱动类型
- 数组中的每个元素格式为：`"GPU名称 (驱动类型)"`，如 `"AMD Radeon 780M Graphics (Vulkan)"`
- 数组的索引（从 0 开始），可用于 `admin/models/load` API 中的 `adapter` 参数，例如 `"adapter": {"Manual": 0}` 指定使用第一个 GPU

> **注意**：截至 0.5.9 版本，Ai00 仅支持 Vulkan 驱动，不再支持 OpenGL 和 DirectX 驱动。

## api/auth/exchange

**API 功能**：切换带管理员权限的账号，`app_id` 和 `app_secret` 需提前在 `config.toml` 中配置。

**API 地址**：（post）`http://localhost:65530/api/auth/exchange`

**参数列表**：

| 参数名称   | 是否可选 | 类型   | 参数解释                               |
| ---------- | -------- | ------ | -------------------------------------- |
| app_id     | 必选     | string | 在 `config.toml` 中配置的 `app_id`     |
| app_secret | 必选     | string | 在 `config.toml` 中配置的 `app_secret` |

**参考的 API 请求主体**

``` json
{
  "app_id": "admin",
  "app_secret": "ai00_is_good"
}
```

**参考的 API 返回结果**

``` json
{
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzaWQiOiJhZG1pbiIsImV4cCI6MTczMjM1MjMwMH0.06fDScO-9GxF6rlq1v7MaC608uKg8qNN-TWZ8HvyZDk",
    "code": 200,
    "message": "SUCCESS"
}
```
返回结果说明：

- `token`: 生成的 JWT 认证令牌，用于后续 API 请求的身份验证
- `code`: 状态码，200 表示成功
- `message`: 状态信息，"SUCCESS" 表示成功
