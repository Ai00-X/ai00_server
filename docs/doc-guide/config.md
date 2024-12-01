# 配置文件详解

Ai00 程序会按照 `assets/configs/Config.toml` 配置文件中的参数启动服务并运行 `RWKV` 模型。可以通过文本编辑软件（如记事本等）修改 `Config.toml` 的配置项，调整模型的运行效果。

下面是一组推荐的 `Config.toml` 配置。

**注意：** 除非你了解其具体作用，否则不要随意改动带有 `【不建议更改】` 标注的配置项。

``` bash copy
[model]
embed_device = "Cpu"                                 # 在 GPU 还是 CPU 上放置模型的 Embed 矩阵
max_batch = 8                                        # 【不建议更改】GPU 上缓存的最大批次
name = "RWKV-x060-World-3B-v2.1-20240417-ctx4096.st" # 模型名称，只支持后缀 .st 格式模型，请下载转换好的模型或自行转换
path = "assets/models"                               # 模型存放的路径
precision = "Fp16"                                   # 【不建议更改】中间张量精度 ("Fp16" or "Fp32")，Fp32 精度更高但速度更慢
quant = 0                                            # 量化层数，调高会提升效率，但可能损失精度，使模型效果变差
quant_type = "Int8"                                  # 量化类型 ("Int8" 或 "NF4")，Int 8 效果比 NF4 好，但需要更多显存
stop = ["\n\n"]                                      # 【不建议更改】添加额外的生成停止词
token_chunk_size = 128                               # 并行 Token 块大小，范围 32-128，显卡越牛逼这个数调越大（64 或 128）

# [[state]] # 是否挂载 state 文件
# id = "fd7a60ed-7807-449f-8256-bccae3246222"   #  state 文件的 UUID，不指定则随机分配 
# name = "x060-3B" # 是否为此 state 文件命名（可选项），可填 null
# path = "rwkv-x060-chn_single_round_qa-3B-20240505-ctx1024.state" # state 文件的路径，存放于 assets/models 目录下可填文件名称

# [[state]] # 继续挂载多个 state 文件
# id = "6a9c60a4-0f4c-40b1-a31f-987f73e20315"    # state 文件的 UUID，不指定则随机分配 
# name = "x060-7B" # 是否为此 state 文件命名（可选项），可填 null
# path = "rwkv-x060-chn_single_round_qa-3B-20240502-ctx1024.state" # 第二个 state 文件的路径，存放于 assets/models 目录下可填文件名称

# [[lora]] # 是否默认挂载 LoRA 文件
# alpha = 192 # LoRA 文件的 alpha 值
# path = "assets/models/rwkv-x060-3b.lora" # LoRA 文件的路径

[tokenizer]
path = "assets/tokenizer/rwkv_vocab_v20230424.json" # 【不建议更改】分词器路径

[bnf]
enable_bytes_cache = true   # 【不建议更改】是否启用缓存机制，以加速 BNF 某些短模式（schemas）的展开过程。
start_nonterminal = "start" # 【不建议更改】指定 BNF 模式中的初始非终结符。

[adapter]
Auto = {} # 【不建议更改】自动选择最佳 GPU。
# Manual = 0 # 手动指定使用哪个 GPU，可以通过 API （get）http://localhost:65530/api/adapters 获取可用的 GPU 列表

[listen]
acme = false # 【不建议更改】是否启用 acme 证书
domain = "local" # 【不建议更改】Ai00 服务域名
ip = "0.0.0.0"   # IPv4 地址
# ip = "::"        # 使用 IPv6
force_pass = true  # 是否强制通过鉴权步骤，改成 false 以使用密钥鉴权，从而控制 admin 系列 API 的访问权限
port = 65530 # Ai00 服务端口
slot = "permisionkey" 
tls = false  # 是否使用 https ，如果你只在本地体验 AI00 ，建议设置为 false

[[listen.app_keys]] # 添加多个用于管理员鉴权的密钥
app_id = "admin"
secret_key = "ai00_is_good"

[web] # 【不建议更改】移除此项以禁用 WebUI
path = "assets/www/index.zip" # 【不建议更改】web 界面资源的路径

# 【不建议更改】启用第三方的嵌入模型（使用 fast-embedding onnx 模型）
# 使用 API（post）http://localhost:65530/api/oai/embeds 可以调用第三方嵌入模型进行 embedding 操作
# [embed] # 取消 [embed] 及以下注释，启用第三方嵌入模型
# endpoint = "https://hf-mirror.com" # 第三方嵌入模型来源
# home = "assets/models/hf" # 第三方嵌入模型存放路径
# lib = "assets/ort/onnxruntime.dll"  # 仅在 windows 下使用
# name = { MultilingualE5Small = {} } # 第三方嵌入模型的名称
```