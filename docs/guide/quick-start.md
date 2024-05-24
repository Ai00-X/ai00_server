
## 下载与安装

对于新手来说，我们建议直接从 Ai00 Server 的 [Release 页面](https://github.com/Ai00-X/ai00_server/releases)下载最新版本。

在每个版本发布的 Assets 版块可以找到已经打包好的 Ai00 Server 压缩包，下载并解压即可使用。

![ai00-download](./doc_img/ai00-download.png)

## 下载/转换 RWKV 模型

Ai00 Server 目前仅支持 `.st` 后缀的 Safetensors 模型，有两种方法可以得到 `.st` 模型：

1. 下载已经转换好的 `.st` 模型（推荐方式）

- RWKV-5 系列：https://huggingface.co/cgisky/AI00_RWKV_V5/tree/main

- RWKV-6 系列：https://huggingface.co/cgisky/ai00_rwkv_x060/tree/main

如果你无法访问上面的网站，请访问以下镜像站：

- RWKV-5 系列：https://hf-mirror.com/cgisky/AI00_RWKV_V5/tree/main

- RWKV-6 系列：https://hf-mirror.com/cgisky/ai00_rwkv_x060/tree/main

2. 下载 `.pth` 后缀模型，并通过工具转换成 `.st` 模型

首先，可以从 RWKV 官方仓库中下载 .pth 后缀的 RWKV 模型，下载地址：

- RWKV-5 系列：https://huggingface.co/BlinkDL/rwkv-5-world/tree/main

- RWKV-6 系列：https://huggingface.co/BlinkDL/rwkv-6-world/tree/main

如果你无法访问上面的网站，请访问以下镜像站：

- RWKV-5 系列：https://hf-mirror.com/BlinkDL/rwkv-5-world/tree/main

- RWKV-6 系列：https://hf-mirror.com/BlinkDL/rwkv-6-world/tree/main

> .pth 模型下载完成后，应该可以在文件夹中找到:
> 
>![ai00-model-list](./doc_img/ai00-model-list.png)

在 Ai00 Server 解压的文件夹中，可以找到名为 “`convert.exe`” 的模型转换工具。

在命令行中执行以下命令，可以将指定路径的 `.pth` 模型转化成 `.st` 模型：
```
$ ./converter --input /path/to/model.pth
```
请将上述命令中的 `/path/to/model.pth` 改成需要转换的模型文件路径。

获得 `.st` 后缀的 RWKV 模型后，我们需要在 `assets` 文件夹中新建一个 `models` 文件夹，并将 RWKV 模型放在此文件夹中。


## 调整配置参数

Ai00 程序会按照 `assets/configs/Config.toml` 配置文件中的参数运行 `RWKV` 模型。可以通过文本编辑软件（如记事本等）修改 `Config.toml` 的配置项，调整模型的运行效果。

下面是一组推荐的 `Config.toml` 配置。

注意：带中文标注的配置项可以尝试更改，其他英文标注的配置项不建议自行更改，除非你了解其具体作用。
```
[model]
embed_device = "Cpu"                                 # 在GPU还是CPU上放模型的Embed矩阵 ("Cpu" or "Gpu").
max_batch = 8                                        # The maximum batches that are cached on GPU.
name = "RWKV-x060-World-3B-v2.1-20240417-ctx4096.st" # 模型名称，只支持后缀st格式模型，请自己在RWKV程序中转换好，或者直接下载转换好的模型
path = "assets/models"                               # 模型路径
precision = "Fp16"                                   # Precision for intermediate tensors ("Fp16" or "Fp32"). "Fp32" yields better outputs but slower.
quant = 0                                            # 量化层数，调高会提升效率，但可能损失精度，使模型效果变差
quant_type = "Int8"                                  # 量化类型 ("Int8" or "NF4")，Int 8 效果比 NF4 好，但需要更多显存
stop = ["\n\n"]                                      # Additional stop words in generation.
token_chunk_size = 128                               # 并行Token块大小，范围32-128，显卡越牛逼这个数调越大(64 or 128)

# [[state]] # 是否挂载 state
# id = "fd7a60ed-7807-449f-8256-bccae3246222"   #  state 文件的 UUID，不指定则随机分配 
# name = "x060-3B" # 是否为此 state 文件命名（可选项）
# path = "rwkv-x060-chn_single_round_qa-3B-20240505-ctx1024.state" # state 文件的路径

# [[state]] # 是否挂载第二个 state 文件
# id = "6a9c60a4-0f4c-40b1-a31f-987f73e20315"    # state 文件的 UUID，不指定则随机分配 
# path = "rwkv-x060-chn_single_round_qa-3B-20240502-ctx1024.state" # state 文件的路径

# [[lora]] # 是否默认启用 LoRA 
# alpha = 192
# path = "assets/models/rwkv-x060-3b.lora" # LoRA 文件的路径

[tokenizer]
path = "assets/tokenizer/rwkv_vocab_v20230424.json" # Path to the tokenizer.

[bnf]
enable_bytes_cache = true   # Enable the cache that accelerates the expansion of certain short schemas.
start_nonterminal = "start" # The initial nonterminal of the BNF schemas.

[adapter]
Auto = {} # Choose the best GPU.
# Manual = 0 # Manually specify which GPU to use.

[listen]
acme = false
domain = "local"
ip = "0.0.0.0"   # IPv4 地址
# ip = "::"        # Use IpV6.
force_pass = true
port = 65530
slot = "permisionkey"
tls = true      # 是否使用 https ，如果你只在本地体验 AI00 请设置为 false

[[listen.app_keys]] # Allow mutiple app keys.
app_id = "JUSTAISERVER"
secret_key = "JUSTSECRET_KEY"

[web] # Remove this to disable WebUI.
path = "assets/www/index.zip" # Path to the WebUI.

```

## 运行 Ai00 程序

配置项修改完毕后，请保存 `Config.toml` 文件，并双击运行 `ai00_server.exe` 程序。

当命令行中出现 `INFO  [ai00_server::middleware] model loaded` 提示时，意味着模型已经加载完成：

![ai00-model-reloaded](./doc_img/ai00-model-reloaded.png)

此时我们打开任意浏览器，并访问 `https://localhost:65530`，即可打开 Ai00 的 Web 界面。

![Ai00-homepage](./doc_img/Ai00-homepage.png)


## 调整右侧解码参数

Web 页面的右侧有一些可设置的模型解码参数，如 `Temperature` 、`Top_P`、`Presence Penalty` 和 `Frequency Penalty` ，调整这些参数会影响模型的生成效果。

参数对应的效果如下：

| API 参数| 效果描述  |
|----|----|
| Temperature | 采样温度，就像给模型喝酒，数值越大随机性越强，更具创造力，数值越小则越保守稳定。                                             |
| Top_P | 就像给模型喂镇静剂，优先考虑前 n% 概率质量的结果。如设置成 0.1 则考虑前 10%，生成内容质量更高但更保守。如设置成 1，则考虑所有质量结果，质量降低但更多样。 |
| Presence Penalty | 存在惩罚，正值根据“新 token 在至今的文本中是否出现过”来对其进行惩罚，从而增加了模型涉及新话题的可能性。|
| Frequency Penalty| 频率惩罚，正值根据“新 token 在至今的文本中出现的频率/次数”来对其进行惩罚，从而减少模型原封不动地重复相同句子的可能性。|

其中 `Temperature` 和 `Top_P` 两个参数对模型生成效果的影响最大。

### 参数推荐

续写小说和对话这一类需要创造性的任务，需要高 `Temperature` + 低 `Top_P` 的参数组合，可以尝试以下四种参数搭配：

`Temperature` 1.2 ，`Top_P` 0.5

`Temperature` 1.4 ，`Top_P` 0.4

`Temperature` 1.4 ，`Top_P` 0.3

`Temperature` 2 ，`Top_P` 0.2

举个例子，续写小说可以尝试将 `Temperature` 设为 2 （ `Temperature` 增加会提高文采，但逻辑会下降），然后将 `Top_P` 设为 0.1 ~ 0.2 （`Top_P` 越低，逻辑能力越强），这样生成的小说内容逻辑和文采都很好。
完成相对机械的任务，例如材料问答、文章摘要等，则可将参数设为：

`Temperature` 1 ，`Top_P` 0.2

`Temperature` 1 ，`Top_P` 0.1

`Temperature` 1 ，`Top_P` 0

举个例子，如果你正在执行像关键词提取之类的机械任务，不需要模型进行任何开放性思考，则可以将 `Temperature` 设为 1 ，`Top_P`、`Presence Penalty`、`Frequency Penalty` 都设为 0 。
