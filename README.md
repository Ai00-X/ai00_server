# 💯AI00 RWKV Server
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

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

2. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在`assets/models/`路径，例如`assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st`

3. 在命令行运行

    ```bash     
    $ ./ai00_rwkv_server --model assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st
    ```
4. 打开浏览器，访问WebUI
   [`http://127.0.0.1:3000`](http://127.0.0.1:3000)

### 📜从源码编译

1. [安装Rust](https://www.rust-lang.org/)

2. 克隆本仓库

    ```bash
    $ git clone https://github.com/cgisky1980/ai00_rwkv_server.git
    $ cd ai00_rwkv_server
    ```
    

3. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在
`assets/models/`路径下，例如`assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st`

4. 编译

    ```bash
    $ cargo build --release
    ```
     

5. 编译完成后运行
   
    ```bash     
    $ cargo run --release -- --model assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st 
    ```
   
6. 打开浏览器，访问WebUI
   [`http://127.0.0.1:3000`](http://127.0.0.1:3000)

    
## 📝支持的启动参数
- `--model`: 模型路径
- `--tokenizer`: 词表路径
- `--port`: 运行端口


## 📙目前可用的API

API 服务开启于 3000 端口, 数据输入已经输出格式遵循Openai API 规范。

- `/v1/chat/completions`
- `/chat/completions`
- `/v1/completions`
- `/completions`
- `/v1/embeddings`
- `/embeddings`

## 📙WebUI 截图

![image](https://github.com/cgisky1980/ai00_rwkv_server/assets/82481660/33e8da0b-5d3f-4dfc-bf35-4a8147d099bc)

![image](https://github.com/cgisky1980/ai00_rwkv_server/assets/82481660/a24d6c72-31a0-4ff7-8a61-6eb98aae46e8)


## 📝TODO List

- [x] 支持text_completions和chat_completions
- [x] 支持sse推送
- [x] 添加embeddings
- [x] 集成基本的调用前端
- [ ] batch serve 并行推理
- [ ] int8量化支持
- [ ] SpQR量化支持
- [ ] LoRA模型支持
- [ ] LoRA模型热加载、切换

## 👥Join Us

我们一直在寻找有兴趣帮助我们改进项目的人。如果你对以下任何一项感兴趣，请加入我们！

- 💀编写代码
- 💬提供反馈
- 🔆提出想法或需求
- 🔍测试新功能
- ✏翻译文档
- 📣推广项目
- 🏅其他任何会对我们有所帮助的事

无论你的技能水平如何，我们都欢迎你加入我们。你可以通过以下方式加入我们：

- 加入我们的 Discord 频道
- 加入我们的 QQ 群
- 在 GitHub 上提交问题或拉取请求
- 在我们的网站上留下反馈
  
我们迫不及待地想与你合作，让这个项目变得更好！希望项目对你有帮助！

## Thanks


[![cryscan](https://avatars.githubusercontent.com/u/16053640?s=32&v=4)](https://github.com/cryscan)
感谢cryscan的辛勤付出，为项目做出了杰出的贡献。

### 感谢下面项目的编写者们做出的杰出工作

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cgisky1980"><img src="https://avatars.githubusercontent.com/u/82481660?v=4?s=100" width="100px;" alt="顾真牛"/><br /><sub><b>顾真牛</b></sub></a><br /><a href="https://github.com/cgisky1980/ai00_rwkv_server/commits?author=cgisky1980" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->




### 感谢下面又好看又有眼光又优秀的杰出人士对项目的支持和无私奉献

- 来自 QQ 群

    ![image](https://github.com/cgisky1980/ai00_rwkv_server/assets/82481660/6e324617-6d0c-49fd-ab1e-fd9cf02df51e)

- 来自 Github 

- 来自 Discord

我们很感激您的帮助，我们很高兴能与您合作。


## Stargazers over time

[![Stargazers over time](https://starchart.cc/cgisky1980/ai00_rwkv_server.svg)](https://starchart.cc/cgisky1980/ai00_rwkv_server)


## BTW

- [什么是 AI00](docs/ai00.md)
- [为什么只支持RWKV](docs/rwkv.md)




## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!