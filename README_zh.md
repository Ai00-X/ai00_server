# 💯AI00 Server
<p align='center'>
<image src="docs/ai00.gif" />
</p>
 
<div align="center"> 
    
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
[![Rust Version](https://img.shields.io/badge/Rust-1.65.0+-blue)](https://releases.rs/docs/1.65.0)
![PRs welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen)     
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->



[English](README.md) | [中文](README_zh.md) 

<div align="left"> 
 
---
`AI00 Server`是一个基于[`RWKV`模型](https://github.com/BlinkDL/ChatRWKV)的推理API服务器。

`AI00 Server`基于 [`WEB-RWKV`推理引擎](https://github.com/cryscan/web-rwkv)进行开发。

支持Vulkan/Dx12/OpenGL作为推理后端，无需臃肿的`pytorch`、`CUDA`等运行环境，小巧身材，开箱即用！

兼容OpenAI的ChatGPT API接口。

100% 开源可商用，采用MIT协议。

如果你是想要在自己的应用程序中内嵌一个LLM，且对用户的机器要求不那么苛刻（6GB以上GRAM的显卡）, `AI00 Server`无疑是一个很好的选择。

立即加入`AI00 RWKV Server`社区，体验AI的魅力！

交流QQ群：30920262

- [什么是 AI00](docs/ai00.md)
- [为什么只支持RWKV](docs/rwkv.md)


### ⭕模型下载和转换

You must [download the model](https://huggingface.co/BlinkDL) and put in assets/models before running if you are building from source. 
You may download the official RWKV World series models from HuggingFace, and convert them via the provided `convert_safetensors.py`.

你可以在这里下载已经转换好的V4 模型： [模型下载](https://huggingface.co/cgisky/RWKV-safetensors-fp16)


## 安装、编译和使用

### 📦直接下载安装

1. 直接从 [Release](https://github.com/cgisky1980/ai00_rwkv_server/releases) 下载最新版本

2. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在`assets/models/`路径，例如`assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st`

3. 你可以修改 [`assets/configs/Config.toml`](./assets/configs/Config.toml) 里面的模型配置，包括模型路径、量化层数等

4. 在命令行运行

    ```bash     
    ./ai00_rwkv_server
    ```
5. 打开浏览器，访问WebUI
   [`http://127.0.0.1:65530`](http://127.0.0.1:65530)

### 📜从源码编译

1. [安装Rust](https://www.rust-lang.org/)

2. 克隆本仓库

    ```bash
    git clone https://github.com/cgisky1980/ai00_rwkv_server.git
    cd ai00_rwkv_server
    ```
    

3. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在
`assets/models/`路径下，例如`assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st`

4. 编译

    ```bash
    cargo build --release
    ```
     

5. 编译完成后运行
   
    ```bash     
    cargo run --release
    ```
   
6. 打开浏览器，访问WebUI
   [`http://127.0.0.1:65530`](http://127.0.0.1:65530)

### 📒模型转换

本项目目前仅支持`.st`后缀的 Safetensors 模型，通过`torch`保存的`.pth`后缀模型需要在使用前进行转换。

1. [下载pth模型](https://huggingface.co/BlinkDL)

2. 克隆或下载本仓库下[convert_safetensors.py](./convert_safetensors.py)程序，并安装相应的依赖库

3. 运行上述程序，并指定输入输出路径

    ```bash
    $ python convert_safetensors.py --input ./filename.pth --output ./filename.st
    ```

4. 根据上文步骤，将转换所得的`.st`模型文件放在`assets/models/`路径下，并修改  [`assets/Config.toml`](./assets/Config.toml) 中的模型路径


## 📝支持的启动参数
- `--config`: 模型配置文件路径（默认`assets/Config.toml`）
- `--ip`: 服务器绑定的IP地址
- `--port`: 运行端口


## 📙目前可用的API

API 服务开启于 65530 端口, 数据输入已经输出格式遵循Openai API 规范。

- `/api/oai/v1/models`
- `/api/oai/models`
- `/api/oai/v1/chat/completions`
- `/api/oai/chat/completions`
- `/api/oai/v1/completions`
- `/api/oai/completions`
- `/api/oai/v1/embeddings`
- `/api/oai/embeddings`

## 📙WebUI 截图




## 📝TODO List

- [x] 支持`text_completions`和`chat_completions`
- [x] 支持`sse`推送
- [x] 添加`embeddings`
- [x] 集成基本的调用前端
- [x] `Batch serve`并行推理
- [x] `int8`量化支持
- [x] `nf4`量化支持
- [x] `LoRA`模型支持
- [ ] `LoRA`模型热加载、切换

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cgisky1980"><img src="https://avatars.githubusercontent.com/u/82481660?v=4?s=100" width="100px;" alt="顾真牛"/><br /><sub><b>顾真牛</b></sub></a><br /><a href="https://github.com/cgisky1980/ai00_rwkv_server/commits?author=cgisky1980" title="Documentation">📖</a> <a href="https://github.com/cgisky1980/ai00_rwkv_server/commits?author=cgisky1980" title="Code">💻</a> <a href="#content-cgisky1980" title="Content">🖋</a> <a href="#design-cgisky1980" title="Design">🎨</a> <a href="#mentoring-cgisky1980" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://cryscan.github.io/profile"><img src="https://avatars.githubusercontent.com/u/16053640?v=4?s=100" width="100px;" alt="研究社交"/><br /><sub><b>研究社交</b></sub></a><br /><a href="https://github.com/cgisky1980/ai00_rwkv_server/commits?author=cryscan" title="Code">💻</a> <a href="#example-cryscan" title="Examples">💡</a> <a href="#ideas-cryscan" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-cryscan" title="Maintenance">🚧</a> <a href="https://github.com/cgisky1980/ai00_rwkv_server/pulls?q=is%3Apr+reviewed-by%3Acryscan" title="Reviewed Pull Requests">👀</a> <a href="#platform-cryscan" title="Packaging/porting to new platform">📦</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/josStorer"><img src="https://avatars.githubusercontent.com/u/13366013?v=4?s=100" width="100px;" alt="josc146"/><br /><sub><b>josc146</b></sub></a><br /><a href="https://github.com/cgisky1980/ai00_rwkv_server/issues?q=author%3AjosStorer" title="Bug reports">🐛</a> <a href="https://github.com/cgisky1980/ai00_rwkv_server/commits?author=josStorer" title="Code">💻</a> <a href="#ideas-josStorer" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-josStorer" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/l15y"><img src="https://avatars.githubusercontent.com/u/11372524?v=4?s=100" width="100px;" alt="l15y"/><br /><sub><b>l15y</b></sub></a><br /><a href="#tool-l15y" title="Tools">🔧</a> <a href="#plugin-l15y" title="Plugin/utility libraries">🔌</a> <a href="https://github.com/cgisky1980/ai00_rwkv_server/commits?author=l15y" title="Code">💻</a></td>
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

 

