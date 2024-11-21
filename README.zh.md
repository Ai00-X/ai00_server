# 💯AI00 Server
<p align='center'>
<image src="docs/public/logo.gif" />
</p>
 
<div align="center"> 
    
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
[![Rust Version](https://img.shields.io/badge/Rust-1.78.0+-blue)](https://releases.rs/docs/1.75.0)
![PRs welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen)     
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-zh-red.svg)](README.zh.md)

<div align="left"> 
 
---
`AI00 Server`是一个基于[`RWKV`模型](https://github.com/BlinkDL/ChatRWKV)的推理API服务器。

`AI00 Server`基于 [`WEB-RWKV`推理引擎](https://github.com/cryscan/web-rwkv)进行开发。

支持`Vulkan`/`Dx12`/`OpenGL`作为推理后端，无需臃肿的`pytorch`、`CUDA`等运行环境，小巧身材，开箱即用！

兼容OpenAI的ChatGPT API接口。

100% 开源可商用，采用MIT协议。

如果你是想要在自己的应用程序中内嵌一个LLM，且对用户的机器要求不那么苛刻（6GB以上GRAM的显卡）, `AI00 Server`无疑是一个很好的选择。

立即加入`AI00 RWKV Server`社区，体验AI的魅力！

交流QQ群：30920262

- [什么是 AI00](docs/ai00.md)
- [为什么只支持RWKV](docs/rwkv.md)


### ⭕模型下载和转换

你必须[下载模型](https://huggingface.co/BlinkDL)并将其放置在`assets/models`中。

你可以在这里下载已经转换好的模型： [V5](https://huggingface.co/cgisky/AI00_RWKV_V5) 或者 [V6](https://huggingface.co/cgisky/ai00_rwkv_x060)


## 安装、编译和使用

### 📦直接下载安装

1. 直接从 [Release](https://github.com/cgisky1980/ai00_rwkv_server/releases) 下载最新版本

2. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在`assets/models/`路径，例如`assets/models/RWKV-x060-World-3B-v2-20240228-ctx4096.st`

3. 你可以修改 [`assets/configs/Config.toml`](./assets/configs/Config.toml) 里面的模型配置，包括模型路径、量化层数等

4. 在命令行运行

    ```bash     
    ./ai00_rwkv_server
    ```

5. 打开浏览器，访问WebUI http://localhost:65530（如果开启了`tls`，访问 https://localhost:65530）

### 📜从源码编译

1. [安装Rust](https://www.rust-lang.org/)

2. 克隆本仓库

    ```bash
    git clone https://github.com/cgisky1980/ai00_rwkv_server.git
    cd ai00_rwkv_server
    ```
    
3. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在
`assets/models/`路径下，例如`assets/models/RWKV-x060-World-3B-v2-20240228-ctx4096.st`

4. 编译

    ```bash
    cargo build --release
    ```
     
5. 编译完成后运行
   
    ```bash     
    cargo run --release
    ```
   
6. 打开浏览器，访问WebUI http://localhost:65530（如果开启了`tls`，访问 https://localhost:65530）

### 📒模型转换

本项目目前仅支持`.st`后缀的 Safetensors 模型，通过`torch`保存的`.pth`后缀模型需要在使用前进行转换。

1. [下载pth模型](https://huggingface.co/BlinkDL)

2. 克隆或下载本仓库下[convert2ai00.py](./convert2ai00.py)或[convert_safetensors.py](./convert_safetensors.py)程序，并安装相应的依赖库（`torch`和`safetensors`）

3. 运行上述程序，并指定输入输出路径

    ```bash
    $ python convert_safetensors.py --input ./filename.pth --output ./filename.st
    ```

4. 如果你不想安装 Python 或 Torch，可以前往[`web-rwkv`](https://github.com/cryscan/web-rwkv/releases)并下载不依赖于 Python 或 Torch 的转换器`web-rwkv-converter`

    ```bash
    $ ./web-rwkv-converter --input /path/to/model.pth --output /path/to/model.st
    ```

5. 根据上文步骤，将转换所得的`.st`模型文件放在`assets/models/`路径下，并修改  [`assets/configs/Config.toml`](./assets/configs/Config.toml) 中的模型路径


## 📝支持的启动参数
- `--config`: 模型配置文件路径（默认`assets/configs/Config.toml`）
- `--ip`: 服务器绑定的IP地址
- `--port`: 运行端口


## 📙目前可用的API

API 服务开启于 65530 端口, 数据输入输出格式遵循 Openai API 规范。
有一些 API，比如`chat`和`completions`有一些可选的额外字段，这些额外字段是为高级功能准备的。可以访问 http://localhost:65530/api-docs 查看具体的 API 参数。

- `/api/oai/v1/models`
- `/api/oai/models`
- `/api/oai/v1/chat/completions`
- `/api/oai/chat/completions`
- `/api/oai/v1/completions`
- `/api/oai/completions`
- `/api/oai/v1/embeddings`
- `/api/oai/embeddings`

下面是一个 Python 的 Ai00 API 调用示例，开箱即用：

```python
import openai

class Ai00:
    def __init__(self,model="model",port=65530,api_key="JUSTSECRET_KEY") :
        openai.api_base = f"http://127.0.0.1:{port}/api/oai"
        openai.api_key = api_key
        self.ctx = []
        self.params = {
            "system_name": "System",
            "user_name": "User", 
            "assistant_name": "Assistant",
            "model": model,
            "max_tokens": 4096,
            "top_p": 0.6,
            "temperature": 1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "half_life": 400,
            "stop": ['\x00','\n\n']
        }
        
    def set_params(self,**kwargs):
        self.params.update(kwargs)
        
    def clear_ctx(self):
        self.ctx = []
        
    def get_ctx(self):
        return self.ctx
    
    def continuation(self, message):
        response = openai.Completion.create(
            model=self.params['model'],
            prompt=message,
            max_tokens=self.params['max_tokens'],
            half_life=self.params['half_life'],
            top_p=self.params['top_p'],
            temperature=self.params['temperature'],
            presence_penalty=self.params['presence_penalty'],
            frequency_penalty=self.params['frequency_penalty'],
            stop=self.params['stop']
        )
        result = response.choices[0].text
        return result
    
    def append_ctx(self,role,content):
        self.ctx.append({
            "role": role,
            "content": content
        })
        
    def send_message(self, message,role="user"):
        self.ctx.append({
            "role": role,
            "content": message
        })
        result = openai.ChatCompletion.create(
            model=self.params['model'],
            messages=self.ctx,
            names={
                "system": self.params['system_name'],
                "user": self.params['user_name'],
                "assistant": self.params['assistant_name']
            },
            max_tokens=self.params['max_tokens'],
            half_life=self.params['half_life'],
            top_p=self.params['top_p'],
            temperature=self.params['temperature'],
            presence_penalty=self.params['presence_penalty'],
            frequency_penalty=self.params['frequency_penalty'],
            stop=self.params['stop']
        )
        result = result.choices[0].message['content']
        self.ctx.append({
            "role": "assistant",
            "content": result
        })
        return result
    
ai00 = Ai00()
ai00.set_params(
    max_tokens = 4096,
    top_p = 0.55,
    temperature = 2,
    presence_penalty = 0.3,
    frequency_penalty = 0.8,
    half_life = 400,
    stop = ['\x00','\n\n']
)
print(ai00.send_message("how are you?"))
print(ai00.send_message("me too!"))
print(ai00.get_ctx())
ai00.clear_ctx()
print(ai00.continuation("i like"))
```

## BNF 采样

从 v0.5 开始 Ai00 有了一个独特功能：BNF 采样。这个采样法通过限定模型能够选择的 tokens 来使得模型强行输出符合一定格式的文本（比如 JSON 或者 Markdown 等等）。

以下是一个强行让模型输出有 "name"、"age" 和 "job" 字段的 JSON 的 BNF:

```
<start> ::= <json_object>
<json_object> ::= "{" <object_members> "}"
<object_members> ::= <json_member> | <json_member> ", " <object_members>
<json_member> ::= <json_key> ": " <json_value>
<json_key> ::= '"' "name" '"' | '"' "age" '"' | '"' "job" '"'
<json_value> ::= <json_string> | <json_number>
<json_string>::='"'<content>'"'
<content>::=<except!([escaped_literals])>|<except!([escaped_literals])><content>|'\\"'<content>|'\\"'
<escaped_literals>::='\t'|'\n'|'\r'|'"'
<json_number> ::= <positive_digit><digits>|'0'
<digits>::=<digit>|<digit><digits>
<digit>::='0'|<positive_digit>
<positive_digit>::="1"|"2"|"3"|"4"|"5"|"6"|"7"|"8"|"9"
```

<image src="img/bnf.png" />

## 📙WebUI 截图

### 对话功能

<image src="img/chat.gif" />

### 续写功能  

<image src="img/continuation.gif" />

### 写论文功能  

<image src="img/paper.gif" />

## 📝TODO List

- [x] 支持`text_completions`和`chat_completions`
- [x] 支持`sse`推送
- [x] 集成基本的调用前端
- [x] `Batch serve`并行推理
- [x] `int8`量化支持
- [x] `nf4`量化支持
- [x] `LoRA`模型支持
- [x] 支持加载微调的初始状态
- [ ] `LoRA`模型热加载、切换
- [x] 初始状态动态加载、切换
- [x] BNF采样

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
      <td align="center" valign="top" width="14.28%"><a href="https://cryscan.github.io/profile"><img src="https://avatars.githubusercontent.com/u/16053640?v=4?s=100" width="100px;" alt="研究社交"/><br /><sub><b>研究社交</b></sub></a><br /><a href="https://github.com/cgisky1980/ai00_rwkv_server/commits?author=cryscan" title="Code">💻</a> <a href="#example-cryscan" title="Examples">💡</a> <a href="#ideas-cryscan" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-cryscan" title="Maintenance">🚧</a> <a href="https://github.com/cgisky1980/ai00_rwkv_server/pulls?q=is%3Apr+reviewed-by%3Acryscan" title="Reviewed Pull Requests">👀</a> <a href="#platform-cryscan" title="Packaging/porting to new platform">📦</a></td>
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

 

