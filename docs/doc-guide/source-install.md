# 从源码安装

按照以下步骤，从源码编译安装 Ai00 程序。

1. [安装Rust](https://www.rust-lang.org/)，Ai00 依赖 rust 版本的 web-rwkv 库，所以需要 rust 编译器。

2. 克隆 Ai00 仓库

    ```bash
    git clone https://github.com/cgisky1980/ai00_rwkv_server.git
    cd ai00_rwkv_server
    ```
3. [下载模型](https://huggingface.co/cgisky/RWKV-safetensors-fp16)后把模型放在
`assets/models/`路径下，例如`assets/models/RWKV-x060-World-3B-v2-20240228-ctx4096.st`

1. 编译

    ```bash
    cargo build --release
    ```
     
2. 编译完成后运行
   
    ```bash     
    cargo run --release
    ```
   
3. 打开浏览器，访问WebUI http://localhost:65530（如果开启了`tls`，访问 https://localhost:65530）