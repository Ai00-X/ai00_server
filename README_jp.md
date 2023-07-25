# 💯AI00 RWKVサーバー
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

`AI00 RWKV Server`は、[`RWKV`モデル](https://github.com/BlinkDL/ChatRWKV)に基づく推論APIサーバーです。

`VULKAN`推論加速をサポートし、すべての`VULKAN`対応GPUで動作します。NVidiaカードは必要ありません！AMDカードや統合グラフィックスカードでも加速可能です！

重たい`pytorch`や`CUDA`などのランタイム環境は必要ありません。コンパクトで、すぐに使えます！

OpenAIのChatGPT APIインターフェースと互換性があります。

100% オープンソースで商用利用可能、MITライセンスを採用。

高速で効率的で使いやすいLLM APIサーバーを探しているなら、`AI00 RWKV Server`が最適です。チャットボット、テキスト生成、翻訳、質問応答など、さまざまなタスクに使用できます。

すぐに`AI00 RWKV Server`コミュニティに参加し、AIの魅力を体験しましょう！

交流QQグループ：30920262

### 💥特徴

*   `RWKV`モデルに基づき、高性能で精度が高い
*   `VULKAN`推論加速をサポートし、`CUDA`がなくてもGPU加速を享受できます！Aカード、統合グラフィックスカードなど、すべての`VULKAN`対応GPUをサポート
*   重たい`pytorch`や`CUDA`などのランタイム環境は不要、コンパクトで、すぐに使えます！
*   OpenAIのChatGPT APIインターフェースと互換性があります

### ⭕用途

*   チャットボット
*   テキスト生成
*   翻訳
*   質問応答
*   その他、LLMが可能なすべてのタスク

### 👻その他

*   [web-rwkv](https://github.com/cryscan/web-rwkv) プロジェクトに基づいています
*   [モデルのダウンロード](https://huggingface.co/cgisky/RWKV-safetensors-fp16)

## インストール、コンパイル、使用方法

### 📦直接ダウンロードしてインストール

1.  [Release](https://github.com/cgisky1980/ai00_rwkv_server/releases)から最新バージョンをダウンロードします
    
2.  [モデルをダウンロード](https://huggingface.co/cgisky/RWKV-safetensors-fp16)し、`assets/models/`パスに配置します。例：`assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st`
    
3.  コマンドラインで実行します
    
    ```bash
    $ ./ai00_rwkv_server --model assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st
    ```
    
4.  ブラウザを開き、WebUIにアクセスします [`http://127.0.0.1:3000`](http://127.0.0.1:3000)
    

### 📜ソースコードからコンパイル

1.  [Rustをインストール](https://www.rust-lang.org/)
    
2.  このリポジトリをクローンします
    
    ```bash
    $ git clone https://github.com/cgisky1980/ai00_rwkv_server.git $ cd ai00_rwkv_server
    ```
    
3.  [モデルをダウンロード](https://huggingface.co/cgisky/RWKV-safetensors-fp16)し、`assets/models/`パスに配置します。例：`assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st`
    
4.  コンパイルします
    
    ```bash
    $ cargo build --release
    ```
    
5.  コンパイルが完了したら実行します
    
    ```bash
    $ cargo run --release -- --model assets/models/RWKV-4-World-0.4B-v1-20230529-ctx4096.st
    ```
    
6.  ブラウザを開き、WebUIにアクセスします [`http://127.0.0.1:3000`](http://127.0.0.1:3000)
    

## 📝サポートされている起動パラメーター

*   `--model`: モデルのパス
*   `--tokenizer`: トークナイザーの
*   `--port`: 実行ポート

## 📙現在利用可能なAPI

APIサービスは3000ポートで開始され、データ入力と出力の形式はOpenai APIの規格に従います。

*   `/v1/models`
*   `/models`
*   `/v1/chat/completions`
*   `/chat/completions`
*   `/v1/completions`
*   `/completions`
*   `/v1/embeddings`
*   `/embeddings`

## 📙WebUIスクリーンショット

![image](https://github.com/cgisky1980/ai00_rwkv_server/assets/82481660/33e8da0b-5d3f-4dfc-bf35-4a8147d099bc)

![image](https://github.com/cgisky1980/ai00_rwkv_server/assets/82481660/a24d6c72-31a0-4ff7-8a61-6eb98aae46e8)

## 📝TODOリスト

*   [x] `text_completions`と`chat_completions`のサポート
*   [x] `sse`プッシュのサポート
*   [x] `embeddings`の追加
*   [x] 基本的なフロントエンドの統合
*   [ ] `batch serve`並行推論
*   [ ] `int8`量子化のサポート
*   [ ] `SpQR`量子化のサポート
*   [ ] `LoRA`モデルのサポート
*   [ ] `LoRA`モデルのホットロード、切り替え

## 👥Join Us

私たちは常にプロジェクトを改善するのを手伝ってくれる人を探しています。以下のいずれかに興味がある場合は、ぜひ参加してください！

*   💀コードの作成
*   💬フィードバックの提供
*   🔆アイデアや要求の提出
*   🔍新機能のテスト
*   ✏ドキュメンテーションの翻訳
*   📣プロジェクトのプロモーション
*   🏅私たちを助ける他の何でも

あなたのスキルレベルに関係なく、私たちはあなたの参加を歓迎します。以下の方法で参加してください：

*   私たちのDiscordチャンネルに参加する
*   私たちのQQグループに参加する
*   GitHubで問題を提出したり、プルリクエストを作成したりする
*   私たちのウェブサイトでフィードバックを残す

私たちはあなたと協力して、このプロジェクトをより良くすることを楽しみにしています！プロジェクトがあなたに役立つことを願っています！
