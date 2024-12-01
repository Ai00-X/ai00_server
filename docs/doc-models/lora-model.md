# LoRA

LORA（Low-Rank Adaptation）是一种针对大型预训练模型的微调技术。它不改变原始模型大部分参数，而是调整模型的部分权重，以此实现对特定任务的优化。

## 在 Ai00 搭载 LoRA 文件

在 Ai00 搭载 LoRA 文件时，需要将 LoRA 文件放置在存放 RWKV 模型的 `models` 目录下。然后在 `config.toml` 文件中添加 LoRA 文件路径。

详情请参见 [配置文件](../doc-guide/config.md) 中 `[[lora]]` 部分的说明。

## 获取 LoRA 文件

LoRA 文件可通过 LoRA 微调获得。

查看 RWKV 官方的 [LoRA 微调教程](https://rwkv.cn/RWKV-Fine-Tuning/LoRA-Fine-Tuning) 了解 RWKV 模型 LoRA 微调的详细过程。

