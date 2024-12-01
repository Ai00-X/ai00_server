# RWKV 基座模型


## RWKV-6-World 基底模型

RWKV-6-World 系列模型均为基底模型（base model ，又称预训练模型）。基底模型在自然语言处理等领域的大规模数据集上进行了训练，具备较强的泛化能力和丰富的知识储备。

但为了保持泛化能力和通用性，基底模型通常不会针对任何一类任务作优化。针对一些垂直的下游任务，可能需要[微调 RWKV 基底模型](https://rwkv.cn/RWKV-Fine-Tuning/Introduction)才能获得更好的任务效果。


- [Hugging Face 主站](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main)
- [Hugging Face 镜像站](https://hf-mirror.com/BlinkDL/rwkv-6-world/tree/main)（国内可访问）
- [ModelScope 仓库](https://modelscope.cn/models/Blink_DL/rwkv-6-world/files)（国内可访问）
- [WiseModel 仓库](https://wisemodel.cn/models/rwkv4fun/Rwkv-6-world/file)（国内可访问）
- BT 种子下载：[1B6](https://rwkv.cn/files/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.torrent) | [3B](https://rwkv.cn/files/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth.torrent) 
| [7B](https://rwkv.cn/files/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.torrent) | [14B](https://rwkv.cn/files/RWKV-x060-World-14B-v2.1-20240719-ctx4096.pth.torrent)

## RWKV-6 中文小说模型


RWKV-6-ChnNovel 系列中文小说模型基于 RWKV-6-World 模型微调而来，在小说续写、小说扩写、角色扮演方面有非常好的效果。

小说模型的具体用法，请参考 [RWKV-6-ChnNovel 中文小说模型教程](https://rwkv.cn/news/read?id=4264)


- [Hugging Face 主站](https://hf-mirror.com/BlinkDL/rwkv-6-misc/tree/main)
- [Hugging Face 镜像站](https://hf-mirror.com/BlinkDL/rwkv-6-misc/tree/main)（国内可访问）
- [ModelScope 仓库](https://modelscope.cn/models/Blink_DL/rwkv-6-misc/files)（国内可访问）
- [WiseModel 仓库](https://wisemodel.cn/models/rwkv4fun/RWKV-6-ChnNovel/file)（国内可访问）
- BT 种子下载：[1B6 中文小说模型](https://rwkv.cn/files/RWKV-x060-ChnNovel-1B-20240807-ctx4096.pth.torrent) | [3B 中文小说模型](https://rwkv.cn/files/RWKV-x060-ChnNovel-3B-20240807-ctx4096.pth.torrent) 
| [7B 中文小说模型](https://rwkv.cn/files/RWKV-x060-ChnNovel-7B-20240803-ctx4096.pth.torrent) | [14B 中文小说模型](https://rwkv.cn/files/RWKV-x060-ChnNovel-14B-20240805-ctx4096.pth.torrent)

## RWKV-6 日文模型

RWKV-6-Jpn 系列日语模型基于 RWKV-6-World 模型微调而来，在日语任务和基准测试上表现良好。

- [Hugging Face 主站](https://hf-mirror.com/BlinkDL/rwkv-6-misc/tree/main)
- [Hugging Face 镜像站](https://hf-mirror.com/BlinkDL/rwkv-6-misc/tree/main)（国内可访问）