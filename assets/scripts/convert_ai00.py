import collections
import numpy
import os
import torch
from safetensors.torch import serialize_file, load_file
import time
import hashlib
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Path to input pth model")
parser.add_argument(
    "--output",
    type=str,
    default="./converted.st",
    help="Path to output safetensors model",
)
args = parser.parse_args()


def rename_key(rename, name):
    for k, v in rename.items():
        if k in name:
            name = name.replace(k, v)
    return name


def convert_file(pt_filename: str, sf_filename: str, rename={}, transpose_names=[], model_info={}):
    loaded: collections.OrderedDict = torch.load(
        pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]

    kk = list(loaded.keys())
    version = 4
    for x in kk:
        if "ln_x" in x:
            version = max(5, version)
        if "gate.weight" in x:
            version = max(5.1, version)
        if int(version) == 5 and "att.time_decay" in x:
            if len(loaded[x].shape) > 1:
                if loaded[x].shape[1] > 1:
                    version = max(5.2, version)
        if "time_maa" in x:
            version = max(6, version)

    print(f"Model detected: v{version:.1f}")

    if version == 5.1:
        _, n_emb = loaded["emb.weight"].shape
        for k in kk:
            if "time_decay" in k or "time_faaaa" in k:
                # print(k, mm[k].shape)
                loaded[k] = (
                    loaded[k].unsqueeze(1).repeat(
                        1, n_emb // loaded[k].shape[0])
                )

    with torch.no_grad():
        for k in kk:
            new_k = rename_key(rename, k).lower()
            v = loaded[k].half()
            del loaded[k]
            for transpose_name in transpose_names:
                if transpose_name in new_k:
                    dims = len(v.shape)
                    v = v.transpose(dims - 2, dims - 1)
            print(f"{new_k}\t{v.shape}\t{v.dtype}")
            loaded[new_k] = {
                "dtype": str(v.dtype).split(".")[-1],
                "shape": v.shape,
                "data": v.numpy().tobytes(),
            }

    # 把 model_info 写入文件
    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    serialize_file(loaded, sf_filename, metadata=model_info)


# reload 函数读取 safetensors 文件中的 metadata 并打印出来
def read_metadata(sf_filename):
    with open(sf_filename, 'rb') as f:
        # 读取文件头部的JSON元数据
        header_size = int.from_bytes(
            f.read(8), byteorder='little', signed=False)
        metadata_json = f.read(header_size)
        return json.loads(metadata_json)


if __name__ == "__main__":

    print(f"请选择语言 (Language): \n1.中文\n2.English\n")
    choice = input("请输入序号 (Enter 1 or 2; default 1): ")
    if choice == "1":
        language = "zh"
        print("\n已选择中文\n")
    elif choice == "2":
        language = "en"
        print("\nUse English\n")
    else:
        language = "zh"
        print("\n已选择中文\n")

    # 假设这里的 model_type_dict 是一个字典，包含了中英文对应的模型类型描述
    model_type_dict = {
        "zh": {
            "ask": {
                "ask0": "输入模型类型 (默认rwkv)",
                "ask1": "请选择要转换的模型类型: ",
                "ask2": "请选择要转换模型的参数量: ",
                "ask3": "请输入作者名: ",
                "ask4": "请输入模型说明: ",
                "ask5": "请输入RWKV版本 (默认x060): ",
            },
            "model_type": {
                "rwkv": "RWKV 模型",
                "lora": "RWKV LoRA",
                "state": "RWKV init State",
            },
            "error": "输入错误，请重新输入！",
        },
        "en": {
            "ask": {
                "ask0": "Input model type (default rwkv)",
                "ask1": "Please select the model you want to convert:",
                "ask2": "Please select the number of parameters for the model you want to convert:",
                "ask3": "Please enter the author name:",
                "ask4": "Please enter the model description:",
                "ask5": "Please enter RWKV version (default x060):",

            },
            "model_type": {
                "rwkv": "RWKV model",
                "lora": "RWKV LoRA",
                "state": "RWKV init State",
            },
            "error": "Input error, please re-enter!"
        }
    }

    print(f"\n{model_type_dict[language]['ask']['ask1']}")
    for k, v in model_type_dict[language]["model_type"].items():
        print(f"{k}: {v}")

    choice = input(f"{model_type_dict[language]['ask']['ask0']}:")
    if choice in model_type_dict[language]["model_type"]:
        model_type = choice
        print(f"\n已选择 {model_type_dict[language]['model_type'][model_type]}\n")
    else:
        model_type = "rwkv"
        print(f"\n已选择 {model_type_dict[language]['model_type'][model_type]}\n")

    print(f"\n{model_type_dict[language]['ask']['ask2']}")
    print(f"(1)1B5\n(2)3B\n(3)7B\n(4)14B")
    choice = input("Enter the number 1 - 4 (default 3): ")
    if choice == "1":
        model_size = "1B5"
    elif choice == "2":
        model_size = "3B"
    elif choice == "3":
        model_size = "7B"
    elif choice == "4":
        model_size = "14B"
    else:
        model_size = "7B"

    rwkv_version = input(f"{model_type_dict[language]['ask']['ask5']}")
    # 检查 rwkv_version 是否符合x060这样的格式
    if not rwkv_version.startswith("x"):
        rwkv_version = "x060"

    elif len(rwkv_version) != 4:
        rwkv_version = "x060"

    # 检查 x 后三位是否是数字
    elif not rwkv_version[1:].isdigit():
        rwkv_version = "x060"

    author_name = input(f"{model_type_dict[language]['ask']['ask3']}")
    model_readme = input(f"{model_type_dict[language]['ask']['ask4']}")

    if model_type == "rwkv":
        sf_filename = f"rwkv_{model_size}.st"
    elif model_type == "lora":
        sf_filename = f"rwkv_{model_size}.lora"
    elif model_type == "state":
        sf_filename = f"rwkv_{model_size}.state"
    else:
        print("输入错误，请重新输入！")
        exit()

    current_time = time.time()

    def get_sha(file_path):
        with open(file_path, 'rb') as f:
            sha1 = hashlib.sha1()
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha1.update(data)
        return sha1.hexdigest()

    pth_SHA = get_sha(args.input)

    model_info = {
        "model_type": model_type,
        "model_size": model_size,
        "author_name": author_name,
        "model_readme": model_readme,
        "covertime": str(current_time),
        "pth_SHA": pth_SHA,
        "rwkv_version": rwkv_version,
    }

    print(f"正在转换模型: {model_info}")

    convert_file(args.input, args.output,
                 rename={"time_faaaa": "time_first", "time_maa": "time_mix",
                         "lora_A": "lora.0", "lora_B": "lora.1"},
                 transpose_names=[
                     "time_mix_w1", "time_mix_w2", "time_decay_w1", "time_decay_w2",
                     "w1", "w2", "a1", "a2", "g1", "g2", "v1", "v2",
                     "time_state", "lora.0"])
    print(f"Saved to {args.output}")

    print(f"{args.output} __metadata__ :\n")
    read_metadata = read_metadata(args.output)
    print(read_metadata['__metadata__'])
    exit()
