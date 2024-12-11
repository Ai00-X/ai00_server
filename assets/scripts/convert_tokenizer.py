#!/usr/bin/python

import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    default="rwkv_vocab_v20230424.txt",
    help="Path to input txt")
parser.add_argument(
    "--output",
    type=str,
    default="rwkv_vocab_v20230424.json",
    help="Path to output JSON",
)
args = parser.parse_args()


I_TO_TOKEN = {}
lines = open(args.input, "r", encoding="utf-8").readlines()
for l in lines:
    idx = int(l[:l.index(' ')])
    x = eval(l[l.index(' '):l.rindex(' ')])
    if not isinstance(x, str):
        x = list(x)
    I_TO_TOKEN[idx] = x

out = open(args.output, "w")
out.write(json.dumps(I_TO_TOKEN, indent=4))
