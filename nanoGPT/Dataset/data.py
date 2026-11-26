import os

with open(os.path.join(os.path.dirname(__file__), "input.txt"), "r", encoding="utf-8") as f:
    data = f.read()

print(f"length of dataset in characters: {len(data)}")
print(data[:1000])

# 创建可识别字符
characters = sorted(list(set(data)))
vocab_size = len(characters)
print("all the unique characters:", "".join(characters))
print(f"vocab size: {vocab_size}")

# 创建字符到索引的映射
stoi = {ch: i for i, ch in enumerate(characters)}
# 创建索引到字符的映射
itos = {i: ch for i, ch in enumerate(characters)}
encode=lambda s: [stoi[c] for c in s]  # 将字符串编码为整数列表
decode=lambda l: "".join([itos[i] for i in l])  # 将整数列表解码为字符串
print(encode("hii there"))
print(decode(encode("hii there")))

import torch
data = torch.tensor(encode(data), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000])

n=int(0.9*len(data))  # 90% 用于训练，10% 用于验证
train_data=data[:n]
val_data=data[n:]

print(f"length of train dataset in characters: {len(train_data)}")
print(f"length of validation dataset in characters: {len(val_data)}")