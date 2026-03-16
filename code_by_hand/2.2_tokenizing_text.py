# %%
import os

obs_path = os.path.abspath(__file__)
print(obs_path)

target_dir = "LLMs-from-scratch"
# 切分并拼接出目标路径
if target_dir in obs_path:
    # 切分后取前半部分 + 目标字符串
    result = obs_path.split(target_dir)[0] + target_dir
else:
    # 如果路径中没有目标字符串，做容错处理
    result = "路径中未找到指定目录"

print("提取的路径：", result)

articlePath = result + "/ch02/01_main-chapter-code/the-verdict.txt"
# read in
with open(articlePath, "r", encoding="UTF-8") as f:
    raw_text = f.read()
print("Total_number", len(raw_text))
print(raw_text[:99])

# %% split
import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# %% [markdown]
## 列表推导式
# 这种顺序是故意设计成接近数学集合记号的：
# 数学写法：
# { x² | x ∈ S }
# 列表推导式：
# [item.strip() for item in preprocessed if item.strip()]
# ---
# 对比等价的 for 循环：
# # for循环：先写遍历，再写条件，最后写要做的操作
# result = []
# for item in preprocessed:
#     if item.strip():
#         result.append(item.strip())
# ---
# 为什么这样设计？
# 1. 表达式在最前面 — 一眼就能看出最终结果是什么（取 item.strip()）
# 2. 从左到右读 — "对preprocessed中的每个item，如果item.strip()非空，就取item.strip()"
# 3. 数学直觉 — 类似 {表达式 | 条件} 的写法
# 如果写成 [for if 表达式] 这种顺序，反而会和传统编程语言的语法不一致。
# %% 将token转化为id
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# creating vocabulary
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


# %% tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


# %%
tokenizer = SimpleTokenizerV1(vocab)
text = (
    """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
)
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))
# %%
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))
# %%
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# %% SimpleTokenizerV2
class SimpleTokenizerV2:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


# %%
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# %% tiktoken子词分词，socket不可用
import tiktoken

# from transformers import GPT2Tokenizer


# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# %%
strings = tokenizer.decode(integers)
print(strings)
# %%
text = "Akwirw ier"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# %%
with open(articlePath, "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
# %%
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]

print(f"x: {x}")
print(f"y:           {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
# %%
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# %%
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# %%
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


# %%
with open(articlePath, "r", encoding="utf-8") as f:
    raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

# %% 不同参数的基于滑动窗口的数据加载器
with open(articlePath, "r", encoding="utf-8") as f:
    raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=2, max_length=4, stride=3, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    print("-------------------")
    print("                   " + str(next(data_iter)))
    print("-------------------")
    print("                   " + str(next(data_iter)))

# %% [markdown]

# 单个batch会输出两个tenser,后一个tensor是前一个的预测目标

# [tensor([[  40,  367, 2885, 1464],[1464, 1807, 3619,  402]]),
#            tensor([[ 367, 2885, 1464, 1807],[1807, 3619,  402,  271]])]
# -------------------
#                    [tensor([[  402,   271, 10899,  2138],
#         [ 2138,   257,  7026, 15632]]), tensor([[  271, 10899,  2138,   257],
#         [  257,  7026, 15632,   438]])]
# -------------------
#                    [tensor([[15632,   438,  2016,   257],
#         [  257,   922,  5891,  1576]]), tensor([[ 438, 2016,  257,  922],
#         [ 922, 5891, 1576,  438]])]

# %%
