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
