# %%
from typing import Any

import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55],
    ]  # step (x^6)
)

# %%计算注意力分数
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, x_i)
print(attn_scores_2)
# %%注意力归一化
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention Weights (Temp):", attn_weights_2_tmp)
print("Sum of Attention Weights (Temp):", attn_weights_2_tmp.sum())


# %%softmax实现
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum()


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# %%
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# %%计算上下文向量
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("Context Vector:", context_vec_2)

# %% 3.3.2
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# %%矩阵乘法替代循环遍历
attn_scores = inputs @ inputs.T
print(attn_scores)

# %%归一化
attn_weights = torch.softmax(attn_scores, dim=-1)  # dim=-1 固定代表张量的最后一个维度
print(attn_weights)
print(sum(attn_weights[0]))
print(attn_weights[0])
# %%计算上下文向量
print(attn_weights, attn_weights.shape)
print("---------------")
print(inputs, inputs.shape)
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# %%
print("Previous 2nd context vector:", context_vec_2)

# %%3.4
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
keys = x_2 @ W_key
values = x_2 @ W_value
print("Query:", query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("inputs.shape:", inputs.shape)
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print("Attention Score:", attn_scores_22)

attn_scores_2 = query_2 @ keys.T
print("Attention Scores:", attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(
    attn_scores_2 / (d_k**0.5), dim=-1
)  # 先缩放再归一化，避免梯度归零
print("Attention Weights:", attn_weights_2)

# 计算上下文向量
context_vec_2 = attn_weights_2 @ values
print("Context Vector:", context_vec_2)
# %%
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, inputs):
        queries = inputs @ self.W_query
        keys = inputs @ self.W_key
        values = inputs @ self.W_value

        d_k = keys.shape[-1]
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)
        context_vecs = attn_weights @ values
        return context_vecs


# %%
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("Context Vectors from Self-Attention v1:\n", sa_v1(inputs))


# %%
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        queries = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)

        d_k = keys.shape[-1]
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)
        context_vecs = attn_weights @ values
        return context_vecs


# %%
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print("Context Vectors from Self-Attention v2:\n", sa_v2(inputs))

# %%练习3.1

print("Context Vectors from Self-Attention v1:\n", sa_v1(inputs))

print("Context Vectors from Self-Attention v2:\n", sa_v2(inputs))


sa_v1_apply_matrix_from_v2 = SelfAttention_v1(d_in, d_out)
sa_v1_apply_matrix_from_v2.W_query.data = sa_v2.W_query.weight.data.clone().T
sa_v1_apply_matrix_from_v2.W_key.data = sa_v2.W_key.weight.data.clone().T
sa_v1_apply_matrix_from_v2.W_value.data = sa_v2.W_value.weight.data.clone().T
print(
    "Context Vectors from Self-Attention v1 with matrices from v2:\n",
    sa_v1_apply_matrix_from_v2(inputs),
)

# %%3.5因果注意力
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# %%
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))  # 创建掩码
print(mask_simple)

# %%
masked_simple = attn_weights * mask_simple
print(masked_simple)

# %%
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# %%
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)  # 负无穷
print(masked)

# %%
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)
# %%
torch.manual_seed(123)
dropout = torch.nn.Dropout(p=0.5)
example = torch.ones(6, 6)
print("Before Dropout:\n", example)
print("After Dropout:\n", dropout(example))
# %%
torch.manual_seed(123)
print(dropout(attn_weights))
# %%3.5.3实现紧凑的因果注意力类
batch = torch.stack((inputs, inputs), dim=0)  # 模拟批处理输入
print("Batch shape:", batch.shape)  # (batch_size, seq_length, d_in)


# %%
class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length), diagonal=1
            ),  # torch.triu 是 PyTorch 中专门用于提取矩阵上三角部分 的函数，全称是 triangle upper（上三角），作用是把矩阵对角线以下的元素全部置为 0，只保留对角线及以上的元素。
        )

    def forward(self, inputs):
        batch_size, seq_length, d_in = inputs.shape
        queries = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)

        d_k = keys.shape[-1]
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:seq_length, :seq_length], -torch.inf
        )  # 在 PyTorch 中,操作  带有尾随下划线 是原地执行的, 避免不必要的 内存复制。
        attn_weights = torch.softmax(attn_scores / d_k**5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = attn_weights @ values
        return context_vecs


# %%
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalSelfAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)


# %%
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# %%
torch.manual_seed(123)
context_length = batch.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# %%
torch.manual_seed(123)
batch_single = batch[:, :1, :]  # [batch_size, sequence_length, feature_dim]
context_length = batch_single.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2
mha_2 = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha_2(batch_single)
print(context_vecs)
print("context_vecs_2_dim.shape:", context_vecs.shape)


# %%Listing 3.5 一个高效的多头注意力类 comm_save/Listing 3.5_comm_save.md
class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # 这个掩码要跟着模型一起走，所以用 register_buffer 注册为模型的一个属性，这样在模型保存和加载时，掩码也会被正确处理。

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 对注意力头进行切分(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # .view() 只改变张量形状，不改变数据；要求元素总数一致（通常用于按新维度重排表示）
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(
            2, 3
        )  # (b, num_heads, num_tokens, num_tokens)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(
            1, 2
        )  # (b, num_tokens, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


# %%
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# %%练习 3.3 初始化 GPT‐2 尺寸注意力模块
batch = torch.randn(2, 1024, 768)  # 模拟一个批处理输入，形状为 (batch_size, context_length, d_in)

batch_size, context_length, d_in = batch.shape
assert d_in == 768, "For GPT-2, the input dimension should be 768"
d_out = 768
mha_GPT_like = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
context_vecs_GPT_like = mha_GPT_like(batch)
print(context_vecs_GPT_like)
print("context_vecs_GPT_like.shape:", context_vecs_GPT_like.shape)