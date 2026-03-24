# %%
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

# %%
