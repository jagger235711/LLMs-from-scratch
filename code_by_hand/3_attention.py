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
print( sum(attn_weights[0]))
print(attn_weights[0])
# %%计算上下文向量
print(attn_weights,attn_weights.shape)
print("---------------")
print(inputs, inputs.shape)
all_context_vecs=attn_weights@inputs
print(all_context_vecs)

# %%
print("Previous 2nd context vector:", context_vec_2)

# %%3.4
