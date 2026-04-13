# %%
import torch
from ch4_impl_GPT import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # 缩减上下文长度
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

# %%文本生成
import tiktoken
from ch4_impl_GPT import generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %%
inputs = torch.tensor(
    [[16833, 3626, 6100], [40, 1107, 588]]  # ["every effort moves",
)  # "I really like"]

targets = torch.tensor(
    [[3626, 6100, 345], [1107, 588, 11311]]  # [" effort moves you",
)  # " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)
    probas = torch.softmax(
        logits, dim=-1
    )  # 词汇表中每个标记的概率 (batch_size, seq_len, vocab_size)
    print(probas.shape)

# %%
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

# %%
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:" f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# %%


# 临时设置打印精度为8位小数，同时开启科学记数法（可选）
# torch.set_printoptions(precision=8, sci_mode=True)

text_idx = 0
target_probas_1 = probas[
    text_idx, [0, 1, 2], targets[text_idx]
]  # 概率 → 目标概率 从 3D 概率表里，精准抽出「每个位置真实标签对应的概率」
# probas[ text_idx ,   [0,1,2],   targets[text_idx] ]
#           ↑           ↑             ↑
#       选哪个样本   选哪几个位置   选真实词的概率
print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)
# %%文本评估
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
# %%
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
# %%
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

# %%使用cross_entropy函数自动处理
print(
    "Logits shape:", logits.shape
)  # batch size, number of tokens, and vocabulary size.
print("Targets shape:", targets.shape)  # batch size and number of tokens.
# %%
logits_flat = logits.flatten(0, 1)  # combining them over the batch dimension
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

# %%
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# %%
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# %%
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# %%
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# %%
from ch2_tokenizing_text import create_dataloader_v1

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

# %%
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)  # (batch_size,num_tokens)


# %%
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss
