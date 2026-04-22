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


# %%
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:  # 如果没有指定num_batches，就计算整个数据加载器的平均损失
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


# %%
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 设置模型为评估模式 （这会禁用dropout等训练时特有的行为）
    with torch.no_grad():  # 禁用评估过程中不需要的梯度跟踪，以减少计算开销
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# %%
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


# %%5.2训练大语言模型
def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,  # 10
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1

    for epoch in range(num_epochs):  # 开始训练循环 一个轮次是指一次完整 地遍历训练集。
        model.train()  # 设置模型为训练模式
        for (
            input_batch,
            target_batch,
        ) in train_loader:  # 遍历训练数据加载器 所有的遍历一遍训练集才遍历一边
            optimizer.zero_grad()  # 清零优化器的梯度
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )  # 计算当前批次的损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            token_seen += (
                input_batch.numel()
            )  # 更新已见标记数量 .numel() 统计一个张量中 总共有多少个元素（返回一个整数）。
            global_step += 1  # 更新全局步数

            if global_step % eval_freq == 0:  # 每隔eval_freq步进行评估
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )  # 计算训练损失和验证损失
                train_losses.append(train_loss)  # 记录训练损失
                val_losses.append(val_loss)  # 记录验证损失
                track_tokens_seen.append(token_seen)  # 记录已见标记数量

                print(
                    f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Tokens Seen: {token_seen}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )  # 打印当前训练状态

                # 生成文本示例以检查模型性能
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )  # 每反向传播一次就生成一次文本示例
    return train_losses, val_losses, track_tokens_seen


# %%
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
# %% 5.3
model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %% 温度缩放
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)  # 未softmax的概率
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])

# %%
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])
# %%
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])


# %%
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


print_sampled_tokens(probas)


# %%
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


# %%
temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(
        x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}"
    )
ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()


# %%练习5.1
print(scaled_probas)
# 分别打印当前温度下的指定token概率
token_index = vocab["pizza"]
for index, probas in enumerate(scaled_probas):
    p = torch.softmax(probas, dim=0)
    print("温度：%d，概率：%f" % (index, p[token_index]))

# %%
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)

# %%
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float("-inf")),
    other=next_token_logits,
)
print(new_logits)

# %%
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)


# %%
def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
        if temperature > 0.0:  # 温度缩放
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:  # 直接取最大概率的token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# %%
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4,
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %%练习5.2
# 强调文本准确性的生成可以采用较低的温度（如0.7或更低）和较小的top_k值（如10或更小），以确保生成的文本更倾向于选择概率较高的词汇，从而提高文本的准确性和连贯性。例如给定知识库问答

# %%练习5.3
# 将温度设置的非常小

# %%5.4
torch.save(model.state_dict(), "model.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# %%
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth",
)

# %%
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
# %% 5.5 加载预训练权重
import urllib.request

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split("/")[-1]
urllib.request.urlretrieve(url, filename)

# %%
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# %%
print("Settings:", settings) 
print("Parameter dictionary keys:", params.keys())
# %%
