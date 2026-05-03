# %%
import json
import os
import urllib


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))

# %%
print("Example entry:\n", data[50])

print("Another Example entry:\n", data[999])


# %%
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


# %%Phi-3提示风格 format
def format_input_phi3(entry):
    instruction_text = (
        # f"Below is an instruction that describes a task. "
        # f"Write a response that appropriately completes the request."
        f"\n\n<|user|>\n{entry['instruction']}"
    )
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


# %%
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

# %%
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

# %% Phi-3提示风格
model_input = format_input_phi3(data[50])
desired_response = f"\n\n<|assistant|>\n{data[50]['output']}"
print(model_input + desired_response)

# %%
model_input = format_input_phi3(data[999])
desired_response = f"\n\n<|assistant|>\n{data[999]['output']}"
print(model_input + desired_response)

# %%划分数据集
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion : train_portion + test_portion]
val_data = data[train_portion + test_portion :]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# %%
import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# %%
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


# %%
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_list = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # 移除最后一个token
        inputs_list.append(inputs)

    inputs_tensor = torch.stack(inputs_list).to(device)
    return inputs_tensor


# %%
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (inputs_1, inputs_2, inputs_3)
print(custom_collate_draft_1(batch))


# %%
def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)


# %%
def custom_collate_fn(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # 用ignore_index替换目标中除第一个填充标记之外的所有标记
        mask = targets == pad_token_id
        indices = torch.nonzero(  # 获取mask中非0位置索引
            mask
        ).squeeze()  # squeeze() 删除张量中所有长度为 1 的维度
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # 除了第一个都忽略

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# %%
inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)
# %%
logits_1 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5]])
targets_1 = torch.tensor([0, 1])  # Correct token indices to generate
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)

# %%
logits_2 = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
targets_2 = torch.tensor([0, 1, 1])
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)

# %%
targets_3 = torch.tensor([0, 1, -100])
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)  # 默认的忽略标记是-100
print(loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)

# %%练习7.2 掩码指令文本


def custom_collate_mask_instruction_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",
    mask_instruction=True,
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # 用ignore_index替换目标中除第一个填充标记之外的所有标记
        mask = targets == pad_token_id
        indices = torch.nonzero(  # 获取mask中非0位置索引
            mask
        ).squeeze()  # squeeze() 删除张量中所有长度为 1 的维度
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # 除了第一个都忽略

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        if mask_instruction == True:  # TODO：掩码指令文本
            pass

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


inputs, targets = custom_collate_mask_instruction_fn(batch)
print(inputs)
print(targets)

# %%ch7.4
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # if torch.backends.mps.is_available(): # device = torch.device("mps")"
print("Device:", device)

# %%
from functools import partial

customized_collate_fn = partial(
    custom_collate_fn, device=device, allowed_max_length=1024
)  # partial 的作用是：固定一个函数的部分参数，返回一个新的可调用对象。

# %%
from torch.utils.data import DataLoader

num_workers = 0  # 把 num_workers 设为非 0（比如 4、8）后，PyTorch 会创建多个子进程来并行加载数据，这会显著影响数据加载的速度和内存使用。
batch_size = 8
torch.manual_seed(123)
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

# %%
print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)#[batch_size,num_tokens] 维度1的大小取决于最长的那个样本

# %%
