import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Define hyperparameters
n_heads = 8
head_size = 64
n_embed = 512
block_size = 128
dropout = 0.1

# load the BookCorpus dataset
dataset = load_dataset("bookcorpus")

# Split the dataset into train and test sets
split_dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Print the size of the train and test sets
print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Define the tokenization function
def tokenize_batch(batch):
    tokenized_output = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    return {"input_ids": tokenized_output["input_ids"], "attention_mask": tokenized_output["attention_mask"]}

# Apply tokenization to the train and test datasets
train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

# Update dataset format to include input_ids and attention_mask
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Print some examples
print(f"Example train data: {train_dataset[0]}")
print(f"Example test data: {test_dataset[0]}")


# Create a custom collate function
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# Create batches
batch_size = 8

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          collate_fn=collate_fn)

# Print an example batch
for batch in train_loader:
    print(f"Batch input ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention mask shape: {batch['attention_mask'].shape}")
    break

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        assert C == self.key.in_features, f"Input size {C} doesn't match expected size {self.key.in_features}"

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, n_heads, head_size, n_embed, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads *  head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Collects the outputs from each head
        head_outputs = [head(x) for head in self.heads]
        # Concatenate the outputs
        concatenated = torch.cat(head_outputs, dim=-1)
        # Apply linear transformation and dropout
        out = self.proj(concatenated)
        out = self.dropout(out)
        return out

