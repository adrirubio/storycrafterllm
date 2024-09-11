import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# load the BookCorpus dataset
dataset = load_dataset("bookcorpus")

# Split the dataset into train and test sets
split_dataset = dataset["train"].train_test_split(test_size=0.1)["train"]
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
        max_length=512,
        return_tensors=None
    )
    return tokenized_output

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

test_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          collate_fn=collate_fn)

# Print an example batch
for batch in train_loader:
    print(f"Batch input ids shape: {batch["input_ids"].shape}")
    print(f"Batch attention mask shape: {batch["attention_mask"].shape}")
    break









