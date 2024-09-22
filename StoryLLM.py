import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Define hyperparameters
vocab_size = 50257
n_heads = 8
n_layers = 6
head_size = 64
n_embd = 512
block_size = 128
dropout = 0.1
learning_rate = 3e-4
weight_decay = 0.1

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
train_dataset = train_dataset.map(tokenize_batch, batched=True, batch_size=16, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize_batch, batched=True, batch_size=16, remove_columns=["text"])

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

# Define model
class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
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

    def __init__(self, n_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads *  head_size, n_embd)
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


class FeedForward(nn.Module):
    """ A simple linear layer followed by non-linearity """

    def __init__(self, n_embd, dropout=0.1, expansion_factor=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion_factor * n_embd),
            nn.ReLU(),
            nn.Linear(expansion_factor * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout=0.1):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, device="cpu"):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.1, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb.unsqueeze(0)  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # Crop to the last block_size tokens
            logits, _ = self(idx_cond)  # Get Predictions
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)  # Get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Samples from the distribution
            idx = torch.cat((idx, idx_next), dim=1)  # Append sampled index
        return idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")


# Instantiate the model
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layers, n_heads, device=device)

# Move the model to the GPU (if available)
model = model.to(device)

# Loss_fn and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def batch_gh(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)  # Initialize arrays here
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()  # Set model to training mode
        t0 = datetime.now()
        train_loss = []
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Create targets by shifting inputs by one position
            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()

            # Zero parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, loss = model(inputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # Get average train_loss
        train_loss = np.mean(train_loss)

        model.eval()  # Set model to evaluation mode
        test_loss = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Create targets by shifting inputs by one position
                targets = inputs[:, 1:].contiguous()
                inputs = inputs[:, :-1].contiguous()  # Corrected

                outputs, loss = model(inputs, targets)
                test_loss.append(loss.item())

            test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
              Test Loss: {test_loss:.4f}, Duration: {dt}')

    return train_losses, test_losses

train_losses, test_losses = batch_gh(model, criterion, optimizer, train_loader, test_loader, epochs=20)

# Plot loss
plt.plot(train_losses, label="train_loss")
plt.plot(test_losses, label="test_loss")
plt.legend()
plt.show()

# Save model weights
model_save_path = "/home/adrian/Documents/StoryCrafterLLM/model_weights.pth"
torch.save(model.state_dict(), model_save_path)  # Corrected save method
print(f"Model saved to {model_save_path}")
