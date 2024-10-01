import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import tiktoken  # Make sure tiktoken is imported

enc = tiktoken.get_encoding("gpt2")  # Initialize the GPT-2 tokenizer

# Load the GPT-2 tokenizer (or your specific tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the GPTLanguageModel class (the one you used for training)
# Ensure that this matches exactly the training-time definition
class Head(nn.Module):
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
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        out = self.proj(concatenated)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
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
    def __init__(self, n_embd, n_head, dropout=0.1):
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
        T = min(T, self.block_size)
        idx = idx[:, :T]
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb.unsqueeze(0)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            targets = targets[:, :T]
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Now that we have the model definition, let's load the weights and perform inference
device = torch.device('cpu')  # Use 'cuda' if you have a GPU

# Hyperparameters (match these with the ones you used for training)
vocab_size = 50257
n_heads = 8
n_layers = 6
head_size = 64
n_embd = 512
block_size = 128
dropout = 0.1
learning_rate = 3e-4
weight_decay = 0.1

# Create an instance of the model
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layers, n_heads).to(device)

# Load the trained weights
model.load_state_dict(torch.load("model_weights.pth", map_location=device))

# Set the model to evaluation mode
model.eval()

# Prompt
context = torch.tensor([enc.encode("Once upon a time there was a knight called Bob and he rode into his greatest battle yet")], dtype=torch.long, device=device)

# Test generation with a higher number of tokens and adjusted temperature
max_new_tokens = 200  # Increase the token limit for a longer generation
temperature = 0.8  # More focused, less random
generated_text_idx = model.generate(context, max_new_tokens)
generated_text = enc.decode(generated_text_idx[0].tolist())

print(f"Generated text: {generated_text}")
