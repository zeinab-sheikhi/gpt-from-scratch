import torch 
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.adamw


# Hyperparameters
batch_size = 32  # how many independent sentences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32

torch.manual_seed(1337)

with open('data/input.txt', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in the text
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)} 
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(itos[ix] for ix in l)
vocab_size = len(stoi)

# Train and val splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # this function averages the loss over muliple batches
    data = train_data if split == 'train' else val_data
    ix = torch.randint(data.shape[0] - block_size, (batch_size, ))
    xb = torch.stack([data[i:i + block_size] for i in ix])
    yb = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, vocab_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb   # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size) 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_tokens=1000):
        for _ in range(max_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_index), dim=1)
        return idx


@torch.no_grad()  # by using this context manager, pytorch will be a lot more efficient in terms of memory since it doesn't have to store all of the intermediate variables
def estimate_loss():
    # averages up the loss over multiple batches
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # some layers will have different behavior at inference time and training time, so it is a good practice to set the model behavior
    model.train()
    return out


model = BigramLanguageModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['eval']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context)[0].tolist()))
