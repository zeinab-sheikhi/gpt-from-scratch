import torch 
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.adamw


# Hyperparameters
BATCH_SIZE = 32 # how many independent sentences will we process in parallel?
BLOCK_SIZE = 8 # what is the maximum context length for predictions?
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200


torch.manual_seed(1337)

with open('data/input.txt', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in the text
chars = sorted(list(set(text)))
stoi = { ch: i for i, ch in enumerate(chars)} 
itos = { i: ch for i, ch in enumerate(chars)}
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
    ix = torch.randint(data.shape[0] - BLOCK_SIZE, (BATCH_SIZE, ))
    xb = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    yb = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table  = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C)
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


@torch.no_grad() # by using this context manager, pytorch will be a lot more efficient in terms of memory since it doesn't have to store all of the intermediate variables
def estimate_loss():
    # averages up the loss over multiple batches
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # some layers will have different behavior at inference time and training time, so it is a good practice to set the model behavior
    model.train()
    return out


model = BigramLanguageModel(vocab_size)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):
    if iter % EVAL_INTERVAL == 0:
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
