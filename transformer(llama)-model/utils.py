import os
import torch
from torch import Tensor

BATCH_SIZE = 64
BLOCK_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT = 0.2
LEARNING_RATE = 3e-4
NUM_EMBED = 384
NUM_HEAD = 6
NUM_LAYER = 6
MAX_ITER = 5000
EVAL_INTER = 500
EVAL_ITERS = 200


def encode(text_seq: str, tokenizer) -> Tensor:
    return torch.tensor(tokenizer.encode(text_seq), dtype=torch.long)


def decode(enc_sec: Tensor, tokenizer) -> str:
    return tokenizer.decode(enc_sec.tolist())


def get_batch(data: Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(data: Tensor, model, block_size: int, batch_size: int) -> float:
    model.eval()
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(model, path: str, epoch: int):
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, f"model_{epoch}.pt")
    torch.save(model.state_dict(), fpath)


def load_checkpoint(model_class, path: str, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path))
    return model
