import torch
from transformers import AutoTokenizer

from llm import LLM
from utils import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    NUM_EMBED,
    NUM_HEAD,
    NUM_LAYER,
    MAX_ITER,
    EVAL_INTER,
    encode,
    decode,
    get_batch,
    save_checkpoint,
    estimate_loss,
)

data_raw = open("data/js_doc.txt", encoding="utf-8").read()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

data = encode(data_raw, tokenizer)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = LLM(
    vocab_size=vocab_size,
    context_size=BLOCK_SIZE,
    dim_emb=NUM_EMBED,
    num_layers=NUM_LAYER,
    attn_num_heads=NUM_HEAD,
    emb_dropout=DROPOUT,
    ffd_dropout=DROPOUT,
)
m = model.to(DEVICE)
print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params")

optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for step in range(MAX_ITER):
    if step % EVAL_INTER == 0 or step == MAX_ITER - 1:
        train_loss = estimate_loss(train_data, m, BLOCK_SIZE, BATCH_SIZE)
        val_loss = estimate_loss(val_data, m, BLOCK_SIZE, BATCH_SIZE)
        print(f"step {step:10} | train {train_loss:6.4f} | val {val_loss:6.4f}")

    xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
    logits = m(xb)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

save_checkpoint(m, "checkpoints", step)

ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
out = m.generate(ctx, max_seq_len=100)
print(decode(out[0], tokenizer))
