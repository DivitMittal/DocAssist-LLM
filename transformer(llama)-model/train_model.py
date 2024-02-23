import torch
from transformers import AutoTokenizer
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
    save_model_to_chekpoint,
    estimate_loss,
)

# load model from checkpoint
# m = load_model_from_checkpoint(Transformer,vocab_size=vocab_size)

# example to decode sequence
# enc_sec = m.generate(idx=torch.zeros((1,1), dtype=torch.long),
# max_new_tokens=20)[0].tolist()
# print(decode(vocab=vocab, enc_sec=enc_sec))

path_do_data = "data/js_doc.txt"
data_raw = open(path_do_data, encoding="utf-8").read()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

data = encode(text_seq=data_raw, tokenizer=tokenizer)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# train a new model
model = Transformer(
    vocab_size=vocab_size,
    num_embed=NUM_EMBED,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEAD,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)
m = model.to(DEVICE)
print(
    "Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6)
)
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for step in range(MAX_ITER):

    if step % EVAL_INTER == 0 or step == MAX_ITER - 1:
        loss_train = estimate_loss(
            data=train_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        loss_val = estimate_loss(
            data=val_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        print("step {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))

    xb, yb = get_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

save_model_to_chekpoint(model=m, path_to_checkpoint="checkpoints", epoch=step)

# generate some output based on the context
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(
    decode(
        enc_sec=m.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
        tokenizer=tokenizer,
    )
)
