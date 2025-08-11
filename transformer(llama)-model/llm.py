import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Sequential, Embedding, Linear, Dropout

from transformer import TransformerBlock, RMSNorm


class LLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        dim_emb: int,
        num_layers: int,
        attn_num_heads: int,
        emb_dropout: float = 0.0,
        ffd_bias: bool = True,
        ffd_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.context_size = context_size
        self.token_embedding = Embedding(vocab_size, dim_emb)
        self.emb_dropout = Dropout(emb_dropout)
        self.transformer = Sequential(
            *[
                TransformerBlock(
                    context_size,
                    dim_emb,
                    attn_num_heads,
                    attn_causal=True,
                    ffd_bias=ffd_bias,
                    ffd_dropout=ffd_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(dim_emb)
        self.projection_head = Linear(dim_emb, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)
        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.projection_head(x)

        return x

    @torch.inference_mode()
    def generate(
        self,
        inputs: Tensor,
        max_seq_len: int,
        temperature: float = 1.0,
        top_p: float = None,
    ) -> Tensor:
        for _ in range(max_seq_len):
            inputs_cond = (
                inputs
                if inputs.size(1) <= self.context_size
                else inputs[:, -self.context_size :]
            )
            logits = self(inputs_cond)[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff_mask = cumulative_probs > top_p
                cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
                cutoff_mask[..., 0] = False
                sorted_probs[cutoff_mask] = 0.0
                probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, next_token), dim=-1)

        return inputs
