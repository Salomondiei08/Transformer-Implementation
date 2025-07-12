# mini_transformer.py
"""
Tiny Transformer implementation from scratch (PyTorch)
=====================================================
This script trains a small Transformer model on two toy sequence-to-sequence
tasks:
    1. Copy – target is identical to the input sequence.
    2. Reverse – target is the reversed input sequence.

It illustrates the core Transformer components:
    • Token Embeddings
    • Positional Encoding (sinusoidal)
    • Multi-Head Self-Attention
    • Position-wise Feed-Forward Network
    • Layer Normalization & residual connections

After training, the script can visualise the encoder-decoder attention maps
for a single example using Matplotlib.

Requirements:
    pip install torch matplotlib

Run example:
    python mini_transformer.py --task copy --epochs 200
    python mini_transformer.py --task reverse --visualise
"""
from __future__ import annotations

import argparse
import math
import random
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- Utility -------------------------------------------------

def generate_batch(batch_size: int, seq_len: int, vocab_size: int, task: str
                   ) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Generate a batch of toy data for copy/reverse tasks."""
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    if task == "copy":
        tgt = src.clone()
    elif task == "reverse":
        tgt = torch.flip(src, dims=[1])
    else:
        raise ValueError("Task must be 'copy' or 'reverse'.")
    return src, tgt


# --------------------------- Model Components -----------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.attn_weights: torch.Tensor | None = None  # to store latest attn for visualisation

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        batch, seq_len, _ = x.size()
        return x.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch, n_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq_len, n_heads * d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # Project and split heads
        q = self._split_heads(self.q_linear(q))
        k = self._split_heads(self.k_linear(k))
        v = self._split_heads(self.v_linear(v))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, q_len, k_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        self.attn_weights = attn.detach().cpu()  # store for visualisation

        context = torch.matmul(attn, v)  # (batch, heads, q_len, d_k)
        context = self._combine_heads(context)
        return self.out_linear(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention
        attn_out = self.mha(src, src, src, src_mask)
        src = self.ln1(src + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ff(src)
        src = self.ln2(src + self.dropout(ff_out))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, n_heads)
        self.cross_mha = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention
        _tgt = self.self_mha(tgt, tgt, tgt, tgt_mask)
        tgt = self.ln1(tgt + self.dropout(_tgt))
        # Cross-attention
        _tgt2 = self.cross_mha(tgt, memory, memory, memory_mask)
        tgt = self.ln2(tgt + self.dropout(_tgt2))
        # Feed-forward
        tgt = self.ln3(tgt + self.dropout(self.ff(tgt)))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.attn_maps: list[torch.Tensor] = []  # store for visualisation

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.attn_maps.clear()
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
            # store the encoder-decoder attention from this layer
            self.attn_maps.append(layer.cross_mha.attn_weights)
        return tgt


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        num_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 64,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.encoder = TransformerEncoder(num_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, n_heads, d_ff, dropout)

        self.out_proj = nn.Linear(d_model, vocab_size)

    def _subsequent_mask(self, sz: int) -> torch.Tensor:
        # Mask out future positions
        mask = torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1).bool()
        return ~mask  # True where allowed

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src, tgt: (batch, seq_len)
        src_mask = None  # no padding in toy data
        tgt_mask = self._subsequent_mask(tgt.size(1)).unsqueeze(0)  # (1, tgt_len, tgt_len)

        # Encoder
        src_emb = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_mask)

        # Decoder (shifted right)
        tgt_emb = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))
        out = self.decoder(tgt_emb, memory, tgt_mask, src_mask)
        logits = self.out_proj(out)  # (batch, seq_len, vocab)
        return logits


# --------------------------- Training & Evaluation ------------------------------------

def train(model: TinyTransformer, task: str, epochs: int, batch_size: int, seq_len: int,
          vocab_size: int, lr: float) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        src, tgt = generate_batch(batch_size, seq_len, vocab_size, task)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        # decoder input is target with prepended BOS token (here we use 0)
        bos = torch.zeros(batch_size, 1, dtype=torch.long, device=DEVICE)
        decoder_in = torch.cat([bos, tgt[:, :-1]], dim=1)

        logits = model(src, decoder_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}, Loss: {loss.item():.4f}")

    print("Training complete.\n")


def greedy_decode(model: TinyTransformer, src: torch.Tensor, max_len: int, bos_id: int = 0
                   ) -> torch.Tensor:
    """Greedy decode for inference."""
    model.eval()
    src = src.to(DEVICE)
    memory = model.encoder(model.pos_enc(model.embedding(src) * math.sqrt(model.d_model)))

    batch_size = src.size(0)
    ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=DEVICE)

    for _ in range(max_len):
        out = model.decoder(model.pos_enc(model.embedding(ys) * math.sqrt(model.d_model)),
                            memory, model._subsequent_mask(ys.size(1)).unsqueeze(0))
        logits = model.out_proj(out[:, -1])  # last timestep
        next_word = torch.argmax(logits, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_word], dim=1)
    return ys[:, 1:]  # drop BOS


# --------------------------- Visualisation --------------------------------------------

def show_attention_maps(model: TinyTransformer, src: torch.Tensor, trg_pred: torch.Tensor,
                        idx2tok: list[str]):
    """Plot encoder-decoder attention heatmaps for the first layer."""
    maps = model.decoder.attn_maps  # list[num_layers] each (batch, heads, tgt_len, src_len)
    if not maps:
        print("No attention maps found. Run a forward pass first.")
        return
    attn = maps[0][0]  # (heads, tgt_len, src_len) – first layer, first batch
    num_heads = attn.size(0)

    fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 3, 3))
    if num_heads == 1:
        axes = [axes]

    src_tokens = [idx2tok[i] for i in src[0].tolist()]
    tgt_tokens = [idx2tok[i] for i in trg_pred[0].tolist()]

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn[h].cpu().detach().numpy())
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90)
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens)
        ax.set_title(f"Head {h}")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


# --------------------------- Main ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tiny Transformer for Copy/Reverse tasks")
    parser.add_argument("--task", choices=["copy", "reverse"], default="copy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--vocab_size", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--visualise", action="store_true", help="Visualise attention after training")
    args = parser.parse_args()

    model = TinyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_model * 2,
        dropout=0.1,
        max_len=args.seq_len + 1,
    ).to(DEVICE)

    print("Training on", DEVICE)
    train(model, args.task, args.epochs, args.batch_size, args.seq_len, args.vocab_size, args.lr)

    # Evaluate on a single example
    src, tgt = generate_batch(1, args.seq_len, args.vocab_size, args.task)
    pred = greedy_decode(model, src, args.seq_len)

    idx2tok = [str(i) for i in range(args.vocab_size)]
    print("Source:", src[0].tolist())
    print("Target:", tgt[0].tolist())
    print("Pred  :", pred[0].cpu().tolist())

    if args.visualise:
        # Forward once more to store attention
        bos = torch.zeros_like(src[:, :1])
        decoder_in = torch.cat([bos, tgt.to(DEVICE)[:, :-1]], dim=1)
        _ = model(src.to(DEVICE), decoder_in)
        show_attention_maps(model, src, pred, idx2tok)


if __name__ == "__main__":
    main()