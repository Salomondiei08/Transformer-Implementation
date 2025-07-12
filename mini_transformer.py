import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding introduced in "Attention is All You Need".

    This module creates fixed positional encodings and adds them to token
    embeddings. It does *not* learn any additional parameters beyond the
    embedding scaling factor (which we omit for clarity).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to *x*.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class MultiHeadAttention(nn.Module):
    """A minimal Multi-Head Attention module implemented from basic PyTorch ops.

    The module also exposes attention weights (of the last forward pass) which
    allows us to visualise how the model attends over the input tokens.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Cache attention map from the most recent forward pass (for visualisation).
        self._attn_map: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention with multiple heads.

        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len) where masked positions are ``True``. Used to
                   implement subsequent masking in the decoder.

        Returns:
            output: (batch, seq_len, d_model)
            attn:   (batch, num_heads, seq_len, seq_len) attention weights
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections & split into heads
        query = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k)
        key = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k)
        value = self.w_v(value).view(
            batch_size, seq_len, self.num_heads, self.d_k
        )

        # Transpose for attention computation: (batch, num_heads, seq_len, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == True, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)  # (batch, num_heads, seq_len, d_k)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        # Save attention map for visualisation
        self._attn_map = attn.detach().cpu()
        return output, attn

    @property
    def attn_map(self) -> Optional[torch.Tensor]:
        """Return last attention map. Shape (batch, num_heads, tgt_len, src_len)."""
        return self._attn_map


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside Transformer blocks."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Masked self-attention (decoder)
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Encoder-decoder attention
        attn_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)

        # Feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        logits = self.out_proj(x)
        return logits


def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent positions (for auto-regressive decoding)."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return subsequent_mask


class TinyTransformer(nn.Module):
    """A minimal encoder-decoder Transformer for sequence-to-sequence toy tasks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 512,
        max_len: int = 128,
    ):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(tgt, memory, tgt_mask, memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, tgt_mask, src_mask)