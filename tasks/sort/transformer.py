import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models.
    Adds position information to input embeddings.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention Is All You Need".
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor [batch_size, num_heads, seq_len, d_k]
            K: Key tensor [batch_size, num_heads, seq_len, d_k]
            V: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Optional mask tensor
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: Input tensors of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        """
        batch_size = query.size(0)

        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1,
                                 self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1,
                               self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1,
                                 self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask)

        # Reshape and apply final linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(attention_output)

        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection
        attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention with residual connection
        attn_output, cross_attn_weights = self.cross_attention(
            x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_src_mask(self, src):
        """Generate padding mask for source sequence."""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def generate_tgt_mask(self, tgt):
        """Generate padding and causal mask for target sequence."""
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)
                              ).unsqueeze(0).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)

        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask & tgt_pad_mask

        return tgt_mask

    def encode(self, src, src_mask):
        """Encode the source sequence."""
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        src = self.dropout(src)

        enc_output = src
        attention_weights = []

        for encoder_layer in self.encoder_layers:
            enc_output, attn_weights = encoder_layer(enc_output, src_mask)
            attention_weights.append(attn_weights)

        return enc_output, attention_weights

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """Decode the target sequence."""
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)

        dec_output = tgt
        self_attention_weights = []
        cross_attention_weights = []

        for decoder_layer in self.decoder_layers:
            dec_output, self_attn, cross_attn = decoder_layer(
                dec_output, enc_output, src_mask, tgt_mask
            )
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)

        return dec_output, self_attention_weights, cross_attention_weights

    def forward(self, src, tgt):
        """
        Forward pass of the Transformer.

        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
        """
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)

        # Encode
        enc_output, enc_attention_weights = self.encode(src, src_mask)

        # Decode
        dec_output, self_attn_weights, cross_attn_weights = self.decode(
            tgt, enc_output, src_mask, tgt_mask
        )

        # Project to vocabulary
        output = self.output_projection(dec_output)

        return output, enc_attention_weights, self_attn_weights, cross_attn_weights


class SimpleTransformer(nn.Module):
    """
    Simplified Transformer for encoder-only tasks (like sequence classification).
    """

    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3,
                 d_ff=512, max_seq_length=50, dropout=0.1):
        super(SimpleTransformer, self).__init__()

        self.d_model = d_model

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, x):
        """Generate padding mask."""
        return (x != 0).unsqueeze(1).unsqueeze(2)

    def forward(self, x):
        """
        Forward pass for encoder-only tasks.

        Args:
            x: Input sequence [batch_size, seq_len]
        """
        mask = self.generate_mask(x)

        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        # Transpose to [seq_len, batch_size, d_model] for positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, d_model]
        x = self.dropout(x)

        # Pass through encoder layers
        attention_weights = []
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, mask)
            attention_weights.append(attn_weights)

        # Project to vocabulary
        output = self.output_projection(x)

        return output, attention_weights
