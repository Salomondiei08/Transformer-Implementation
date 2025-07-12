import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for visualization
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention."""
        # Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights  # Store for visualization
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding using sinusoidal functions."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MiniTransformer(nn.Module):
    """Mini Transformer model for sequence-to-sequence tasks."""
    
    def __init__(self, vocab_size, d_model=128, num_heads=8, num_layers=4, 
                 d_ff=512, max_len=100, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, seq, pad_token=0):
        """Create mask to ignore padding tokens."""
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, seq_len):
        """Create mask to prevent looking at future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0
    
    def forward(self, src, tgt=None, use_look_ahead_mask=False):
        """
        Forward pass.
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len) - for training
            use_look_ahead_mask: Whether to apply causal mask
        """
        batch_size, seq_len = src.size()
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create masks
        padding_mask = self.create_padding_mask(src)
        mask = padding_mask
        
        if use_look_ahead_mask:
            look_ahead_mask = self.create_look_ahead_mask(seq_len).to(src.device)
            mask = padding_mask & look_ahead_mask.unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def get_attention_weights(self):
        """Get attention weights from all layers for visualization."""
        attention_weights = []
        for block in self.transformer_blocks:
            if hasattr(block.attention, 'attention_weights') and block.attention.attention_weights is not None:
                attention_weights.append(block.attention.attention_weights)
        return attention_weights


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    vocab_size = 20
    model = MiniTransformer(vocab_size=vocab_size, d_model=128, num_heads=8, num_layers=4)
    
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {src.shape}")
    
    output = model(src)
    print(f"Output shape: {output.shape}")
    
    # Test with look-ahead mask (for autoregressive generation)
    output_masked = model(src, use_look_ahead_mask=True)
    print(f"Output with causal mask shape: {output_masked.shape}")