#!/usr/bin/env python3
"""
Simple test script to verify the Transformer implementation works correctly.
"""

import torch
import torch.nn as nn
from transformer import SimpleTransformer, PositionalEncoding, MultiHeadAttention
from trainer import SequenceDataset, create_vocabulary
from torch.utils.data import DataLoader


def test_positional_encoding():
    """Test positional encoding."""
    print("Testing Positional Encoding...")

    d_model = 64
    max_len = 20
    pe = PositionalEncoding(d_model, max_len)

    x = torch.randn(max_len, 1, d_model)
    output = pe(x)

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print("✅ Positional encoding test passed!")


def test_multi_head_attention():
    """Test multi-head attention."""
    print("Testing Multi-Head Attention...")

    d_model = 64
    num_heads = 4
    seq_len = 8
    batch_size = 2

    attention = MultiHeadAttention(d_model, num_heads)

    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    output, attention_weights = attention(query, key, value)

    assert output.shape == query.shape, f"Output shape mismatch: {output.shape}"
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Attention weights shape mismatch: {attention_weights.shape}"
    print("✅ Multi-head attention test passed!")


def test_simple_transformer():
    """Test SimpleTransformer model."""
    print("Testing SimpleTransformer...")

    vocab_size = 10
    seq_length = 8
    d_model = 64
    num_heads = 4
    num_layers = 2

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_length=seq_length
    )

    # Test forward pass
    input_seq = torch.randint(1, vocab_size, (2, seq_length))
    output, attention_weights = model(input_seq)

    assert output.shape == (2, seq_length, vocab_size), \
        f"Output shape mismatch: {output.shape}"
    assert len(attention_weights) == num_layers, \
        f"Number of attention layers mismatch: {len(attention_weights)}"
    print("✅ SimpleTransformer test passed!")


def test_dataset():
    """Test dataset generation."""
    print("Testing Dataset...")

    vocab_size = 10
    seq_length = 8
    num_samples = 100

    for task in ['copy', 'reverse', 'sort', 'shift']:
        dataset = SequenceDataset(vocab_size, seq_length, num_samples, task)

        assert len(
            dataset) == num_samples, f"Dataset length mismatch: {len(dataset)}"

        input_seq, target_seq = dataset[0]
        assert input_seq.shape == (
            seq_length,), f"Input shape mismatch: {input_seq.shape}"
        assert target_seq.shape == (
            seq_length,), f"Target shape mismatch: {target_seq.shape}"

        print(f"✅ Dataset test passed for task: {task}")


def test_training_step():
    """Test a single training step."""
    print("Testing Training Step...")

    # Configuration
    vocab_size = 10
    seq_length = 8
    d_model = 64
    num_heads = 4
    num_layers = 2
    batch_size = 4

    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )

    # Create dataset and dataloader
    dataset = SequenceDataset(vocab_size, seq_length, 100, 'copy')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Single training step
    model.train()
    input_seq, target_seq = next(iter(dataloader))

    optimizer.zero_grad()
    output, _ = model(input_seq)

    # Reshape for loss calculation
    output = output.view(-1, output.size(-1))
    target = target_seq.view(-1)

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"
    print("✅ Training step test passed!")


def test_vocabulary():
    """Test vocabulary creation."""
    print("Testing Vocabulary...")

    vocab_size = 10
    vocab = create_vocabulary(vocab_size)

    assert len(vocab) == vocab_size, f"Vocabulary size mismatch: {len(vocab)}"
    assert vocab['<PAD>'] == 0, "PAD token should have index 0"
    assert vocab['1'] == 1, "Token '1' should have index 1"
    print("✅ Vocabulary test passed!")


def main():
    """Run all tests."""
    print("🧪 Running Transformer Implementation Tests")
    print("=" * 50)

    try:
        test_positional_encoding()
        test_multi_head_attention()
        test_simple_transformer()
        test_dataset()
        test_training_step()
        test_vocabulary()

        print("\n🎉 All tests passed! The implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
