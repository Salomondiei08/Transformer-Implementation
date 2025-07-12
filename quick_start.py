#!/usr/bin/env python3
"""
Quick Start Guide for the Custom Transformer Implementation
This script demonstrates the key features in a concise way.
"""

import torch
from transformer import SimpleTransformer, PositionalEncoding, MultiHeadAttention
from trainer import SequenceDataset, create_vocabulary
from visualization import plot_positional_encoding, plot_attention_weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def quick_demo():
    """Quick demonstration of the Transformer implementation."""
    print("🚀 Custom Transformer from Scratch - Quick Demo")
    print("=" * 50)

    # 1. Create a simple model
    print("\n1️⃣ Creating a mini Transformer...")
    model = SimpleTransformer(
        vocab_size=10,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_seq_length=8
    )

    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 2. Generate some data
    print("\n2️⃣ Generating training data...")
    dataset = SequenceDataset(
        vocab_size=10, seq_length=8, num_samples=100, task='copy')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    input_seq, target_seq = next(iter(dataloader))
    print(f"✅ Generated {len(dataset)} training examples")

    # 3. Test forward pass
    print("\n3️⃣ Testing forward pass...")
    model.eval()
    with torch.no_grad():
        output, attention_weights = model(input_seq)

    print(f"Input shape: {input_seq.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention layers: {len(attention_weights)}")

    # 4. Show example prediction
    print("\n4️⃣ Example prediction:")
    vocab = create_vocabulary(10)

    # Get first example
    input_tokens = [
        vocab.get(idx.item(), f'<{idx.item()}>') for idx in input_seq[0]]
    target_tokens = [
        vocab.get(idx.item(), f'<{idx.item()}>') for idx in target_seq[0]]
    pred_tokens = [vocab.get(idx.item(), f'<{idx.item()}>')
                   for idx in torch.argmax(output[0], dim=-1)]

    print(f"Input:  {' '.join(input_tokens)}")
    print(f"Target: {' '.join(target_tokens)}")
    print(f"Pred:   {' '.join(pred_tokens)}")

    # 5. Demonstrate positional encoding
    print("\n5️⃣ Positional Encoding Visualization...")
    plot_positional_encoding(d_model=64, max_len=20,
                             save_path="quick_demo_pe.png")
    print("✅ Positional encoding plot saved as 'quick_demo_pe.png'")

    # 6. Demonstrate attention visualization
    print("\n6️⃣ Attention Weights Visualization...")
    plot_attention_weights(attention_weights, layer_idx=0, head_idx=0,
                           title="Quick Demo - Attention Weights",
                           save_path="quick_demo_attention.png")
    print("✅ Attention weights plot saved as 'quick_demo_attention.png'")

    print("\n🎉 Quick demo complete!")
    print("\n📚 What you've seen:")
    print("   • Transformer model creation")
    print("   • Data generation for sequence tasks")
    print("   • Forward pass with attention weights")
    print("   • Positional encoding visualization")
    print("   • Attention weights visualization")

    print("\n🚀 Next steps:")
    print("   • Run 'python example.py' for full training demo")
    print("   • Run 'python test_transformer.py' to verify everything works")
    print("   • Check the README.md for detailed documentation")


if __name__ == "__main__":
    quick_demo()
