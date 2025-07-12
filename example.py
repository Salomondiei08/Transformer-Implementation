#!/usr/bin/env python3
"""
Comprehensive example demonstrating the custom Transformer implementation.
This script shows how to train a mini Transformer on sequence tasks and visualize attention.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformer import SimpleTransformer, PositionalEncoding, MultiHeadAttention
from trainer import Trainer, SequenceDataset, create_vocabulary, generate_test_sequences
from visualization import (
    plot_attention_weights, plot_multi_head_attention, plot_positional_encoding,
    plot_training_history, visualize_sequence_prediction, plot_embedding_space
)
from torch.utils.data import DataLoader
import os


def demonstrate_positional_encoding():
    """Demonstrate positional encoding visualization."""
    print("🔹 Demonstrating Positional Encoding...")

    # Create positional encoding
    d_model = 64
    max_len = 20

    pe = PositionalEncoding(d_model, max_len)

    # Generate sample input
    x = torch.randn(max_len, 1, d_model)
    output = pe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Visualize positional encoding
    plot_positional_encoding(d_model=d_model, max_len=max_len,
                             save_path="positional_encoding.png")

    print("✅ Positional encoding demonstration complete!\n")


def demonstrate_multi_head_attention():
    """Demonstrate multi-head attention mechanism."""
    print("🔹 Demonstrating Multi-Head Attention...")

    # Create multi-head attention
    d_model = 64
    num_heads = 4
    seq_len = 8
    batch_size = 2

    attention = MultiHeadAttention(d_model, num_heads)

    # Generate sample input
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output, attention_weights = attention(query, key, value)

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Visualize attention weights for first batch, first head
    attn_vis = [attention_weights]  # Wrap in list for visualization function
    plot_attention_weights(attn_vis, layer_idx=0, head_idx=0,
                           title="Multi-Head Attention Demo",
                           save_path="attention_demo.png")

    print("✅ Multi-head attention demonstration complete!\n")


def train_and_analyze_model(task='copy', save_model=True):
    """Train a model on a specific task and analyze its behavior."""
    print(f"🔹 Training model for task: {task}")

    # Configuration
    vocab_size = 10
    seq_length = 8
    d_model = 64
    num_heads = 4
    num_layers = 3
    batch_size = 32
    num_epochs = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create vocabulary
    vocab = create_vocabulary(vocab_size)

    # Create datasets
    train_dataset = SequenceDataset(vocab_size, seq_length, 800, task)
    val_dataset = SequenceDataset(vocab_size, seq_length, 200, task)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_length=seq_length
    )

    # Train model
    trainer = Trainer(model, device)
    save_path = f"model_{task}.pth" if save_model else None
    train_losses, val_losses, val_accuracies = trainer.train(
        train_dataloader, val_dataloader, num_epochs, save_path=save_path
    )

    # Plot training history
    plot_training_history(train_losses, val_accuracies,
                          save_path=f"training_history_{task}.png")

    # Test model performance
    test_data = generate_test_sequences(vocab_size, seq_length, 100, task)

    model.eval()
    correct = 0
    total = 0
    all_attention_weights = []

    with torch.no_grad():
        for input_seq, target_seq in test_data:
            input_seq = input_seq.unsqueeze(0).to(device)
            target_seq = target_seq.to(device)

            output, attention_weights = model(input_seq)
            predictions = torch.argmax(output, dim=-1)

            if torch.equal(predictions[0], target_seq):
                correct += 1
            total += 1

            # Store attention weights for visualization
            if len(all_attention_weights) < 5:  # Store first 5 examples
                all_attention_weights.append(attention_weights)

    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Visualize attention patterns
    if all_attention_weights:
        print("\n🔹 Visualizing attention patterns...")

        # Plot attention for first layer, first head
        plot_attention_weights(all_attention_weights[0], layer_idx=0, head_idx=0,
                               title=f"Attention Weights - {task} task",
                               save_path=f"attention_{task}_layer0_head0.png")

        # Plot multi-head attention for first layer
        plot_multi_head_attention(all_attention_weights[0], layer_idx=0,
                                  title=f"Multi-Head Attention - {task} task",
                                  save_path=f"multihead_attention_{task}.png")

    # Show example predictions
    print("\n🔹 Example predictions:")
    for i in range(3):
        input_seq, target_seq = test_data[i]
        input_seq = input_seq.unsqueeze(0).to(device)

        with torch.no_grad():
            output, attention_weights = model(input_seq)
            predictions = torch.argmax(output, dim=-1)

        input_text = [
            vocab.get(idx.item(), f'<{idx.item()}>') for idx in input_seq[0]]
        target_text = [
            vocab.get(idx.item(), f'<{idx.item()}>') for idx in target_seq]
        pred_text = [
            vocab.get(idx.item(), f'<{idx.item()}>') for idx in predictions[0]]

        print(f"Input:  {' '.join(input_text)}")
        print(f"Target: {' '.join(target_text)}")
        print(f"Pred:   {' '.join(pred_text)}")
        print()

    return model, vocab, accuracy


def analyze_embedding_space(model, vocab):
    """Analyze the learned embedding space."""
    print("🔹 Analyzing embedding space...")

    # Get embedding weights
    embeddings = model.embedding.weight.detach()

    # Visualize embedding space
    plot_embedding_space(embeddings, vocab, save_path="embedding_space.png")

    print("✅ Embedding space analysis complete!\n")


def compare_tasks():
    """Compare model performance across different tasks."""
    print("🔹 Comparing performance across different tasks...")

    tasks = ['copy', 'reverse', 'sort', 'shift']
    results = {}

    for task in tasks:
        print(f"\n{'='*40}")
        print(f"Training for task: {task}")
        print(f"{'='*40}")

        model, vocab, accuracy = train_and_analyze_model(
            task, save_model=False)
        results[task] = accuracy

    # Plot comparison
    plt.figure(figsize=(10, 6))
    tasks_list = list(results.keys())
    accuracies = list(results.values())

    bars = plt.bar(tasks_list, accuracies, color=[
                   '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Model Performance Across Different Tasks')
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('task_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✅ Task comparison complete!")
    print("Results:")
    for task, acc in results.items():
        print(f"  {task}: {acc:.4f}")


def interactive_demo():
    """Interactive demo for testing the model."""
    print("🔹 Interactive Demo")
    print("Enter sequences to test the model (e.g., '1 2 3 4 5 6 7 8')")
    print("Type 'quit' to exit")

    # Load a trained model (assuming we have one)
    try:
        model = SimpleTransformer(
            vocab_size=10, d_model=64, num_heads=4, num_layers=3)
        model.load_state_dict(torch.load('model_copy.pth', map_location='cpu'))
        model.eval()
        vocab = create_vocabulary(10)
        print("✅ Loaded trained model")
    except:
        print("❌ No trained model found. Please train a model first.")
        return

    while True:
        user_input = input("\nEnter sequence: ").strip()

        if user_input.lower() == 'quit':
            break

        try:
            # Parse input
            tokens = user_input.split()
            input_seq = torch.tensor([int(token)
                                     for token in tokens], dtype=torch.long)

            if len(input_seq) != 8:
                print("Please enter exactly 8 numbers")
                continue

            # Make prediction
            with torch.no_grad():
                output, attention_weights = model(input_seq.unsqueeze(0))
                predictions = torch.argmax(output, dim=-1)

            # Display results
            input_text = [
                vocab.get(idx.item(), f'<{idx.item()}>') for idx in input_seq]
            pred_text = [
                vocab.get(idx.item(), f'<{idx.item()}>') for idx in predictions[0]]

            print(f"Input:  {' '.join(input_text)}")
            print(f"Output: {' '.join(pred_text)}")

        except ValueError:
            print("Please enter valid numbers separated by spaces")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run all demonstrations."""
    print("🚀 Custom Transformer from Scratch - Comprehensive Demo")
    print("=" * 60)

    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)

    # 1. Demonstrate core components
    demonstrate_positional_encoding()
    demonstrate_multi_head_attention()

    # 2. Train and analyze models for different tasks
    print("🔹 Training models for different sequence tasks...")

    # Train a model for copy task
    model, vocab, accuracy = train_and_analyze_model('copy', save_model=True)

    # Analyze embedding space
    analyze_embedding_space(model, vocab)

    # 3. Compare performance across tasks
    compare_tasks()

    # 4. Interactive demo
    print("\n" + "="*60)
    print("🎯 Interactive Demo")
    print("="*60)
    interactive_demo()

    print("\n🎉 Demo complete! Check the generated plots and saved models.")
    print("Generated files:")
    print("  - model_copy.pth (trained model)")
    print("  - *.png (visualization plots)")
    print("  - training_history_*.png (training curves)")


if __name__ == "__main__":
    main()
