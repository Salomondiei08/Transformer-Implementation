# Custom Transformer from Scratch 🚀

A complete implementation of the Transformer architecture from scratch in PyTorch, designed for educational purposes and understanding the core concepts behind attention mechanisms.

## 📚 Overview

This project implements a mini Transformer model that can learn sequence-to-sequence tasks like:

- **Copy**: Replicate the input sequence
- **Reverse**: Reverse the input sequence  
- **Sort**: Sort the input sequence
- **Shift**: Shift the sequence by one position

## 🏗️ Architecture Components

### Core Building Blocks

1. **Positional Encoding** (`PositionalEncoding`)
   - Adds position information to input embeddings
   - Uses sinusoidal functions for different frequencies
   - Enables the model to understand sequence order

2. **Multi-Head Attention** (`MultiHeadAttention`)
   - Implements scaled dot-product attention
   - Multiple attention heads for different representation subspaces
   - Key innovation of the Transformer architecture

3. **Feed-Forward Network** (`FeedForward`)
   - Position-wise fully connected layers
   - Applied to each position separately and identically

4. **Layer Normalization** (`LayerNorm`)
   - Stabilizes training by normalizing activations
   - Applied after each sub-layer

### Model Variants

- **`Transformer`**: Full encoder-decoder architecture for sequence-to-sequence tasks
- **`SimpleTransformer`**: Encoder-only architecture for simpler tasks

## 🎯 Key Features

- ✅ **Complete Implementation**: All core Transformer components
- ✅ **Attention Visualization**: Visualize attention weights and patterns
- ✅ **Multiple Tasks**: Copy, reverse, sort, and shift sequence tasks
- ✅ **Training Utilities**: Comprehensive training pipeline
- ✅ **Interactive Demo**: Test the model interactively
- ✅ **Educational**: Well-documented code with explanations

## 📦 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Transformer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from transformer import SimpleTransformer
from trainer import Trainer, SequenceDataset
from torch.utils.data import DataLoader

# Create model
model = SimpleTransformer(
    vocab_size=10,
    d_model=64,
    num_heads=4,
    num_layers=3
)

# Create dataset
dataset = SequenceDataset(vocab_size=10, seq_length=8, num_samples=1000, task='copy')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
trainer = Trainer(model)
trainer.setup_optimizer(lr=0.001)
train_loss = trainer.train_epoch(dataloader)
```

### Run Complete Demo

```bash
python example.py
```

This will:

1. Demonstrate positional encoding visualization
2. Show multi-head attention in action
3. Train models on different sequence tasks
4. Visualize attention patterns
5. Compare performance across tasks
6. Provide interactive testing

## 📊 Visualization Examples

### Positional Encoding

```python
from visualization import plot_positional_encoding
plot_positional_encoding(d_model=64, max_len=20)
```

### Attention Weights

```python
from visualization import plot_attention_weights
plot_attention_weights(attention_weights, layer_idx=0, head_idx=0)
```

### Multi-Head Attention

```python
from visualization import plot_multi_head_attention
plot_multi_head_attention(attention_weights, layer_idx=0)
```

## 🔧 Model Configuration

### SimpleTransformer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 10 | Size of vocabulary |
| `d_model` | 128 | Model dimension |
| `num_heads` | 4 | Number of attention heads |
| `num_layers` | 3 | Number of encoder layers |
| `d_ff` | 512 | Feed-forward dimension |
| `max_seq_length` | 50 | Maximum sequence length |
| `dropout` | 0.1 | Dropout rate |

### Training Configuration

```python
# Recommended settings for sequence tasks
config = {
    'vocab_size': 10,
    'seq_length': 8,
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 3,
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 0.001
}
```

## 📈 Understanding the Results

### Task Performance

Different tasks have varying difficulty levels:

1. **Copy** (Easiest): Model learns to replicate input
2. **Reverse**: Model learns to reverse sequence order
3. **Sort**: Model learns to sort numbers
4. **Shift** (Hardest): Model learns to shift sequence

### Attention Patterns

- **Copy Task**: Attention focuses on corresponding positions
- **Reverse Task**: Attention shows diagonal patterns
- **Sort Task**: Attention learns to compare values
- **Shift Task**: Attention learns temporal relationships

## 🧠 Educational Insights

### Why This Implementation?

1. **Understanding Attention**: See how attention weights evolve during training
2. **Positional Encoding**: Visualize how position information is encoded
3. **Multi-Head Mechanism**: Observe different attention heads learning different patterns
4. **Training Dynamics**: Understand how Transformers learn sequence tasks

### Key Concepts Demonstrated

- **Self-Attention**: How tokens attend to other tokens in the sequence
- **Residual Connections**: How gradients flow through the network
- **Layer Normalization**: How it stabilizes training
- **Scaled Dot-Product**: The mathematical foundation of attention

## 🔍 Code Structure

```
├── transformer.py      # Core Transformer implementation
├── trainer.py          # Training utilities and data generation
├── visualization.py    # Attention and model visualization
├── example.py          # Comprehensive demonstration script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🎓 Learning Objectives

After working with this implementation, you should understand:

1. **Attention Mechanism**: How queries, keys, and values work
2. **Positional Encoding**: Why and how position information is added
3. **Multi-Head Attention**: How multiple attention heads capture different patterns
4. **Training Process**: How Transformers learn sequence tasks
5. **Visualization**: How to interpret attention weights

## 🤝 Contributing

Feel free to contribute by:

- Adding new sequence tasks
- Improving visualizations
- Optimizing the implementation
- Adding more educational examples

## 📚 Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Transformer Architecture](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) - PyTorch tutorial

## 🎉 Acknowledgments

This implementation is designed for educational purposes to help understand the Transformer architecture. It's inspired by the original "Attention Is All You Need" paper and various educational resources.

---

**Happy Learning! 🚀**
