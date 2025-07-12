# 🔹 Mini Transformer from Scratch

A complete implementation of a small Transformer model in PyTorch for educational purposes. This project demonstrates all core Transformer components and trains on simple sequence-to-sequence tasks like copying and reversing sequences.

## 🎯 Project Goals

- **Educational**: Understand Transformer architecture from ground up
- **Implementation**: Build Multi-Head Attention, Positional Encoding, LayerNorm, etc.
- **Visualization**: See what the model learns through attention maps
- **Practical**: Train on toy tasks to validate the implementation

## 🏗️ Architecture Components

### Core Components Implemented

1. **Multi-Head Attention** (`transformer.py`)
   - Scaled dot-product attention
   - Multiple attention heads
   - Linear projections for Q, K, V
   - Attention weight storage for visualization

2. **Positional Encoding** (`transformer.py`)
   - Sinusoidal position embeddings
   - Learnable position information
   - Supports sequences up to specified max length

3. **Transformer Block** (`transformer.py`)
   - Self-attention layer
   - Feed-forward network
   - Residual connections
   - Layer normalization

4. **Complete Model** (`transformer.py`)
   - Token embeddings
   - Positional encoding
   - Stack of Transformer blocks
   - Output projection layer

## 📊 Tasks Implemented

### 1. Copy Task
- **Input**: `[START, 5, 7, 9, END]`
- **Target**: `[START, 5, 7, 9, END]`
- **Goal**: Learn to copy the input sequence exactly

### 2. Reverse Task
- **Input**: `[START, 5, 7, 9, END]`
- **Target**: `[START, 9, 7, 5, END]`
- **Goal**: Learn to reverse the input sequence

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Implementation

```bash
# Test the transformer components
python transformer.py

# Test data generation
python data.py
```

### 3. Train the Models

```bash
# Train on both copy and reverse tasks
python train.py
```

This will:
- Train a model on the copy task for 50 epochs
- Train a model on the reverse task for 50 epochs
- Save models and training plots
- Display sample predictions during training

### 4. Visualize Attention Maps

```bash
# Visualize attention patterns (after training)
python visualize.py
```

This will:
- Load trained models
- Generate attention heatmaps
- Compare copy vs reverse attention patterns
- Save visualization plots

## 📁 Project Structure

```
├── transformer.py      # Core Transformer implementation
├── data.py            # Dataset generation and loading
├── train.py           # Training loop and evaluation
├── visualize.py       # Attention visualization
├── requirements.txt   # Dependencies
├── README.md         # This file
└── models/           # Saved models (created during training)
    ├── copy_task/
    └── reverse_task/
```

## 🔍 Model Architecture Details

### Hyperparameters (Default)

```python
vocab_size = 20        # Size of vocabulary
d_model = 128         # Model dimension
num_heads = 8         # Number of attention heads
num_layers = 4        # Number of transformer blocks
d_ff = 512           # Feed-forward dimension
seq_len = 8          # Maximum sequence length
batch_size = 32      # Batch size
learning_rate = 1e-3 # Learning rate
```

### Model Size
- **Parameters**: ~400K trainable parameters
- **Memory**: Fits comfortably on CPU or GPU
- **Training Time**: ~5-10 minutes on CPU for both tasks

## 📈 Expected Results

### Copy Task
- **Accuracy**: Should reach 95%+ within 20-30 epochs
- **Pattern**: Model learns identity mapping
- **Attention**: Diagonal attention patterns (token attends to itself)

### Reverse Task
- **Accuracy**: Should reach 85%+ within 30-40 epochs  
- **Pattern**: Model learns sequence reversal
- **Attention**: Anti-diagonal patterns (first token attends to last)

## 🎨 Attention Visualizations

The project includes comprehensive attention visualization:

1. **Single Head Attention**: Heatmap for specific layer/head
2. **All Heads in Layer**: Compare all attention heads
3. **Across Layers**: See how attention evolves
4. **Task Comparison**: Copy vs Reverse attention patterns

### Sample Attention Patterns

**Copy Task**: 
- Tends to show diagonal attention (each position attends to itself)
- Earlier layers may show more distributed attention
- Later layers focus on exact position matching

**Reverse Task**:
- Shows anti-diagonal patterns (position i attends to position n-i)
- More complex attention patterns in middle layers
- Clear reversal strategy in final layers

## 🧪 Experiments to Try

### 1. Modify Architecture
```python
# Try different model sizes
model = MiniTransformer(
    vocab_size=20,
    d_model=64,      # Smaller model
    num_heads=4,     # Fewer heads
    num_layers=2     # Fewer layers
)
```

### 2. New Tasks
- **Sort Task**: Learn to sort sequences
- **Add Task**: Learn arithmetic on sequences
- **Pattern Completion**: Fill in missing tokens

### 3. Analysis
- Compare attention patterns across different sequence lengths
- Study how different heads specialize
- Analyze layer-wise attention evolution

## 📚 Educational Value

This implementation helps understand:

1. **Attention Mechanism**: How tokens interact through attention
2. **Positional Encoding**: How models understand sequence order
3. **Layer Structure**: How information flows through layers
4. **Training Dynamics**: How loss and accuracy evolve
5. **Visualization**: What the model actually learns

## 🔧 Customization

### Adding New Tasks

```python
# In data.py, add new task logic
if self.task == 'sort':
    target_seq = [self.start_token] + sorted(seq) + [self.end_token]
elif self.task == 'double':
    target_seq = [self.start_token] + seq + seq + [self.end_token]
```

### Modifying Architecture

```python
# In transformer.py, experiment with different components
class CustomTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Add your modifications here
        # e.g., different normalization, activation functions
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size or model dimensions
2. **Slow Training**: Reduce num_layers or use GPU if available
3. **Poor Performance**: Check learning rate, increase epochs
4. **Visualization Errors**: Ensure models are trained first

### Debug Mode

```python
# Add debug prints in transformer.py
print(f"Attention shape: {attention_output.shape}")
print(f"Attention weights: {self.attention_weights.shape}")
```

## 📖 References

This implementation is inspired by:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "The Illustrated Transformer" (Jay Alammar)
- PyTorch Transformer tutorials

## 🤝 Contributing

Feel free to:
- Add new tasks or datasets
- Improve visualization capabilities
- Optimize the implementation
- Add more detailed analysis tools

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning about Transformers!
