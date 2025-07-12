import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformer import MiniTransformer
from data import create_dataloaders, tokens_to_string
import os


class AttentionVisualizer:
    """Visualizer for transformer attention patterns."""
    
    def __init__(self, model, vocab_size):
        self.model = model
        self.vocab_size = vocab_size
        self.model.eval()
    
    def get_attention_weights(self, input_seq):
        """Extract attention weights for a given input sequence."""
        with torch.no_grad():
            # Forward pass
            _ = self.model(input_seq.unsqueeze(0), use_look_ahead_mask=True)
            
            # Get attention weights from all layers
            attention_weights = self.model.get_attention_weights()
            
            return attention_weights
    
    def plot_attention_head(self, attention_weights, input_tokens, layer_idx=0, head_idx=0, 
                           figsize=(10, 8), save_path=None):
        """Plot attention weights for a specific head."""
        if not attention_weights or layer_idx >= len(attention_weights):
            print(f"No attention weights available for layer {layer_idx}")
            return
        
        # Get attention weights for specific layer and head
        # attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
        
        # Create token labels
        token_labels = [self.token_to_string(token.item()) for token in input_tokens]
        
        # Only show non-padding tokens
        seq_len = len([t for t in token_labels if t != '<PAD>'])
        attn = attn[:seq_len, :seq_len]
        token_labels = token_labels[:seq_len]
        
        # Plot heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(attn, 
                   xticklabels=token_labels,
                   yticklabels=token_labels,
                   cmap='Blues',
                   annot=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f'Attention Patterns - Layer {layer_idx+1}, Head {head_idx+1}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_all_attention_heads(self, attention_weights, input_tokens, layer_idx=0, 
                                figsize=(16, 12), save_path=None):
        """Plot attention weights for all heads in a layer."""
        if not attention_weights or layer_idx >= len(attention_weights):
            print(f"No attention weights available for layer {layer_idx}")
            return
        
        # Get number of heads
        num_heads = attention_weights[layer_idx].shape[1]
        
        # Calculate subplot layout
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Create token labels
        token_labels = [self.token_to_string(token.item()) for token in input_tokens]
        seq_len = len([t for t in token_labels if t != '<PAD>'])
        token_labels = token_labels[:seq_len]
        
        for head_idx in range(num_heads):
            row = head_idx // cols
            col = head_idx % cols
            
            # Get attention weights for this head
            attn = attention_weights[layer_idx][0, head_idx, :seq_len, :seq_len].cpu().numpy()
            
            # Plot heatmap
            sns.heatmap(attn,
                       ax=axes[row, col],
                       xticklabels=token_labels,
                       yticklabels=token_labels,
                       cmap='Blues',
                       cbar=False,
                       square=True)
            
            axes[row, col].set_title(f'Head {head_idx+1}')
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')
        
        # Hide unused subplots
        for idx in range(num_heads, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'All Attention Heads - Layer {layer_idx+1}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_across_layers(self, attention_weights, input_tokens, head_idx=0,
                                    figsize=(16, 10), save_path=None):
        """Plot attention patterns across all layers for a specific head."""
        num_layers = len(attention_weights)
        
        # Calculate subplot layout
        cols = 4
        rows = (num_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if num_layers == 1:
            axes = np.array([axes])
        
        # Create token labels
        token_labels = [self.token_to_string(token.item()) for token in input_tokens]
        seq_len = len([t for t in token_labels if t != '<PAD>'])
        token_labels = token_labels[:seq_len]
        
        for layer_idx in range(num_layers):
            row = layer_idx // cols
            col = layer_idx % cols
            
            # Get attention weights for this layer and head
            attn = attention_weights[layer_idx][0, head_idx, :seq_len, :seq_len].cpu().numpy()
            
            # Plot heatmap
            sns.heatmap(attn,
                       ax=axes[row, col],
                       xticklabels=token_labels,
                       yticklabels=token_labels,
                       cmap='Blues',
                       cbar=False,
                       square=True)
            
            axes[row, col].set_title(f'Layer {layer_idx+1}')
            axes[row, col].set_xlabel('')
            axes[row, col].set_ylabel('')
        
        # Hide unused subplots
        for idx in range(num_layers, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'Attention Across Layers - Head {head_idx+1}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def token_to_string(self, token):
        """Convert token ID to string representation."""
        token_names = {
            0: '<PAD>',
            1: '<START>',
            2: '<END>'
        }
        
        if token in token_names:
            return token_names[token]
        else:
            return str(token)
    
    def analyze_attention_patterns(self, attention_weights, input_tokens):
        """Analyze and describe attention patterns."""
        print("Attention Pattern Analysis:")
        print("-" * 50)
        
        token_labels = [self.token_to_string(token.item()) for token in input_tokens]
        seq_len = len([t for t in token_labels if t != '<PAD>'])
        
        print(f"Sequence: {' '.join(token_labels[:seq_len])}")
        print(f"Sequence length: {seq_len}")
        print(f"Number of layers: {len(attention_weights)}")
        print()
        
        for layer_idx, layer_attn in enumerate(attention_weights):
            print(f"Layer {layer_idx + 1}:")
            
            # Get attention weights (batch_size, num_heads, seq_len, seq_len)
            attn = layer_attn[0, :, :seq_len, :seq_len].cpu().numpy()
            num_heads = attn.shape[0]
            
            for head_idx in range(num_heads):
                head_attn = attn[head_idx]
                
                # Find tokens with highest attention
                max_attention_pairs = []
                for i in range(seq_len):
                    for j in range(seq_len):
                        if i != j:  # Skip self-attention
                            max_attention_pairs.append((head_attn[i, j], i, j))
                
                # Sort by attention weight
                max_attention_pairs.sort(reverse=True)
                top_pair = max_attention_pairs[0]
                
                print(f"  Head {head_idx + 1}: Strongest attention: "
                      f"{token_labels[top_pair[1]]} → {token_labels[top_pair[2]]} "
                      f"(weight: {top_pair[0]:.3f})")
            print()
    
    def compare_tasks(self, copy_model_path, reverse_model_path, test_input, device='cpu'):
        """Compare attention patterns between copy and reverse tasks."""
        print("Comparing Attention Patterns: Copy vs Reverse")
        print("=" * 60)
        
        # Load copy model
        copy_model = MiniTransformer(vocab_size=self.vocab_size)
        copy_checkpoint = torch.load(copy_model_path, map_location=device)
        copy_model.load_state_dict(copy_checkpoint['model_state_dict'])
        copy_model.eval()
        
        # Load reverse model
        reverse_model = MiniTransformer(vocab_size=self.vocab_size)
        reverse_checkpoint = torch.load(reverse_model_path, map_location=device)
        reverse_model.load_state_dict(reverse_checkpoint['model_state_dict'])
        reverse_model.eval()
        
        # Get attention weights for both models
        with torch.no_grad():
            # Copy model
            _ = copy_model(test_input.unsqueeze(0), use_look_ahead_mask=True)
            copy_attention = copy_model.get_attention_weights()
            
            # Reverse model
            _ = reverse_model(test_input.unsqueeze(0), use_look_ahead_mask=True)
            reverse_attention = reverse_model.get_attention_weights()
        
        # Visualize both
        print("Copy Task Attention:")
        copy_visualizer = AttentionVisualizer(copy_model, self.vocab_size)
        copy_visualizer.analyze_attention_patterns(copy_attention, test_input)
        
        print("Reverse Task Attention:")
        reverse_visualizer = AttentionVisualizer(reverse_model, self.vocab_size)
        reverse_visualizer.analyze_attention_patterns(reverse_attention, test_input)
        
        return copy_attention, reverse_attention


def visualize_sample_predictions(model_path, task='copy', vocab_size=20, device='cpu'):
    """Visualize attention for sample predictions."""
    # Load model
    model = MiniTransformer(vocab_size=vocab_size)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test data
    _, test_loader = create_dataloaders(
        task=task,
        num_samples=100,
        seq_len=6,
        vocab_size=vocab_size,
        batch_size=4
    )
    
    # Get a sample
    for batch in test_loader:
        sample_input = batch['input'][0]  # First sample
        sample_target = batch['target'][0]
        break
    
    print(f"Visualizing {task.upper()} task:")
    print(f"Input:  {tokens_to_string(sample_input, vocab_size)}")
    print(f"Target: {tokens_to_string(sample_target, vocab_size)}")
    print()
    
    # Create visualizer
    visualizer = AttentionVisualizer(model, vocab_size)
    
    # Get attention weights
    attention_weights = visualizer.get_attention_weights(sample_input)
    
    if attention_weights:
        # Analyze patterns
        visualizer.analyze_attention_patterns(attention_weights, sample_input)
        
        # Plot attention for first layer, first head
        visualizer.plot_attention_head(attention_weights, sample_input, 
                                     layer_idx=0, head_idx=0,
                                     save_path=f'attention_{task}_layer1_head1.png')
        
        # Plot all heads in first layer
        visualizer.plot_all_attention_heads(attention_weights, sample_input,
                                          layer_idx=0,
                                          save_path=f'attention_{task}_all_heads_layer1.png')
        
        # Plot across all layers for first head
        visualizer.plot_attention_across_layers(attention_weights, sample_input,
                                               head_idx=0,
                                               save_path=f'attention_{task}_across_layers_head1.png')
    else:
        print("No attention weights available. Make sure the model stores attention weights.")


def main():
    """Main visualization function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 20
    
    # Check if models exist
    copy_model_path = 'models/copy_task/best_model.pt'
    reverse_model_path = 'models/reverse_task/best_model.pt'
    
    if os.path.exists(copy_model_path):
        print("Visualizing Copy Task Attention:")
        print("=" * 50)
        visualize_sample_predictions(copy_model_path, task='copy', 
                                   vocab_size=vocab_size, device=device)
        print("\n")
    else:
        print(f"Copy model not found at {copy_model_path}")
        print("Please train the model first using: python train.py")
    
    if os.path.exists(reverse_model_path):
        print("Visualizing Reverse Task Attention:")
        print("=" * 50)
        visualize_sample_predictions(reverse_model_path, task='reverse',
                                   vocab_size=vocab_size, device=device)
        print("\n")
    else:
        print(f"Reverse model not found at {reverse_model_path}")
        print("Please train the model first using: python train.py")
    
    # Compare tasks if both models exist
    if os.path.exists(copy_model_path) and os.path.exists(reverse_model_path):
        print("Comparing Copy vs Reverse Attention Patterns:")
        print("=" * 60)
        
        # Create a test sequence
        test_input = torch.tensor([1, 5, 7, 9, 2])  # START, 5, 7, 9, END
        
        dummy_model = MiniTransformer(vocab_size=vocab_size)
        visualizer = AttentionVisualizer(dummy_model, vocab_size)
        
        copy_attn, reverse_attn = visualizer.compare_tasks(
            copy_model_path, reverse_model_path, test_input, device
        )


if __name__ == "__main__":
    main()