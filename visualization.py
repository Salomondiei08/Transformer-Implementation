import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_attention_weights(attention_weights, layer_idx=0, head_idx=0,
                           title="Attention Weights", save_path=None):
    """
    Plot attention weights for a specific layer and head.

    Args:
        attention_weights: List of attention weight tensors from model
        layer_idx: Index of the layer to visualize
        head_idx: Index of the attention head to visualize
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    if layer_idx >= len(attention_weights):
        print(
            f"Layer index {layer_idx} out of range. Available layers: {len(attention_weights)}")
        return

    # Extract attention weights for the specified layer and head
    attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(attn.shape[1]),
                yticklabels=range(attn.shape[0]))
    plt.title(f"{title} - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multi_head_attention(attention_weights, layer_idx=0,
                              title="Multi-Head Attention", save_path=None):
    """
    Plot attention weights for all heads in a specific layer.

    Args:
        attention_weights: List of attention weight tensors from model
        layer_idx: Index of the layer to visualize
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    if layer_idx >= len(attention_weights):
        print(
            f"Layer index {layer_idx} out of range. Available layers: {len(attention_weights)}")
        return

    attn = attention_weights[layer_idx][0].detach(
    ).cpu().numpy()  # [num_heads, seq_len, seq_len]
    num_heads = attn.shape[0]

    fig, axes = plt.subplots(2, num_heads//2, figsize=(15, 8))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        sns.heatmap(attn[head_idx], annot=False,
                    cmap='Blues', ax=axes[head_idx])
        axes[head_idx].set_title(f'Head {head_idx}')
        axes[head_idx].set_xlabel('Key Position')
        axes[head_idx].set_ylabel('Query Position')

    plt.suptitle(f"{title} - Layer {layer_idx}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_positional_encoding(d_model=128, max_len=50, save_path=None):
    """
    Visualize positional encoding patterns.

    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        save_path: Optional path to save the plot
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-torch.log(torch.tensor(10000.0)) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pe.numpy(), cmap='RdBu', center=0,
                xticklabels=range(0, d_model, 10),
                yticklabels=range(0, max_len, 5))
    plt.title("Positional Encoding Visualization")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(losses, accuracies=None, save_path=None):
    """
    Plot training history (loss and accuracy).

    Args:
        losses: List of training losses
        accuracies: Optional list of accuracies
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2 if accuracies else 1, figsize=(15, 5))

    if accuracies:
        axes[0].plot(losses, label='Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(accuracies, label='Training Accuracy', color='orange')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
    else:
        axes.plot(losses, label='Training Loss')
        axes.set_title('Training Loss')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_sequence_prediction(model, input_sequence, vocab,
                                  max_length=20, temperature=1.0, save_path=None):
    """
    Visualize model predictions for a given input sequence.

    Args:
        model: Trained transformer model
        input_sequence: Input sequence as tensor
        vocab: Vocabulary mapping
        max_length: Maximum length for generation
        temperature: Temperature for sampling
        save_path: Optional path to save the plot
    """
    model.eval()
    with torch.no_grad():
        # Get attention weights during forward pass
        output, attention_weights = model(input_sequence.unsqueeze(0))

        # Convert to probabilities
        probs = torch.softmax(output / temperature, dim=-1)

        # Get predicted tokens
        predicted_tokens = torch.argmax(probs, dim=-1)

        # Convert to readable format
        input_text = [
            vocab.get(idx.item(), f'<{idx.item()}>') for idx in input_sequence]
        output_text = [
            vocab.get(idx.item(), f'<{idx.item()}>') for idx in predicted_tokens[0]]

        # Plot attention weights for the first layer
        if attention_weights:
            plot_attention_weights(attention_weights, layer_idx=0, head_idx=0,
                                   title=f"Input: {' '.join(input_text)}",
                                   save_path=save_path)

        print(f"Input: {' '.join(input_text)}")
        print(f"Output: {' '.join(output_text)}")


def plot_embedding_space(embeddings, vocab, save_path=None):
    """
    Visualize embedding space using PCA or t-SNE.

    Args:
        embeddings: Embedding weights from the model
        vocab: Vocabulary mapping
        save_path: Optional path to save the plot
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Convert embeddings to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Use PCA for dimensionality reduction
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

    # Add labels for some tokens
    for i, (token, idx) in enumerate(vocab.items()):
        if i < 20:  # Only label first 20 tokens to avoid clutter
            plt.annotate(token, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]))

    plt.title("Embedding Space Visualization (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_attention_animation(attention_weights, layer_idx=0, head_idx=0,
                               save_path="attention_animation.gif"):
    """
    Create an animated visualization of attention weights over time.

    Args:
        attention_weights: List of attention weight tensors
        layer_idx: Layer index to visualize
        head_idx: Head index to visualize
        save_path: Path to save the animation
    """
    import matplotlib.animation as animation

    if layer_idx >= len(attention_weights):
        print(f"Layer index {layer_idx} out of range")
        return

    attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    def animate(frame):
        ax.clear()
        # Show cumulative attention up to current frame
        cumulative_attn = np.sum(
            attn[:frame+1, :], axis=0) if frame > 0 else attn[0, :]
        ax.bar(range(len(cumulative_attn)), cumulative_attn)
        ax.set_title(f'Attention at Position {frame}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Attention Weight')
        ax.set_ylim(0, 1)

    anim = animation.FuncAnimation(fig, animate, frames=attn.shape[0],
                                   interval=200, repeat=True)
    anim.save(save_path, writer='pillow')
    plt.close()
