import argparse

import matplotlib.pyplot as plt
import torch

from mini_transformer import TinyTransformer
from tasks import get_dataloader
from train import create_masks


def parse_args():
    p = argparse.ArgumentParser(description="Visualize attention weights of a tiny Transformer.")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--task", choices=["copy", "reverse"], default="copy")
    p.add_argument("--vocab_size", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = TinyTransformer(vocab_size=args.vocab_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    loader = get_dataloader(
        task=args.task,
        batch_size=1,
        num_samples=1,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
    src, tgt = next(iter(loader))
    src, tgt = src.to(device), tgt.to(device)

    src_mask, tgt_mask = create_masks(src, tgt)
    with torch.no_grad():
        _ = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

    # Fetch attention map from first encoder layer, first head (example)
    attn_maps = []
    for layer in model.encoder.layers:
        if layer.self_attn.attn_map is not None:
            attn_maps.append(layer.self_attn.attn_map[0, 0].cpu())  # (seq, seq)

    if not attn_maps:
        print("No attention maps found. Did you run a forward pass?")
        return

    for i, attn in enumerate(attn_maps, 1):
        plt.figure(figsize=(4, 4))
        plt.imshow(attn, cmap="viridis")
        plt.title(f"Encoder Layer {i} Head 0")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()