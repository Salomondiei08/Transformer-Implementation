import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from mini_transformer import TinyTransformer, subsequent_mask
from tasks import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny Transformer on toy tasks.")
    parser.add_argument("--task", choices=["copy", "reverse"], default="copy")
    parser.add_argument("--vocab_size", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
    """Generate encoder and decoder attention masks.

    Returns:
        src_mask: (batch, 1, 1, src_len)
        tgt_mask: (batch, 1, tgt_len, tgt_len)
    """
    # Mask padding tokens in the source sequence.
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,src_len)

    batch_size, tgt_len = tgt.size()
    tgt_pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,tgt_len)
    subsequent = subsequent_mask(tgt_len).to(tgt.device)  # (1,tgt_len,tgt_len)

    # Combine padding and subsequent masks.
    tgt_mask = tgt_pad_mask | subsequent  # broadcast to (batch,1,tgt_len,tgt_len)
    return src_mask, tgt_mask


def train_one_epoch(model, dataloader: DataLoader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt  # teacher forcing (already masked later)

        src_mask, tgt_mask = create_masks(src, tgt_input)

        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        # reshape for CE: (batch*seq, vocab)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader: DataLoader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = create_masks(src, tgt)
            logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_loader = get_dataloader(
        task=args.task,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
    val_loader = get_dataloader(
        task=args.task,
        batch_size=args.batch_size,
        num_samples=max(1000, args.batch_size * 10),
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )

    model = TinyTransformer(vocab_size=args.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")

    # Save model
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/tiny_transformer_{args.task}.pt")
    print("Model saved.")

    # Show a qualitative example
    src, tgt = next(iter(val_loader))
    src, tgt = src[:1].to(device), tgt[:1].to(device)
    src_mask, tgt_mask = create_masks(src, tgt)
    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    pred = logits.argmax(dim=-1)
    print("Src:", src.cpu().squeeze().tolist())
    print("Tgt:", tgt.cpu().squeeze().tolist())
    print("Pred:", pred.cpu().squeeze().tolist())


if __name__ == "__main__":
    main()