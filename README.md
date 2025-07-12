# Mini Transformer

This project implements a tiny Transformer model from scratch using PyTorch, intended for educational purposes. The model is trained on simple toy tasks (sequence copy or reverse) to demonstrate the mechanics of attention, positional encoding, and other core Transformer concepts.

## Setup
```bash
pip install -r requirements.txt
```

## Training Examples
Copy task:
```bash
python mini_transformer.py --task copy --epochs 200
```

Reverse task with attention visualisation:
```bash
python mini_transformer.py --task reverse --visualise
```

## Files
* `mini_transformer.py` – single-file implementation of the model, training loop, and optional attention visualisation.
* `requirements.txt` – dependencies.

## What You’ll Learn
* How Multi-Head Attention works under the hood.
* Sinusoidal positional encodings.
* Layer normalisation, residual connections, and feed-forward sublayers.
* How to visualise encoder-decoder attention heads.

Enjoy experimenting! ✨
