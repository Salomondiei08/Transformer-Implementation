Tiny Transformer from Scratch
=============================

This project contains a *minimal* implementation of the Transformer architecture in PyTorch, built **from scratch** using only low-level `torch.nn` layers. It is designed for educational purposes on small, synthetic tasks such as **sequence copy** and **sequence reverse**.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train the model on the *copy* task:

```bash
python train.py --task copy --epochs 15
```

Train on the *reverse* task:

```bash
python train.py --task reverse --epochs 15
```

Training logs will print the loss per epoch and save a checkpoint to `checkpoints/`.

## Visualising Attention

After training, visualise the encoder self-attention maps:

```bash
python visualize_attention.py --checkpoint checkpoints/tiny_transformer_copy.pt --task copy
```

A heat-map for each encoder layer (head 0) will appear showing how the model attends to different token positions.

## Files Overview

* `mini_transformer.py` – implementation of positional encoding, multi-head attention, encoder/decoder layers, and the `TinyTransformer` model.
* `tasks.py` – synthetic datasets for *copy* and *reverse* tasks.
* `train.py` – training loop and utilities.
* `visualize_attention.py` – script to plot attention maps.
* `requirements.txt` – minimal dependencies (`torch`, `matplotlib`).

---

Created for the purpose of deepening understanding of Transformer internals 🌟
