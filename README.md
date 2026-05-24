# Convolutional VAE for SVHN and CIFAR10

A PyTorch project for training a Convolutional Variational Autoencoder on the SVHN and CIFAR10 datasets.

## Project Structure

```
├── configs/            # YAML configuration files
│   ├── svhn.yaml
│   └── cifar10.yaml
├── src/                # Source code
│   ├── model.py        # ConvVAE model definition
│   ├── datasets.py     # Data loading utilities
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script
│   ├── visualize.py    # Visualization utilities
│   └── utils.py        # Helper functions
├── scripts/            # Shell scripts for training/evaluation
│   ├── train_svhn.sh
│   ├── train_cifar10.sh
│   ├── eval_svhn.sh
│   └── eval_cifar10.sh
├── data/               # Downloaded datasets
├── checkpoints/        # Saved model checkpoints
├── outputs/            # Outputs (loss curves, metrics, images)
│   ├── svhn/
│   └── cifar10/
└── logs/               # Training log files
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train on SVHN:
```bash
bash scripts/train_svhn.sh
```

Train on CIFAR10:
```bash
bash scripts/train_cifar10.sh
```

Or directly with Python:
```bash
python src/train.py --config configs/svhn.yaml
python src/train.py --config configs/cifar10.yaml
```

## Evaluation

Evaluate on SVHN:
```bash
bash scripts/eval_svhn.sh
```

Evaluate on CIFAR10:
```bash
bash scripts/eval_cifar10.sh
```

Or directly with Python:
```bash
python src/evaluate.py --config configs/svhn.yaml
python src/evaluate.py --config configs/cifar10.yaml
```

## Outputs

Training produces:
- `outputs/{dataset}/loss.csv` — Per-epoch loss history
- `outputs/{dataset}/loss_curve.png` — Loss curves plot
- `checkpoints/{dataset}_best.pt` — Best model checkpoint
- `checkpoints/{dataset}_last.pt` — Last epoch checkpoint

Evaluation produces:
- `outputs/{dataset}/metrics.json` — Test metrics (total loss, recon loss, KL loss, MSE)
- `outputs/{dataset}/reconstruction.png` — Original vs reconstructed images
- `outputs/{dataset}/generation.png` — Randomly generated samples
- `outputs/{dataset}/interpolation.png` — Latent space interpolation

## GPU Selection

Select a specific GPU using the `CUDA_VISIBLE_DEVICES` environment variable:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_svhn.sh
```

If CUDA is unavailable, the code automatically falls back to CPU.
