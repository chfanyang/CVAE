import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
import os


def _ensure_dir(path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def save_image_grid(images, path, nrow=8, title=None):
    _ensure_dir(path)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_np)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_grid(model, dataloader, device, path, num_images=8):
    _ensure_dir(path)
    model.eval()
    x, _ = next(iter(dataloader))
    x = x[:num_images].to(device)
    with torch.no_grad():
        recon_x, _, _ = model(x)
    combined = torch.cat([x, recon_x], dim=0)
    grid = torchvision.utils.make_grid(combined, nrow=num_images, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(grid_np)
    ax.axis("off")
    ax.set_title("Top: Original  /  Bottom: Reconstructed")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_generation_grid(model, device, path, z_dim, num_images=64):
    _ensure_dir(path)
    model.eval()
    z = torch.randn(num_images, z_dim).to(device)
    with torch.no_grad():
        generated = model.decode(z)
    grid = torchvision.utils.make_grid(generated, nrow=8, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_np)
    ax.axis("off")
    ax.set_title("Generated Samples")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_interpolation_grid(model, device, path, z_dim, steps=10):
    _ensure_dir(path)
    model.eval()
    z1 = torch.randn(1, z_dim).to(device)
    z2 = torch.randn(1, z_dim).to(device)
    alpha = torch.linspace(0, 1, steps).view(-1, 1).to(device)
    z_interp = z1 * (1 - alpha) + z2 * alpha
    with torch.no_grad():
        generated = model.decode(z_interp)
    grid = torchvision.utils.make_grid(generated, nrow=steps, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(grid_np)
    ax.axis("off")
    ax.set_title("Latent Space Interpolation")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_csv_path, output_path):
    import pandas as pd
    _ensure_dir(output_path)
    df = pd.read_csv(loss_csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_total_loss"], label="Train Total")
    axes[0].plot(df["epoch"], df["test_total_loss"], label="Test Total")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["train_recon_loss"], label="Train Recon")
    axes[1].plot(df["epoch"], df["train_kl_loss"], label="Train KL")
    axes[1].plot(df["epoch"], df["test_recon_loss"], label="Test Recon")
    axes[1].plot(df["epoch"], df["test_kl_loss"], label="Test KL")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Component Losses")
    axes[1].legend()
    axes[1].grid(True)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
