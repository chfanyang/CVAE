import argparse
import os
import sys
import torch
import torch.nn.functional as F
import torch.multiprocessing
import pandas as pd
from tqdm import tqdm

from utils import load_config, set_seed, get_device, ensure_dir, count_parameters
from datasets import get_dataloaders
from model import ConvVAE
from visualize import plot_loss_curve

# Avoid NFS "Device or resource busy" errors with DataLoader workers
torch.multiprocessing.set_sharing_strategy("file_system")


def loss_fn(recon_x, x, mu, logvar, beta, batch_size):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss


def train_epoch(model, loader, optimizer, device, beta, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False,
                file=sys.stdout, dynamic_ncols=True)
    for x, _ in pbar:
        x = x.to(device)
        batch_size = x.size(0)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss, recon, kl = loss_fn(recon_x, x, mu, logvar, beta, batch_size)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        pbar.set_postfix(loss=loss.item(), recon=recon.item(), kl=kl.item())
    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def evaluate(model, loader, device, beta, desc="Eval"):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    pbar = tqdm(loader, desc=desc, leave=False,
                file=sys.stdout, dynamic_ncols=True)
    for x, _ in pbar:
        x = x.to(device)
        batch_size = x.size(0)
        recon_x, mu, logvar = model(x)
        loss, recon, kl = loss_fn(recon_x, x, mu, logvar, beta, batch_size)
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device(config)
    print(f"Using device: {device}")

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    train_loader, test_loader = get_dataloaders(config)
    model = ConvVAE(z_dim=config["z_dim"]).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    beta = config["beta"]
    epochs = config["epochs"]
    best_loss = float("inf")

    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, beta, epoch, epochs
        )
        test_loss, test_recon, test_kl = evaluate(
            model, test_loader, device, beta, desc=f"Epoch {epoch}/{epochs} [Test] "
        )

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} (R: {train_recon:.4f} KL: {train_kl:.4f}) | "
            f"Test Loss: {test_loss:.4f} (R: {test_recon:.4f} KL: {test_kl:.4f})"
        )

        history.append({
            "epoch": epoch,
            "train_total_loss": train_loss,
            "train_recon_loss": train_recon,
            "train_kl_loss": train_kl,
            "test_total_loss": test_loss,
            "test_recon_loss": test_recon,
            "test_kl_loss": test_kl,
        })

        # Save last checkpoint
        ensure_dir(os.path.dirname(config["last_checkpoint_path"]))
        torch.save(model.state_dict(), config["last_checkpoint_path"])

        # Save best checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            ensure_dir(os.path.dirname(config["checkpoint_path"]))
            torch.save(model.state_dict(), config["checkpoint_path"])

    # Save loss history
    loss_csv = os.path.join(output_dir, "loss.csv")
    pd.DataFrame(history).to_csv(loss_csv, index=False)

    loss_png = os.path.join(output_dir, "loss_curve.png")
    plot_loss_curve(loss_csv, loss_png)

    print(f"Training complete. Best test loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
