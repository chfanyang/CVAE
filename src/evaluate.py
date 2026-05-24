import argparse
import os
import torch
import torch.multiprocessing

from utils import load_config, set_seed, get_device, ensure_dir, save_json
from datasets import get_dataloaders
from model import ConvVAE
from train import evaluate
from visualize import (
    save_reconstruction_grid,
    save_generation_grid,
    save_interpolation_grid,
)

# Avoid NFS "Device or resource busy" errors with DataLoader workers
torch.multiprocessing.set_sharing_strategy("file_system")


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

    _, test_loader = get_dataloaders(config)
    model = ConvVAE(z_dim=config["z_dim"]).to(device)
    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    model.eval()

    test_loss, test_recon, test_kl = evaluate(model, test_loader, device, config["beta"])

    # Compute MSE separately
    total_mse = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            recon_x, _, _ = model(x)
            total_mse += torch.nn.functional.mse_loss(recon_x, x).item()
    test_mse = total_mse / len(test_loader)

    metrics = {
        "test_total_loss": test_loss,
        "test_recon_loss": test_recon,
        "test_kl_loss": test_kl,
        "test_mse": test_mse,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_json(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")
    print(metrics)

    # Generate visualizations
    recon_path = os.path.join(output_dir, "reconstruction.png")
    save_reconstruction_grid(model, test_loader, device, recon_path)
    print(f"Reconstruction saved to {recon_path}")

    gen_path = os.path.join(output_dir, "generation.png")
    save_generation_grid(model, device, gen_path, config["z_dim"])
    print(f"Generation saved to {gen_path}")

    interp_path = os.path.join(output_dir, "interpolation.png")
    save_interpolation_grid(model, device, interp_path, config["z_dim"])
    print(f"Interpolation saved to {interp_path}")


if __name__ == "__main__":
    main()
