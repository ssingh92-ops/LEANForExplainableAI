"""
autoencoder.py

Deterministic AutoEncoder in PyTorch.

Lean mapping (informal):
- X : input space (R^d in this script)
- Z : latent space (R^k)
- enc : X → Z            (AutoEncoder.encode)
- dec : Z → X            (AutoEncoder.decode)
- recon(x) = dec(enc(x)) (AutoEncoder.forward)

This file is intentionally simple and "spec-aligned":
- exposes `.encode(x)` (representation) for explainability probes
- exposes `.decode(z)` and `.forward(x)` for reconstruction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def make_mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, *,
             activation: nn.Module = nn.ReLU(), out_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(activation)
        d = h
    layers.append(nn.Linear(d, out_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class AutoEncoderConfig:
    input_dim: int
    latent_dim: int
    enc_hidden: Tuple[int, ...] = (256, 128)
    dec_hidden: Tuple[int, ...] = (128, 256)
    # If your inputs are in [0,1] (e.g., images), set output_activation=torch.nn.Sigmoid()
    output_activation: Optional[str] = None  # "sigmoid" | None


class AutoEncoder(nn.Module):
    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        self.cfg = cfg

        out_act = None
        if cfg.output_activation == "sigmoid":
            out_act = nn.Sigmoid()

        self.encoder = make_mlp(cfg.input_dim, cfg.enc_hidden, cfg.latent_dim)
        self.decoder = make_mlp(cfg.latent_dim, cfg.dec_hidden, cfg.input_dim, out_activation=out_act)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """enc : X → Z"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """dec : Z → X"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """recon(x) = dec(enc(x))"""
        return self.decode(self.encode(x))


def train_autoencoder(
    model: AutoEncoder,
    x_train: torch.Tensor,
    x_val: Optional[torch.Tensor] = None,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
) -> None:
    """
    Minimal training loop (MSE reconstruction loss).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n = x_train.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_train[idx].to(device)

            opt.zero_grad(set_to_none=True)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()

            total += loss.item() * xb.shape[0]

        avg = total / n

        if x_val is not None and ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                xb = x_val.to(device)
                vloss = loss_fn(model(xb), xb).item()
            print(f"[AE] epoch {ep:03d}  train_mse={avg:.6f}  val_mse={vloss:.6f}")
        elif ep % 10 == 0:
            print(f"[AE] epoch {ep:03d}  train_mse={avg:.6f}")


if __name__ == "__main__":
    # Quick smoke test on synthetic data
    torch.manual_seed(0)
    n, d = 20000, 32
    x = torch.randn(n, d)

    cfg = AutoEncoderConfig(input_dim=d, latent_dim=8, enc_hidden=(128, 64), dec_hidden=(64, 128))
    ae = AutoEncoder(cfg)

    train_autoencoder(ae, x_train=x[:18000], x_val=x[18000:], epochs=30, batch_size=512, lr=1e-3)

    # Show reconstruction error on a small batch
    ae.eval()
    with torch.no_grad():
        xb = x[18000:18010]
        recon = ae(xb)
        mse = torch.mean((recon - xb) ** 2).item()
    print("smoke_test_mse:", mse)
