"""
neural_network.py

Neural network written as "representation + head" (RepModel style).

Lean mapping (informal):
- rep : X → Z  (RepNet.represent)
- head : Z → Y (RepNet.head)
- model(x) = head(rep(x)) (RepNet.forward)

This design is explainability-first:
- `.represent(x)` exposes the internal representation for probes
- `.forward(x)` is the end-to-end predictor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

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
class RepNetConfig:
    input_dim: int
    rep_dim: int
    output_dim: int
    rep_hidden: Tuple[int, ...] = (256, 128)
    head_hidden: Tuple[int, ...] = ()
    task: Literal["regression", "binary"] = "regression"  # binary uses BCEWithLogitsLoss


class RepNet(nn.Module):
    def __init__(self, cfg: RepNetConfig):
        super().__init__()
        self.cfg = cfg
        self.rep = make_mlp(cfg.input_dim, cfg.rep_hidden, cfg.rep_dim)
        self.head = make_mlp(cfg.rep_dim, cfg.head_hidden, cfg.output_dim)

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        """rep : X → Z"""
        return self.rep(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """model(x) = head(rep(x))"""
        z = self.represent(x)
        return self.head(z)


def train_repnet(
    model: RepNet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
) -> None:
    """
    Minimal training loop for a rep+head network.

    - regression: MSELoss
    - binary: BCEWithLogitsLoss (expects y in {0,1} as float)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if model.cfg.task == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = x_train.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)

            # Ensure shape compatibility
            if yb.ndim == 1 and pred.ndim == 2 and pred.shape[1] == 1:
                yb = yb.view(-1, 1)

            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            total += loss.item() * xb.shape[0]

        avg = total / n

        if x_val is not None and y_val is not None and ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                xv = x_val.to(device)
                yv = y_val.to(device)
                pv = model(xv)
                if yv.ndim == 1 and pv.ndim == 2 and pv.shape[1] == 1:
                    yv = yv.view(-1, 1)
                vloss = loss_fn(pv, yv).item()
            print(f"[RepNet] epoch {ep:03d}  train_loss={avg:.6f}  val_loss={vloss:.6f}")
        elif ep % 10 == 0:
            print(f"[RepNet] epoch {ep:03d}  train_loss={avg:.6f}")


if __name__ == "__main__":
    # Quick smoke test with synthetic regression
    torch.manual_seed(0)
    n, d = 20000, 16
    x = torch.randn(n, d)

    # A synthetic "concept" c(x) you may later probe from the representation
    c = x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2]

    # Let the downstream label be a noisy function of c
    y = c + 0.1 * torch.randn_like(c)

    cfg = RepNetConfig(input_dim=d, rep_dim=8, output_dim=1, rep_hidden=(128, 64), head_hidden=(32,), task="regression")
    net = RepNet(cfg)
    train_repnet(net, x[:18000], y[:18000], x[18000:], y[18000:], epochs=30, batch_size=512, lr=1e-3)

    net.eval()
    with torch.no_grad():
        pred = net(x[18000:18010]).squeeze(-1)
    print("smoke_pred[:3]:", pred[:3].cpu().numpy())
