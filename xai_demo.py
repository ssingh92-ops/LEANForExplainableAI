# xai_demo.py
"""
xai_demo.py

Self-contained demo that aligns with your Lean specs:

- trains a deterministic AutoEncoder (AE) to reconstruct inputs
- trains a RepNet (representation + head) to predict a downstream label
- fits linear probes on the learned representations to predict a "concept" c(x)
- reports empirical metrics, including delta_max := max_i |c_i - w(rep(x_i))|

No external datasets required (synthetic data).

Requires:
  pip install torch numpy
and these local files:
  autoencoder.py
  neural_network.py
  xai_probe_utils.py
"""

from __future__ import annotations

import numpy as np
import torch

from autoencoder import AutoEncoder, AutoEncoderConfig, train_autoencoder
from neural_network import RepNet, RepNetConfig, train_repnet
from xai_probe_utils import fit_linear_probe_ridge, probe_metrics


def make_synthetic(
    n: int = 6000,
    input_dim: int = 32,
    true_latent_dim: int = 6,
    noise_x: float = 0.05,
    noise_y: float = 0.10,
    seed: int = 0,
):
    """
    Generative story:
      s ~ N(0, I_true_latent_dim)
      x = A s + eps_x
      concept c(x) = s0 + 0.5*s1 - 0.25*s2   (ground-truth concept tied to true latents)
      label y = c + eps_y

    This makes "linear probe on learned representations" meaningful.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(input_dim, true_latent_dim)).astype(np.float32) / np.sqrt(true_latent_dim)
    s = rng.normal(size=(n, true_latent_dim)).astype(np.float32)
    x = (s @ A.T) + noise_x * rng.normal(size=(n, input_dim)).astype(np.float32)

    c = (s[:, 0] + 0.5 * s[:, 1] - 0.25 * s[:, 2]).astype(np.float32)
    y = (c + noise_y * rng.normal(size=(n,)).astype(np.float32)).astype(np.float32)

    return x, c, y


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Data
    x, c, y = make_synthetic(n=6000, input_dim=32, true_latent_dim=6, seed=0)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    c_np = c.astype(np.float64)

    # Train/val split
    n = x.shape[0]
    n_train = int(0.85 * n)
    x_train, x_val = x_t[:n_train], x_t[n_train:]
    y_train, y_val = y_t[:n_train], y_t[n_train:]

    print("\n=== 1) Train AutoEncoder (reconstruction) ===")
    ae_cfg = AutoEncoderConfig(input_dim=32, latent_dim=8, enc_hidden=(64, 32), dec_hidden=(32, 64))
    ae = AutoEncoder(ae_cfg)
    train_autoencoder(ae, x_train=x_train, x_val=x_val, epochs=12, batch_size=512, lr=1e-3)

    ae.eval()
    with torch.no_grad():
        z_ae = ae.encode(x_val).cpu().numpy().astype(np.float64)

    # Probe concept from AE representation
    print("\n=== 2) Probe concept from AE.encode(x) ===")
    c_val = c_np[n_train:]
    # For closer alignment with Lean's continuous linear map, use fit_intercept=False
    probe_ae = fit_linear_probe_ridge(z_ae, c_val, ridge=1e-2, fit_intercept=False)
    m_ae = probe_metrics(z_ae, c_val, probe_ae)
    print("AE probe metrics:", m_ae)

    print("\n=== 3) Train RepNet (rep + head) on downstream label y ===")
    nn_cfg = RepNetConfig(input_dim=32, rep_dim=8, output_dim=1, rep_hidden=(64, 32), head_hidden=(16,), task="regression")
    net = RepNet(nn_cfg)
    train_repnet(net, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, epochs=12, batch_size=512, lr=1e-3)

    net.eval()
    with torch.no_grad():
        z_nn = net.represent(x_val).cpu().numpy().astype(np.float64)

    # Probe concept from NN representation
    print("\n=== 4) Probe concept from RepNet.represent(x) ===")
    probe_nn = fit_linear_probe_ridge(z_nn, c_val, ridge=1e-2, fit_intercept=False)
    m_nn = probe_metrics(z_nn, c_val, probe_nn)
    print("RepNet probe metrics:", m_nn)

    print("\nDone. The key Lean-aligned number is delta_max (empirical δ̂).")
    print("  δ̂_AE =", m_ae["delta_max"])
    print("  δ̂_NN =", m_nn["delta_max"])


if __name__ == "__main__":
    main()
