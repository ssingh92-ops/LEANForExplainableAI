"""
vae_probe.py

Goal of this file
-----------------
Give a very small PyTorch script that:

1. Defines a tiny VAE on 28x28 grayscale images (MNIST-like).
2. Defines some "concept scores" for each image (size, thickness, slant).
3. Fits a *linear probe* from latent vectors to these concept scores.
4. Computes:
   - R^2   : how much variance in the concept the probe explains.
   - delta : empirical sup error max_i |c_i - w(z_i)|.

These are the numerical quantities that correspond to the Lean spec:

- `w`      ↔ `Probe.w   : Z →L[ℝ] ℝ`
- `delta`  ↔ `ProbeSpec.δ`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 1. Define a tiny VAE model
# ------------------------------------------------------------

class MLPVAE(nn.Module):
    """
    A minimal Variational Autoencoder (VAE) for 28x28 grayscale images.

    * Input:  x ∈ [0,1]^{1×28×28}
    * Encoder: flatten → hidden layer → latent mean & log-variance
    * Decoder: latent → hidden layer → reconstruction of the image

    We only return:
      - recon: reconstructed image
      - mu   : latent mean vector
    The variance is used during training but we omit details here.
    """

    def __init__(self, d: int = 8):
        """
        Args:
            d: dimension of latent space Z (e.g. 8)
        """
        super().__init__()

        self.d = d  # store latent dimension for later use

        # Encoder network: flatten 1×28×28 image into 784 and map to 256
        self.enc = nn.Sequential(
            nn.Flatten(),             # shape: (N, 1, 28, 28) -> (N, 784)
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
        )

        # From the 256-dimensional hidden vector, we produce:
        # - mu     : latent mean (dimension d)
        # - logvar : log-variance (dimension d)
        self.fc_mu = nn.Linear(256, d)
        self.fc_logvar = nn.Linear(256, d)

        # Decoder network: latent vector of dimension d back to image
        self.dec = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),             # output in [0,1]
        )

    def encode(self, x):
        """
        Encoder step.

        Args:
            x: tensor of shape (N, 1, 28, 28), with values in [0,1].

        Returns:
            mu     : (N, d) latent means
            logvar : (N, d) latent log-variances
        """
        h = self.enc(x)          # shape (N, 256)
        mu = self.fc_mu(h)       # (N, d)
        logvar = self.fc_logvar(h)  # (N, d)
        return mu, logvar

    def reparam(self, mu, logvar):
        """
        Reparameterization trick: sample z = mu + eps * sigma.

        This lets us backpropagate through a sample from N(mu, sigma^2).

        Args:
            mu     : (N, d)
            logvar : (N, d) (log-variance)

        Returns:
            z : (N, d) sampled latent vector
        """
        eps = torch.randn_like(mu)                  # N(0, I)
        sigma = torch.exp(0.5 * logvar)             # sqrt(var)
        z = mu + sigma * eps
        return z

    def decode(self, z):
        """
        Decoder step.

        Args:
            z: latent tensor of shape (N, d)

        Returns:
            recon: tensor of shape (N, 1, 28, 28) with values in [0,1]
        """
        x_flat = self.dec(z)                        # (N, 784)
        x_recon = x_flat.view(-1, 1, 28, 28)        # back to image shape
        return x_recon

    def forward(self, x, beta: float = 4.0):
        """
        Full VAE forward pass.

        Args:
            x   : (N, 1, 28, 28) input images
            beta: weight on the KL-divergence term (β-VAE).

        Returns:
            recon : reconstruction of x
            mu    : latent means
            loss  : scalar VAE loss (recon + β * KL)

        Note: In a real training loop you would average `loss` and call
        `loss.backward()` + optimizer step. Here we just compute it.
        """
        # 1) Encode
        mu, logvar = self.encode(x)

        # 2) Sample latent z
        z = self.reparam(mu, logvar)

        # 3) Decode to reconstruction
        recon = self.decode(z)

        # 4) Compute reconstruction loss and KL loss.
        #    - Flatten images to (N, 784)
        #    - Use binary cross-entropy as a typical VAE recon loss.
        recon_loss = F.binary_cross_entropy(
            recon.view_as(x), x, reduction="sum"
        )

        # KL divergence between N(mu, sigma^2) and N(0, I)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + beta * kl

        # In the context of the Lean spec:
        # - mu   : latent point in Z
        # - recon: concrete recon(x)
        return recon, mu, loss


# ------------------------------------------------------------
# 2. Concept scores: cheap "human concepts" on images
# ------------------------------------------------------------

def concept_scores(x):
    """
    Compute simple "concept scores" for each image.

    Args:
        x: tensor of shape (N, 1, 28, 28), values in [0,1].

    Returns:
        c: tensor of shape (N, 3) with:
            c[:,0] = "size"    = number of foreground pixels
            c[:,1] = "thick"   = rough thickness (via dilation heuristic)
            c[:,2] = "slant"   = crude slant angle proxy (moment-based)

    These are meant as stand-ins for human labels such as:
        - "how big is the digit?"
        - "how thick is the stroke?"
        - "is it leaning left/right?"
    """

    with torch.no_grad():
        # Convert probabilities in [0,1] to a binary-ish mask
        # threshold at 0.5
        b = (x > 0.5).float()      # shape: (N, 1, 28, 28)

        # --- size: number of foreground pixels ---
        # sum over spatial + channel dims => (N,)
        size = b.sum(dim=(1, 2, 3))  # pixel count per image

        # --- thickness: (dilate - original) heuristic ---
        # Define a simple 3x3 kernel of ones; use conv2d as dilation.
        k = torch.ones(1, 1, 3, 3, device=x.device)
        # Use convolution to "dilate" the binary image.
        # If any pixel in the 3x3 neighborhood is 1, the output > 0.
        dil = F.conv2d(b, k, padding=1)
        dil = (dil > 0).float()

        # Thickness proxy = number of pixels gained by dilation
        thick = (dil - b).sum(dim=(1, 2, 3))

        # --- slant: approximate using image moments ---
        # Create coordinate grid for 28x28
        xs = torch.arange(28, device=x.device).view(1, 1, 28, 1)  # (1,1,28,1)
        ys = torch.arange(28, device=x.device).view(1, 1, 1, 28)  # (1,1,1,28)

        # Compute 0th moment (sum of intensities)
        m00 = b.sum(dim=(1, 2, 3)) + 1e-6  # avoid division by zero

        # Compute first moments (center of mass)
        mx = (b * xs).sum(dim=(1, 2, 3)) / m00
        my = (b * ys).sum(dim=(1, 2, 3)) / m00

        # Shift coordinates by center of mass
        xs_centered = xs - mx.view(-1, 1, 1, 1)
        ys_centered = ys - my.view(-1, 1, 1, 1)

        # Second moments
        uxx = (b * xs_centered ** 2).sum(dim=(1, 2, 3)) / m00
        uyy = (b * ys_centered ** 2).sum(dim=(1, 2, 3)) / m00
        uxy = (b * xs_centered * ys_centered).sum(dim=(1, 2, 3)) / m00

        # Slant proxy: angle derived from the second-moment matrix.
        # This is not rigorous, but serves as a continuous feature.
        slant = 0.5 * torch.atan2(2 * uxy, (uxx - uyy) + 1e-6)

        # Stack all three concepts into a single tensor of shape (N, 3)
        c = torch.stack([size, thick, slant], dim=1)

    return c


# ------------------------------------------------------------
# 3. Fit a linear probe from latent μ to concept scores
# ------------------------------------------------------------

def fit_probe(mu, C, lam=1e-3):
    """
    Fit a linear probe w : Z -> R^k using ridge regression.

    Args:
        mu : tensor of shape (N, d)         (latent means)
        C  : tensor of shape (N, k)         (concept scores per image)
        lam: ridge regularization strength (λ >= 0)

    Returns:
        W : tensor of shape (d+1, k)
            - the first d rows are weights for each latent dimension.
            - the last  row is the bias term.

    We solve the closed-form ridge regression:

        W = (Z^T Z + λ I)^{-1} Z^T C

    where Z is the design matrix with a column of ones for bias.
    """

    # Number of examples N and latent dimension d
    N, d = mu.shape
    k = C.shape[1]

    # Construct design matrix Z with an extra bias column:
    # shape (N, d+1).  Last column is all ones.
    ones = torch.ones(N, 1, device=mu.device)
    Z = torch.cat([mu, ones], dim=1)

    # Form the normal equations for ridge regression:
    #   (Z^T Z + λ I) W = Z^T C
    # Z^T Z has shape (d+1, d+1)
    ZtZ = Z.T @ Z

    # Add λ I for regularization (on weights and bias).
    lamI = lam * torch.eye(d + 1, device=mu.device)
    A = ZtZ + lamI               # (d+1, d+1)

    # Right-hand side: Z^T C   (shape (d+1, k))
    B = Z.T @ C

    # Solve the linear system A W = B
    W = torch.linalg.solve(A, B)  # shape (d+1, k)

    return W


def probe_metrics(mu, C, W):
    """
    Compute R^2 and sup error for a fitted probe.

    Args:
        mu : (N, d) latent means
        C  : (N, k) true concept scores
        W  : (d+1, k) weights from `fit_probe`

    Returns:
        R2     : tensor of shape (k,)
                 R^2 score for each concept dimension.
        delta  : tensor of shape (k,)
                 empirical sup error: max_i |c_i - w(z_i)|.
    """

    N, d = mu.shape
    k = C.shape[1]

    # Build design matrix Z = [mu, 1] (N, d+1)
    ones = torch.ones(N, 1, device=mu.device)
    Z = torch.cat([mu, ones], dim=1)

    # Predicted concept scores
    C_pred = Z @ W  # (N, k)

    # Residual sum of squares and total sum of squares
    residual = C - C_pred
    ssr = (residual ** 2).sum(dim=0)               # (k,)
    sst = ((C - C.mean(dim=0)) ** 2).sum(dim=0)    # (k,)

    # R^2 = 1 - SSR/SST  (for each concept dim)
    R2 = 1.0 - ssr / (sst + 1e-8)

    # sup error δ = max_i |c_i - w(z_i)|
    delta = residual.abs().max(dim=0).values       # (k,)

    return R2, delta


# ------------------------------------------------------------
# 4. Minimal test harness
# ------------------------------------------------------------

def main():
    """
    Minimal "shape test" for the whole pipeline.

    Instead of using real MNIST data, we:
      * create N random images,
      * run them through the VAE,
      * compute concept scores,
      * fit a probe,
      * compute R^2 and delta.

    This tests that:
      - the VAE code runs,
      - concept_scores works,
      - linear algebra in fit_probe / probe_metrics is correct.

    Later you can replace the random data with real images.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Fake batch of images: N examples of shape (1, 28, 28)
    N = 128
    x = torch.rand(N, 1, 28, 28, device=device)

    # 2) Build VAE and move to device
    vae = MLPVAE(d=8).to(device)

    # 3) Forward pass through VAE
    recon, mu, loss = vae(x)
    print("recon shape:", recon.shape)  # (N, 1, 28, 28)
    print("mu shape    :", mu.shape)    # (N, d)
    print("loss        :", loss.item())

    # 4) Concept scores on the original images (not recon)
    C = concept_scores(x)
    print("concept scores shape:", C.shape)  # (N, 3)

    # 5) Fit linear probe
    W = fit_probe(mu, C)
    print("W shape:", W.shape)  # (d+1, 3)

    # 6) Compute R^2 and sup error δ
    R2, delta = probe_metrics(mu, C, W)
    print("R^2 per concept :", R2)
    print("delta per concept:", delta)

    # These numbers (W, delta) are what you conceptually export to Lean:
    #  - `W` is the data for a linear map w : Z → ℝ^3
    #  - each column of W is one Probe.w for a particular concept
    #  - `delta` is the empirical sup error bound to plug into ProbeSpec.δ


if __name__ == "__main__":
    main()
