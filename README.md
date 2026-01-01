# README.md
# Lean ↔ Python: Explainable Representations (VAE / AE / Neural Nets)

This repo (as we’ve been building it) is a **spec-first XAI stack**:

- **Lean**: defines *what it means* for a representation to “explain” a concept, and how those claims compose.
- **Python**: trains models and produces **empirical witnesses** (linear probe fits + error metrics) that correspond to those Lean specs.

---

## Big idea

Most practical explainability (probes, linear separability, concept vectors) is fundamentally a statement about a **representation**:

> A concept `c(x)` is explainable from a representation `rep(x)` if there exists a (linear) probe `w`
> such that `w(rep(x))` approximates `c(x)` with bounded error.

In Lean we encode that as `ProbeSpecRep` / `ProbeSpecRepOn`.  
In Python we fit a linear probe and compute an **empirical δ̂**.

---

## Lean side (your structure)

### Kernel
- `VAE.lean`  
  Minimal categorical shell + concrete specialization + probe spec.

### Derivations / Lemmas
- `VAEDerivations.lean`  
  Monotonicity, transport, on-distribution specs, etc.

### Representation generalization
- `XAIRepresentation.lean`  
  The bridge that subsumes:
  - AutoEncoders (Y = X)
  - Neural Nets (arbitrary Y)
  - Explainability specs living on `rep : X → Z`

### Optional splits (if you keep them)
- `XAIExplainability.lean`  
  Aliases for `ProbeSpecRep` / `ProbeSpecRepOn`.
- `XAIComposition.lean`  
  Aliases for `precompose` / `pullback_latent`.
- `XAIFactorization.lean`  
  Bridge from any end-to-end `f : X → Y` to a chosen `rep/head` factorization.

---

## Python side (files)

### 1) Deterministic AutoEncoder (AE)
- `autoencoder.py`
  - `encode(x)` is the representation `rep : X → Z`
  - `decode(z)` is the decoder `dec : Z → X`
  - `forward(x)` is reconstruction `recon(x) = dec(rep(x))`

### 2) Representation + Head Neural Network
- `neural_network.py`
  - `represent(x)` is `rep : X → Z`
  - `head(z)` is `head : Z → Y`
  - `forward(x) = head(represent(x))`

### 3) Linear probe utilities
- `xai_probe_utils.py`
  - `fit_linear_probe_ridge(z, c)` fits a linear probe for concept prediction
  - `probe_metrics(...)` reports:
    - MAE, RMSE, R²
    - `delta_max` = `max_i |c_i - w(z_i)|` (**empirical sup error δ̂**)

### 4) Demo script
- `xai_demo.py`
  Trains AE + RepNet on a synthetic dataset and probes a ground-truth concept from learned reps.

---

## Run the demo

```bash
python xai_demo.py
