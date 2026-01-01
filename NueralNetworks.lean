/-
NeuralNetworks.lean

Neural network layer as "representation + head".
Explainability specs attach to the representation.

Flat-module imports:
  import VAE
  import VAEDerivations
  import XAIRepresentation
-/

import VAE
import VAEDerivations
import XAIRepresentation

open scoped BigOperators

namespace NN

/-- A neural network is modeled as:
    rep : X → Z   (embedding / hidden state)
    head : Z → Y  (task head)

This is the *right abstraction* for explainability via probes on Z.
-/
abbrev NeuralNet (X Z Y : Type) : Type := XAI.RepModel X Z Y

namespace NeuralNet

variable {X Z Y : Type}

/-- End-to-end function X → Y. -/
abbrev eval (N : NeuralNet X Z Y) : X → Y := N.model

@[simp] lemma eval_def (N : NeuralNet X Z Y) :
    N.eval = fun x => N.head (N.rep x) := rfl

end NeuralNet


/-
────────────────────────────────────────────────────────
  Explainability specs for NNs (representation-level)
────────────────────────────────────────────────────────
-/

section Explainability

variable {X Z Y : Type}
variable [NormedAddCommGroup Z] [NormedSpace ℝ Z]

/-- Probe spec: concept is linearly decodable from NN representation. -/
abbrev ProbeSpec (N : NeuralNet X Z Y) (p : VAE.Probe Z) (c : X → ℝ) :=
  XAI.ProbeSpecRep (X:=X) (Z:=Z) N.rep p c

/-- On-distribution probe spec. -/
abbrev ProbeSpecOn (N : NeuralNet X Z Y) (p : VAE.Probe Z) (c : X → ℝ) (D : X → Prop) :=
  XAI.ProbeSpecRepOn (X:=X) (Z:=Z) N.rep p c D

/-- Transport along preprocessing g : X' → X (same rep factoring trick). -/
theorem ProbeSpec_precompose
    {X' : Type} (g : X' → X)
    {N : NeuralNet X Z Y} {p : VAE.Probe Z} {c : X → ℝ}
    (S : ProbeSpec (X:=X) (Z:=Z) (Y:=Y) N p c) :
    XAI.ProbeSpecRep (X:=X') (Z:=Z) (fun x' => N.rep (g x')) p (fun x' => c (g x')) := by
  exact XAI.ProbeSpecRep.precompose (X:=X) (Z:=Z) (rep:=N.rep) (p:=p) (c:=c) g S

/-- Pull back probes along latent linear maps L : Z →L Z' (change of representation coords). -/
theorem ProbeSpec_pullback_latent
    {Z' : Type} [NormedAddCommGroup Z'] [NormedSpace ℝ Z']
    (L : Z →L[ℝ] Z')
    {N : NeuralNet X Z Y} {p' : VAE.Probe Z'} {c : X → ℝ}
    (S : XAI.ProbeSpecRep (X:=X) (Z:=Z') (fun x => L (N.rep x)) p' c) :
    XAI.ProbeSpecRep (X:=X) (Z:=Z) N.rep (XAI.pullbackProbe (X:=X) (Z:=Z) (Z':=Z') L p') c := by
  exact XAI.ProbeSpecRep_pullback_latent (X:=X) (Z:=Z) (Z':=Z') (rep:=N.rep) (p':=p') (c:=c) L S

end Explainability


/-
────────────────────────────────────────────────────────
  Optional next: robustness/spec shells (placeholders)
  (You can fill these with Lipschitz, invariance, etc.)
────────────────────────────────────────────────────────
-/

section Robustness

variable {X Z Y : Type}

/-- Placeholder: invariance of representation under a nuisance transform τ : X → X.
    You can strengthen this later (e.g., distance-based stability).
-/
def RepInvariant (N : NeuralNet X Z Y) (τ : X → X) : Prop :=
  ∀ x, N.rep (τ x) = N.rep x

end Robustness

end NN
