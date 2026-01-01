/-
AutoEncoders.lean

Deterministic AutoEncoder layer.

- AutoEncoder = encoder (rep) + decoder (dec) with recon := dec (rep x).
- Provides reconstruction spec: approxId recon ε.
- Bridges from your VAE-shaped structure by forgetting stochasticity.

Flat-module imports:
  import VAE
  import VAEDerivations
  import XAIRepresentation
-/

import VAE
import VAEDerivations
import XAIRepresentation

open scoped BigOperators

namespace AE

/-- A deterministic autoencoder: representation + decoder back to X. -/
structure AutoEncoder (X Z : Type) where
  enc : X → Z
  dec : Z → X

namespace AutoEncoder

variable {X Z : Type}

/-- Reconstruction function X → X. -/
def recon (A : AutoEncoder X Z) : X → X :=
  fun x => A.dec (A.enc x)

@[simp] lemma recon_def (A : AutoEncoder X Z) :
    A.recon = fun x => A.dec (A.enc x) := rfl

/-- View an AutoEncoder as an XAI.RepModel with Y = X. -/
def toRepModel (A : AutoEncoder X Z) : XAI.RepModel X Z X :=
  { rep := A.enc
    head := A.dec }

@[simp] lemma toRepModel_rep (A : AutoEncoder X Z) :
    A.toRepModel.rep = A.enc := rfl

@[simp] lemma toRepModel_head (A : AutoEncoder X Z) :
    A.toRepModel.head = A.dec := rfl

/-- Build an AutoEncoder from your VAE-shaped structure (enc/dec). -/
def ofVAE (V : VAE.VAE X Z) : AutoEncoder X Z :=
  { enc := V.enc
    dec := V.dec }

@[simp] lemma ofVAE_enc (V : VAE.VAE X Z) : (ofVAE (X:=X) (Z:=Z) V).enc = V.enc := rfl
@[simp] lemma ofVAE_dec (V : VAE.VAE X Z) : (ofVAE (X:=X) (Z:=Z) V).dec = V.dec := rfl

end AutoEncoder


/-
────────────────────────────────────────────────────────
  Reconstruction specs (metric-space)
────────────────────────────────────────────────────────
-/

section ReconSpecs

variable {X Z : Type}
variable [PseudoMetricSpace X]

/-- Reconstruction is ε-close to identity (sup metric). -/
def ReconSpec (A : AutoEncoder X Z) (ε : ℝ) : Prop :=
  VAE.TypeConcrete.approxId (X:=X) (A.recon) ε

/-- Monotonicity in ε for ReconSpec. -/
theorem ReconSpec_mono {A : AutoEncoder X Z} {ε ε' : ℝ}
    (h : ε ≤ ε') :
    ReconSpec (X:=X) (Z:=Z) A ε → ReconSpec (X:=X) (Z:=Z) A ε' := by
  intro hε
  exact VAE.approxId_mono (X:=X) (f:=A.recon) h hε

end ReconSpecs


/-
────────────────────────────────────────────────────────
  Explainability specs for AutoEncoders
  (re-use the representation-level specs)
────────────────────────────────────────────────────────
-/

section Explainability

variable {X Z : Type}
variable [NormedAddCommGroup Z] [NormedSpace ℝ Z]

/-- AE-level probe spec is just representation probe spec on enc. -/
abbrev ProbeSpec (A : AutoEncoder X Z) (p : VAE.Probe Z) (c : X → ℝ) :=
  XAI.ProbeSpecRep (X:=X) (Z:=Z) A.enc p c

/-- On-distribution variant. -/
abbrev ProbeSpecOn (A : AutoEncoder X Z) (p : VAE.Probe Z) (c : X → ℝ) (D : X → Prop) :=
  XAI.ProbeSpecRepOn (X:=X) (Z:=Z) A.enc p c D

/-- Transport probe specs along preprocessing g : X' → X (AE). -/
theorem ProbeSpec_precompose
    {X' : Type} (g : X' → X)
    {A : AutoEncoder X Z} {p : VAE.Probe Z} {c : X → ℝ}
    (S : ProbeSpec (X:=X) (Z:=Z) A p c) :
    XAI.ProbeSpecRep (X:=X') (Z:=Z) (fun x' => A.enc (g x')) p (fun x' => c (g x')) := by
  exact XAI.ProbeSpecRep.precompose (X:=X) (Z:=Z) (rep:=A.enc) (p:=p) (c:=c) g S

end Explainability

end AE
