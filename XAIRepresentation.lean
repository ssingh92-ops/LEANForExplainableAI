/-
XAIRepresentation.lean

Generalization layer:
- Move from "VAE-specific" to "representation + head" (covers AEs + NNs).
- Keep explainability specs (probes) attached to the representation map `rep : X → Z`.

This file assumes your project has flat modules:
  - VAE.lean
  - VAEDerivations.lean

So imports are:
  import VAE
  import VAEDerivations
-/

import VAE
import VAEDerivations

open scoped BigOperators

namespace XAI

/-
────────────────────────────────────────────────────────
  1) Generic "Representation Model"
────────────────────────────────────────────────────────
-/

/-- A model that factors through a representation `rep : X → Z`
    followed by a head `head : Z → Y`. -/
structure RepModel (X Z Y : Type) where
  rep  : X → Z
  head : Z → Y

namespace RepModel

variable {X Z Y : Type}

/-- The induced end-to-end function `X → Y`. -/
def model (M : RepModel X Z Y) : X → Y :=
  fun x => M.head (M.rep x)

@[simp] lemma model_def (M : RepModel X Z Y) :
    M.model = fun x => M.head (M.rep x) := rfl

end RepModel


/-
────────────────────────────────────────────────────────
  2) AutoEncoder as a special case
────────────────────────────────────────────────────────
-/

/-- An autoencoder is a RepModel with Y = X. -/
abbrev AutoEncoder (X Z : Type) : Type := RepModel X Z X

namespace AutoEncoder

variable {X Z : Type}

/-- Turn your VAE-shaped structure (enc/dec) into an AutoEncoder,
    ignoring stochasticity (so this works for deterministic AEs too). -/
def ofVAE (V : VAE.VAE X Z) : AutoEncoder X Z :=
  { rep := V.enc
    head := V.dec }

@[simp] lemma ofVAE_rep (V : VAE.VAE X Z) :
    (ofVAE (X:=X) (Z:=Z) V).rep = V.enc := rfl

@[simp] lemma ofVAE_head (V : VAE.VAE X Z) :
    (ofVAE (X:=X) (Z:=Z) V).head = V.dec := rfl

end AutoEncoder


/-
────────────────────────────────────────────────────────
  3) Probe specs that depend ONLY on the representation
     (this is the key for NN generalization)
────────────────────────────────────────────────────────
-/

section ProbeSpecs

variable {X Z : Type}
variable [NormedAddCommGroup Z] [NormedSpace ℝ Z]

/-- Global probe spec for a representation `rep : X → Z`. -/
structure ProbeSpecRep (rep : X → Z) (p : VAE.Probe Z) (c : X → ℝ) where
  δ        : ℝ
  δ_nonneg : 0 ≤ δ
  approx   : ∀ x : X, |c x - p.w (rep x)| ≤ δ

/-- On-distribution / domain-restricted probe spec. -/
structure ProbeSpecRepOn (rep : X → Z) (p : VAE.Probe Z) (c : X → ℝ) (D : X → Prop) where
  δ        : ℝ
  δ_nonneg : 0 ≤ δ
  approx   : ∀ x : X, D x → |c x - p.w (rep x)| ≤ δ

namespace ProbeSpecRep

variable {rep : X → Z} {p : VAE.Probe Z} {c : X → ℝ}

/-- Widening: if the spec holds with δ it holds with any δ' ≥ δ. -/
theorem widen (S : ProbeSpecRep (X:=X) (Z:=Z) rep p c)
    {δ' : ℝ} (h : S.δ ≤ δ') :
    ProbeSpecRep (X:=X) (Z:=Z) rep p c := by
  refine
    { δ := δ'
      δ_nonneg := le_trans S.δ_nonneg h
      approx := ?_ }
  intro x
  exact le_trans (S.approx x) h

/-- Transport along preprocessing `g : X' → X`. -/
theorem precompose
    {X' : Type} (g : X' → X)
    (S : ProbeSpecRep (X:=X) (Z:=Z) rep p c) :
    ProbeSpecRep (X:=X') (Z:=Z) (fun x' => rep (g x')) p (fun x' => c (g x')) := by
  refine
    { δ := S.δ
      δ_nonneg := S.δ_nonneg
      approx := ?_ }
  intro x'
  simpa using S.approx (g x')

end ProbeSpecRep


namespace ProbeSpecRepOn

variable {rep : X → Z} {p : VAE.Probe Z} {c : X → ℝ} {D : X → Prop}

/-- Restrict a global spec to a domain. -/
def of_global (S : ProbeSpecRep (X:=X) (Z:=Z) rep p c) (D : X → Prop) :
    ProbeSpecRepOn (X:=X) (Z:=Z) rep p c D :=
  { δ := S.δ
    δ_nonneg := S.δ_nonneg
    approx := fun x _hxD => S.approx x }

/-- Widen δ on a restricted spec. -/
theorem widen (S : ProbeSpecRepOn (X:=X) (Z:=Z) rep p c D)
    {δ' : ℝ} (h : S.δ ≤ δ') :
    ProbeSpecRepOn (X:=X) (Z:=Z) rep p c D := by
  refine
    { δ := δ'
      δ_nonneg := le_trans S.δ_nonneg h
      approx := ?_ }
  intro x hxD
  exact le_trans (S.approx x hxD) h

/-- Transport a restricted spec along preprocessing `g : X' → X`. -/
theorem precompose
    {X' : Type} (g : X' → X)
    (S : ProbeSpecRepOn (X:=X) (Z:=Z) rep p c D) :
    ProbeSpecRepOn (X:=X') (Z:=Z)
      (fun x' => rep (g x')) p (fun x' => c (g x')) (fun x' => D (g x')) := by
  refine
    { δ := S.δ
      δ_nonneg := S.δ_nonneg
      approx := ?_ }
  intro x' hx'D
  exact S.approx (g x') hx'D

end ProbeSpecRepOn


/-
────────────────────────────────────────────────────────
  4) Latent linear transport (pullback probes)
────────────────────────────────────────────────────────
-/

section LatentLinear

variable {Z' : Type}
variable [NormedAddCommGroup Z'] [NormedSpace ℝ Z']

/-- Pull a probe back along a latent linear map `L : Z →L Z'`. -/
def pullbackProbe (L : Z →L[ℝ] Z') (p' : VAE.Probe Z') : VAE.Probe Z :=
  { w := p'.w.comp L
    name := p'.name ++ "∘L" }

/-- If a concept is explained from transformed latents `L (rep x)`,
    then it is explained from original latents `rep x` using the pulled-back probe. -/
theorem ProbeSpecRep_pullback_latent
    {rep : X → Z} {p' : VAE.Probe Z'} {c : X → ℝ}
    (L : Z →L[ℝ] Z')
    (S : ProbeSpecRep (X:=X) (Z:=Z') (fun x => L (rep x)) p' c) :
    ProbeSpecRep (X:=X) (Z:=Z) rep (pullbackProbe (X:=X) (Z:=Z) (Z':=Z') L p') c := by
  refine
    { δ := S.δ
      δ_nonneg := S.δ_nonneg
      approx := ?_ }
  intro x
  -- unfold pullbackProbe; simp reduces `comp` application
  simpa [pullbackProbe] using S.approx x

end LatentLinear

end ProbeSpecs

end XAI
