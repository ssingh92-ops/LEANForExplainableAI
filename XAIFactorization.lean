/-
XAIFactorization.lean

Bridge layer:
Given a model f : X → Y, record that it factors as head ∘ rep.

This lets you attach representation-level explainability specs
(ProbeSpecRep / ProbeSpecRepOn) to *any* model f once you choose a rep/head split.

Key object:
  Factors f

Key idea:
  Factors f provides a rep : X → Z and head : Z → Y with proof:
    ∀ x, head (rep x) = f x
-/

import XAIRepresentation

namespace XAI

universe u v

/-- A factorization witness for an end-to-end model f : X → Y. -/
structure Factors {X : Type u} {Y : Type v} (f : X → Y) where
  Z    : Type u
  rep  : X → Z
  head : Z → Y
  fac  : ∀ x : X, head (rep x) = f x

namespace Factors

variable {X : Type u} {Y : Type v} {f : X → Y}

/-- Turn a factorization witness into a RepModel. -/
def toRepModel (F : Factors (X:=X) (Y:=Y) f) : RepModel X F.Z Y :=
  { rep := F.rep
    head := F.head }

/-- The RepModel induced by a factorization computes f (pointwise). -/
@[simp] lemma model_eq (F : Factors (X:=X) (Y:=Y) f) :
    (F.toRepModel.model) = f := by
  funext x
  -- unfold model, then use the stored factorization proof
  simpa [RepModel.model] using (F.fac x)

/-- Convenience: pull out the representation function directly. -/
abbrev repFn (F : Factors (X:=X) (Y:=Y) f) : X → F.Z := F.rep

/-- Convenience: explainability specs “for f” are really specs on F.rep. -/
section Explainability

variable {Z : Type}  -- not used directly; keeps section shape familiar
variable [NormedAddCommGroup (Factors.Z (X:=X) (Y:=Y) f)] [NormedSpace ℝ (Factors.Z (X:=X) (Y:=Y) f)]

/-- Global explainability spec attached to a factorization. -/
abbrev ExplainabilitySpec
    (F : Factors (X:=X) (Y:=Y) f)
    (p : VAE.Probe F.Z) (c : X → ℝ) :=
  ProbeSpecRep (X:=X) (Z:=F.Z) F.rep p c

/-- On-distribution explainability spec attached to a factorization. -/
abbrev ExplainabilitySpecOn
    (F : Factors (X:=X) (Y:=Y) f)
    (p : VAE.Probe F.Z) (c : X → ℝ) (D : X → Prop) :=
  ProbeSpecRepOn (X:=X) (Z:=F.Z) F.rep p c D

end Explainability

end Factors

end XAI
