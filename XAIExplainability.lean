/-
XAIExplainability.lean

Thin entry-point for "explainability = readable from representation".

This file does NOT redefine the core objects.
It just imports XAIRepresentation and provides short aliases.

You use:
  XAI.ExplainabilitySpec rep p c
  XAI.ExplainabilitySpecOn rep p c D
-/

import XAIRepresentation

namespace XAI

section

variable {X Z : Type}
variable [NormedAddCommGroup Z] [NormedSpace ℝ Z]

/-- Global explainability spec: concept is approximately readable from `rep`. -/
abbrev ExplainabilitySpec (rep : X → Z) (p : VAE.Probe Z) (c : X → ℝ) :=
  ProbeSpecRep (X:=X) (Z:=Z) rep p c

/-- On-distribution explainability spec: only required on domain predicate `D`. -/
abbrev ExplainabilitySpecOn (rep : X → Z) (p : VAE.Probe Z) (c : X → ℝ) (D : X → Prop) :=
  ProbeSpecRepOn (X:=X) (Z:=Z) rep p c D

end

end XAI
