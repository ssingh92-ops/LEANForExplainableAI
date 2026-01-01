/-
XAIComposition.lean

Thin entry-point for "composition/transport of explainability specs".

You use:
  - XAI.precomposeSpec (g) S
  - XAI.pullbackSpec (L) S
-/

import XAIRepresentation

namespace XAI

section

variable {X Z : Type}
variable [NormedAddCommGroup Z] [NormedSpace ℝ Z]

/-- Transport a global explainability spec along preprocessing g : X' → X. -/
abbrev precomposeSpec
    {X' : Type} (g : X' → X)
    {rep : X → Z} {p : VAE.Probe Z} {c : X → ℝ}
    (S : ProbeSpecRep (X:=X) (Z:=Z) rep p c) :
    ProbeSpecRep (X:=X') (Z:=Z) (fun x' => rep (g x')) p (fun x' => c (g x')) :=
  ProbeSpecRep.precompose (X:=X) (Z:=Z) (rep:=rep) (p:=p) (c:=c) g S

/-- Pull back a global explainability spec along a latent linear map L : Z →L Z'. -/
abbrev pullbackSpec
    {Z' : Type} [NormedAddCommGroup Z'] [NormedSpace ℝ Z']
    (L : Z →L[ℝ] Z')
    {rep : X → Z} {p' : VAE.Probe Z'} {c : X → ℝ}
    (S : ProbeSpecRep (X:=X) (Z:=Z') (fun x => L (rep x)) p' c) :
    ProbeSpecRep (X:=X) (Z:=Z) rep (pullbackProbe (X:=X) (Z:=Z) (Z':=Z') L p') c :=
  ProbeSpecRep_pullback_latent (X:=X) (Z:=Z) (Z':=Z') (rep:=rep) (p':=p') (c:=c) L S

end

end XAI
