/-
VAE/Spec.lean

Goal of this file:
------------------
Give a *very small* category-theoretic shell for a VAE and a simple
"probe spec" that talks about explainability.

We do this in two layers:

1. Abstract categorical layer:
   - Work in an arbitrary category `C`.
   - A VAE is just two morphisms: enc : X ⟶ Z and dec : Z ⟶ X.
   - The reconstruction map is recon := dec ≫ enc : X ⟶ X.

2. Concrete layer in `Type`:
   - Now `X` and `Z` are actual types.
   - `X` is a metric space (to talk about distances between inputs).
   - `Z` is a real normed vector space (latent space).
   - We define:
       * approxId f ε : "f is ε-close to identity on X"
       * a concrete VAE with enc : X → Z, dec : Z → X
       * a linear probe w : Z →L[ℝ] ℝ
       * ProbeSpec: a predicate that says the probe explains a
         client concept `c : X → ℝ` up to error δ.

This file is *not* trying to implement training,
only the *shape* and *specs* of the system.
-/

import Mathlib        -- umbrella import that pulls in most of mathlib

open CategoryTheory   -- brings category-theory notation into scope

/-
────────────────────────────────────────────────────────
  1. Abstract categorical shell for a VAE
────────────────────────────────────────────────────────
-/

universe u v

/-- We assume we have some category `C`.

    * `C` is a *type* whose elements are "objects" of our category.
    * `[Category.{v} C]` is an *instance* that tells Lean:
        - what the morphisms X ⟶ Y are for any objects X Y : C,
        - how to compose them,
        - that composition is associative,
        - and that identity morphisms exist.

    The curly braces `{C : Type u}` mean `C` is an *implicit* argument:
    Lean usually infers it automatically.
-/
variable {C : Type u} [Category.{v} C]

/-- `VAEcat X Z` is the *categorical* interface of a VAE between objects
    `X` (data) and `Z` (latent) in the category `C`.

    We don't say anything here about probabilities, KL terms, etc.
    Just: "there is an encoder and a decoder".
-/
structure VAEcat (X Z : C) :=
  /-- Encoder: sends inputs in `X` to latents in `Z`. -/
  (enc : X ⟶ Z)
  /-- Decoder: sends latents in `Z` back to inputs in `X`. -/
  (dec : Z ⟶ X)

namespace VAEcat

/-
  We open a namespace so that we can write `VAEcat.recon` etc.

  Here we fix:
  * two objects `X Z : C`
  * one particular VAE `V : VAEcat X Z`
-/
variable {X Z : C} (V : VAEcat X Z)

/-- Reconstruction morphism `X ⟶ X`.

    In categorical notation, `≫` means composition.  So:

      V.dec ≫ V.enc : X ⟶ X

    is the morphism that first encodes `X` into `Z`, then decodes it
    back to `X`.  This mirrors `dec(enc(x))` in usual functional code.
-/
def recon : X ⟶ X :=
  V.dec ≫ V.enc

/-- This lemma just says that `recon` is *definitionally* equal to
    the composition `dec ≫ enc`. `rfl` ("reflexivity") means
    "by definition they are the same".
-/
@[simp] lemma recon_def :
    V.recon = V.dec ≫ V.enc := rfl

/-- Composing any morphism on the *left* with `recon`.

    Statement:
      for any `f : Y ⟶ X`,
      f ≫ V.recon = f ≫ V.dec ≫ V.enc.

    This is often useful because it lets the simplifier rewrite
    through `recon` into the explicit `dec ≫ enc`.
-/
@[simp] lemma comp_recon_left {Y : C} (f : Y ⟶ X) :
    f ≫ V.recon = f ≫ V.dec ≫ V.enc := by
  -- `simp` uses the lemma `recon_def` we just proved.
  simp [recon]

/-- Composing any morphism on the *right* with `recon`.

    Statement:
      for any `g : X ⟶ Y`,
      V.recon ≫ g = V.dec ≫ V.enc ≫ g.
-/
@[simp] lemma comp_recon_right {Y : C} (g : X ⟶ Y) :
    V.recon ≫ g = V.dec ≫ V.enc ≫ g := by
  -- Here we use associativity of composition (`Category.assoc`)
  -- and unfold `recon`.
  simp [recon, Category.assoc]

end VAEcat


/-
────────────────────────────────────────────────────────
  2. Concrete specialization in `Type`
────────────────────────────────────────────────────────

Here we drop down to the "ordinary" world of *types* and functions:

* `X` and `Z` are types (think: sets of inputs and latents).
* `X` has a (pseudo-)metric structure.
* `Z` is a real normed vector space.

This lets us talk about:

* distance between inputs,
* linear probes on the latent space.
-/

section TypeConcrete

open scoped BigOperators

/-- We now work with actual types `X` and `Z`. -/
variable {X Z : Type}

/-- Assume `X` (inputs) is a metric space.

    `PseudoMetricSpace` allows distance(x,y) = 0 even when x ≠ y,
    which is often enough for ML purposes.
-/
variable [PseudoMetricSpace X]

/-- Latent space `Z` is a real normed vector space.

    * `NormedAddCommGroup Z`: `Z` is an additive group with a norm.
    * `NormedSpace ℝ Z`: we can scale vectors in `Z` by real numbers,
      and this scaling is compatible with the norm.
-/
variable [NormedAddCommGroup Z] [NormedSpace ℝ Z]

/-- `approxId f ε` means:

      "for every point `x`, the distance between `f x` and `x`
       is at most `ε`"

    In words: `f` is ε-close to the identity map on `X` (sup-norm).
    We'll use this to say that `dec ∘ enc` is approximately the
    identity map on data.
-/
def approxId (f : X → X) (ε : ℝ) : Prop :=
  ∀ x : X, dist (f x) x ≤ ε

/-- A *concrete* VAE between types `X` and `Z`.

    This is the same shape as `VAEcat`, but now the arrows are
    *functions* instead of abstract morphisms.  We keep it minimal:
    only encoder and decoder.
-/
structure VAE where
  /-- encoder from inputs `X` to latents `Z` -/
  enc : X → Z
  /-- decoder from latents `Z` back to inputs `X` -/
  dec : Z → X

namespace VAE

/-- In this namespace we fix a particular VAE `V`. -/
variable (V : VAE X Z)

/-- Concrete reconstruction function `X → X`, mirroring the
    categorical `recon : X ⟶ X` from above.

    For an input `x`, we encode to latent `z := enc x`, then decode
    back: `dec z`.  This is the deterministic part of the VAE.
-/
def recon (x : X) : X :=
  V.dec (V.enc x)

end VAE

/-- A linear "concept probe" from latent space `Z` to the reals,
    plus a client-readable `name` for the concept.

    `Z →L[ℝ] ℝ` is Lean's type for *continuous linear maps* from `Z`
    to `ℝ` over the field `ℝ`.  You can think of this as a learned
    linear probe.
-/
structure Probe where
  w    : Z →L[ℝ] ℝ
  name : String

/-- Specification relating a VAE, a probe, and a client concept.

    Given:
      * `V` : VAE X Z
      * `p` : Probe Z
      * `c` : X → ℝ       (a "client concept function" on inputs)

    A value `ProbeSpec V p c` is a *witness* that:

      * There exists an allowed error budget `δ ≥ 0`.
      * For every input `x`, the difference between
            c x              (true concept)
        and  p.w (V.enc x)   (probe applied to latent)
        is at most `δ` in absolute value.

    This is one simple way to turn "the probe explains concept c"
    into a precise logical proposition.
-/
structure ProbeSpec (V : VAE X Z) (p : Probe Z) (c : X → ℝ) :=
  /-- Allowed sup-norm error between `c x` and `p.w (enc x)`. -/
  δ        : ℝ
  /-- We usually require our error budget to be non-negative. -/
  δ_nonneg : 0 ≤ δ
  /-- Approximation property: the probe tracks the concept up to `δ`. -/
  approx   : ∀ x : X, |c x - p.w (V.enc x)| ≤ δ

/-
could later extend `ProbeSpec` with a "nuisance subspace"
`N : Submodule ℝ Z` and an invariance condition:

  ∀ (z : Z) (n ∈ N), p.w (z + n) = p.w z

but keep that out of this first file to stay simple.
-/

end TypeConcrete
