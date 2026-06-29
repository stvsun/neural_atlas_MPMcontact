/-
  OTContact.BrenierProposed
  =========================

  The genuinely measure-theoretic backbone of the OT-contact thesis.  Brenier existence/uniqueness
  and "the 1-D monotone rearrangement `T = F_B⁻¹ ∘ F_A` pushes `μ_A` to `μ_B`" require the
  mathlib `MeasureTheory` / `Probability` libraries (Lebesgue–Stieltjes measures, pushforward,
  monotone rearrangement, cyclical monotonicity).  To keep this project hermetic and fast we do NOT
  pull mathlib here, so those two results are stated as FORMAL statements with a documented `sorry`
  and a rigorous prose proof, clearly labelled

        "PROPOSED — not machine-checked (requires mathlib measure theory)".

  What IS machine-checked here (mathlib-free, over an abstract CDF interface): the ALGEBRAIC heart of
  the monotone map — the quantile / marginal identity `F_B(T(x)) = F_A(x)` — which is the property
  the code's `MonotoneCoupling1D.map` (`coupling.py`) computes and the `math_verification.md`
  Component 1 verifies to machine precision (`max|F_m(T(ξ)) − F_s(ξ)| = 1.1e-16`).  The identity
  follows purely from `F_B⁻¹` being a right inverse of `F_B` on the range of `F_A`; no measure theory
  is needed for the identity itself, only for the statement that it *characterises the pushforward*.
-/
namespace OTContact.BrenierProposed

/- -------------------------------------------------------------------------------------------------
   MACHINE-CHECKED: the quantile / marginal identity that defines the 1-D monotone (Brenier) map.
   -------------------------------------------------------------------------------------------------

   We model the CDFs `F_A, F_B : Int → Int` and the pseudo-inverse `F_B⁻¹ : Int → Int` as functions
   on `Int` (a discretised quantile axis — exactly the `np.interp(q, Fm1, master_x)` lookup the code
   performs).  The transport map is `T := F_B⁻¹ ∘ F_A`.  Given that `F_B⁻¹` is a right inverse of
   `F_B` on the relevant range (the well-definedness guaranteed in code by the STRICT monotonicity of
   the arclength CDF, `√(1+h'²) ≥ 1 > 0`, see `_arclength_cdf`), the quantile identity holds. -/

variable (F_A F_B Finv_B : Int → Int)

/-- The transport map `T = F_B⁻¹ ∘ F_A`. -/
def T (x : Int) : Int := Finv_B (F_A x)

/-- **Quantile / marginal identity (machine-checked).**  If `F_B⁻¹` is a right inverse of `F_B`
    (`F_B (F_B⁻¹ q) = q`, the well-defined inverse of the strictly increasing arclength CDF), then
    the monotone map `T = F_B⁻¹ ∘ F_A` satisfies `F_B(T(x)) = F_A(x)` for every `x`.  This is the
    identity `F_m(T(ξ)) = F_s(ξ)` of `coupling.py` / Component 1 — the discrete mass-marginal /
    equal-cumulative-arclength-quantile characterisation of the OT map. -/
theorem quantile_identity (right_inv : ∀ q, F_B (Finv_B q) = q) (x : Int) :
    F_B (T F_A Finv_B x) = F_A x := by
  unfold T
  exact right_inv (F_A x)

/-- **Endpoint mass normalisation (machine-checked).**  At the endpoints the normalised CDFs hit
    `0` and `1` (`coupling.py`: `_Fs1 = Fs / Fs[-1]`, asserted `Fs1[0]=0`, `Fs1[-1]=1`).  Encoded
    as: if `F_A` sends the left endpoint `xL` to `0` and `F_B⁻¹ 0 = yL`, then `T xL = yL` — the map
    carries left endpoint to left endpoint (mass conservation at the boundary). -/
theorem endpoint_left (xL yL : Int)
    (hA : F_A xL = 0) (hInv : Finv_B 0 = yL) :
    T F_A Finv_B xL = yL := by
  unfold T
  rw [hA, hInv]

/- -------------------------------------------------------------------------------------------------
   PROPOSED — not machine-checked (requires mathlib measure theory).
   ------------------------------------------------------------------------------------------------- -/

/-- Opaque interface stand-ins for the measure-theoretic objects (would be
    `MeasureTheory.Measure ℝ`, `MeasureTheory.Measure.map`, `Monotone`, etc. in a mathlib build). -/
opaque IsProbMeasure : (Int → Int) → Prop
opaque Pushforward : (Int → Int) → (Int → Int) → (Int → Int)   -- (map, μ) ↦ T_# μ
opaque MeasEq : (Int → Int) → (Int → Int) → Prop                 -- measure equality
opaque MonotoneMap : (Int → Int) → Prop
opaque GradConvex : (Int → Int) → Prop                            -- T = ∇φ, φ convex

/--
  **PROPOSED — not machine-checked (requires mathlib measure theory).**

  *Brenier's theorem, 1-D specialisation.*  Let `μ_A, μ_B` be Borel probability measures on `ℝ`
  with `μ_A` atomless, and quadratic cost `c(x,y) = ½|x−y|²`.  Then there exists a μ_A-a.e. unique
  optimal transport map `T` pushing `μ_A` onto `μ_B`, `T_# μ_A = μ_B`, and `T` is the gradient of a
  convex function — in 1-D, `T` is non-decreasing and equals the monotone rearrangement
  `T = F_B⁻¹ ∘ F_A` (`F_A, F_B` the CDFs).

  *Rigorous prose proof (1-D).*  Existence/uniqueness: the optimal Kantorovich potential `φ` for the
  quadratic cost is convex (c-concavity for `c=½|·|²` is concavity of `x ↦ φ(x) − ½x²`, equivalently
  convexity of `φ`), and `T = ∇φ = φ'` exists `μ_A`-a.e. since a convex function is differentiable
  outside a countable set, which is `μ_A`-null because `μ_A` is atomless.  `φ'` is non-decreasing
  (derivative of a convex function), so `T` is monotone.  A monotone map pushing the atomless `μ_A`
  to `μ_B` must equate cumulative masses: for every `x`, `μ_B((−∞, T(x)]) = μ_A((−∞, x])`, i.e.
  `F_B(T(x)) = F_A(x)`; solving gives `T(x) = F_B⁻¹(F_A(x))` (with `F_B⁻¹` the left-continuous
  generalised inverse).  Uniqueness: any two monotone maps pushing `μ_A` to `μ_B` agree
  `μ_A`-a.e. because the equal-CDF constraint pins `T` at every continuity point of `F_A`, a full
  set.  Optimality: a monotone (cyclically monotone) map is optimal for any convex cost by the
  rearrangement inequality.  ∎

  The DISCRETE algebraic core of this — the identity `F_B(T(x)) = F_A(x)` — is machine-checked above
  as `quantile_identity`. -/
theorem brenier_existence_uniqueness_proposed (μA μB : Int → Int)
    (hA : IsProbMeasure μA) (hB : IsProbMeasure μB) :
    ∃ Tmap, MonotoneMap Tmap ∧ GradConvex Tmap ∧ MeasEq (Pushforward Tmap μA) μB := by
  sorry

/--
  **PROPOSED — not machine-checked (requires mathlib measure theory).**

  *The 1-D monotone map pushes `μ_A` to `μ_B`.*  With `T = F_B⁻¹ ∘ F_A`, the pushforward
  `T_# μ_A = μ_B`.

  *Rigorous prose proof.*  For any `y`, `T_#μ_A((−∞,y]) = μ_A(T⁻¹((−∞,y])) = μ_A({x : F_B⁻¹(F_A(x)) ≤ y})`.
  Since `F_B⁻¹` is the generalised inverse of the non-decreasing right-continuous `F_B`, the event
  `F_B⁻¹(q) ≤ y` is `q ≤ F_B(y)` (Galois connection of the generalised inverse).  Hence the set is
  `{x : F_A(x) ≤ F_B(y)}`, whose `μ_A`-measure is `F_A(F_A⁻¹(F_B(y))) = F_B(y)` because
  `F_A_# μ_A = Uniform[0,1]` (probability integral transform for atomless `μ_A`).  Therefore
  `T_#μ_A((−∞,y]) = F_B(y) = μ_B((−∞,y])` for all `y`, and two probability measures with equal CDFs
  are equal.  ∎

  The discrete witness of the central step (`F_B(T(x)) = F_A(x)`) is `quantile_identity` above. -/
theorem monotone_map_pushes_forward_proposed (μA μB : Int → Int)
    (Tmono : Int → Int)
    (hT : ∀ x, Tmono x = Finv_B (F_A x))
    (hInvGalois : ∀ q, F_B (Finv_B q) = q) :
    MeasEq (Pushforward Tmono μA) μB := by
  sorry

end OTContact.BrenierProposed
