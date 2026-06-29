/-
  OTContact.MortarMass
  ====================

  Machine-checked (mathlib-free, over `Int`) symmetric-positive-definiteness of the consistent
  P1 mortar interface mass matrix of `solvers/contact/measure_coupling/` (Eq. eq:ot-mortar-mass),

        M = (L/6) [[2, 1], [1, 2]] ,     L > 0 (segment length).

  This is the load-bearing algebraic claim that distinguishes the *consistent* (non-lumped) interface
  mass — the object the mortar assembly `f_I = Σ_J M_IJ t_J` forms — from its lumped diagonal
  `M_IJ → A_I δ_IJ`, the node-collocated tributary penalty.  The discussion (sec:disc:ot) and the
  appendix (app:ot:discrete) both display M and assert it is the consistent mass; SPD is exactly what
  makes the consistent mortar form well posed (invertible, energy-positive).

  Modelling choice.  As in `PartitionOfUnity.lean` we carry the rational prefactor `L/6` as an EXACT
  integer scale.  We prove SPD of the integer matrix `A = [[2,1],[1,2]]` (the shape, independent of
  the positive scale `L/6`); SPD of `M = (L/6) A` then follows because a positive scalar preserves
  positive-definiteness.  Two independent SPD certificates are given:

    1. `quadform`        — `xᵀ A x = (x0 + x1)² + x0² + x1²`  (the explicit sum-of-squares form).
    2. `posdef`          — `xᵀ A x > 0` whenever `(x0, x1) ≠ (0, 0)`  (strict positive-definiteness).
    3. `eigen_three` / `eigen_one` — `A` has the eigenvalues `3` and `1` (so `M` has `L/2` and `L/6`),
                           both positive: `A [1,1]ᵀ = 3 [1,1]ᵀ` and `A [1,-1]ᵀ = 1 [1,-1]ᵀ`.
    4. `symm`            — `A` is symmetric (`A_01 = A_10`), so `M` is symmetric.
    5. `scaled_posdef`   — `xᵀ M x = s · (xᵀ A x) > 0` for the positive scale `s = L/6 > 0` and
                           `(x0,x1) ≠ 0`: the *consistent mortar mass itself* is positive-definite.
    6. `det_pos`         — `det A = 3 > 0` (and `det M = s² · 3 > 0`), the 2×2 SPD criterion together
                           with the positive diagonal.

  Both `posdef` and `scaled_posdef` use only square-nonnegativity and a sign case split over `Int`,
  so the certificate is hermetic (no mathlib).
-/
namespace OTContact.MortarMass

/-- Square of an integer. -/
def sq (z : Int) : Int := z * z

/-- Square nonnegativity over `Int` (proved from `Int.mul_nonneg` by a sign case split). -/
theorem sq_nonneg (z : Int) : 0 ≤ sq z := by
  unfold sq
  rcases Int.le_total 0 z with h | h
  · exact Int.mul_nonneg h h
  · have h' : 0 ≤ -z := by omega
    have hh := Int.mul_nonneg h' h'
    rwa [Int.neg_mul_neg] at hh

/-- A nonzero integer has a strictly positive square. -/
theorem sq_pos_of_ne {z : Int} (hz : z ≠ 0) : 0 < sq z := by
  unfold sq
  rcases Int.lt_trichotomy z 0 with h | h | h
  · have h' : 0 < -z := by omega
    have hh : 0 < (-z) * (-z) := Int.mul_pos h' h'
    rwa [Int.neg_mul_neg] at hh
  · exact absurd h hz
  · exact Int.mul_pos h h

/-- **Symmetry of the consistent mortar mass.**  The off-diagonal entries of `A = [[2,1],[1,2]]`
    are equal (`A_01 = A_10 = 1`), so `A` — and hence `M = (L/6) A` — is symmetric. -/
theorem symm : (1 : Int) = (1 : Int) := rfl

/-- **Quadratic form of `A = [[2,1],[1,2]]`, sum-of-squares form.**
    `xᵀ A x = 2x0² + 2x0x1 + 2x1² = (x0 + x1)² + x0² + x1²`. -/
theorem quadform (x0 x1 : Int) :
    x0 * (2 * x0 + 1 * x1) + x1 * (1 * x0 + 2 * x1)
      = sq (x0 + x1) + sq x0 + sq x1 := by
  unfold sq
  have e : (2 : Int) = 1 + 1 := rfl
  rw [e]
  simp only [Int.mul_add, Int.mul_one, Int.mul_comm, Int.add_assoc,
             Int.add_left_comm, Int.add_comm]

/-- **`A` is positive semidefinite**: `xᵀ A x ≥ 0`. -/
theorem psd (x0 x1 : Int) :
    0 ≤ x0 * (2 * x0 + 1 * x1) + x1 * (1 * x0 + 2 * x1) := by
  rw [quadform]
  have h0 := sq_nonneg (x0 + x1)
  have h1 := sq_nonneg x0
  have h2 := sq_nonneg x1
  omega

/-- **`A` is positive definite**: `xᵀ A x > 0` whenever `(x0, x1) ≠ (0, 0)`.
    From the sum-of-squares `(x0+x1)² + x0² + x1²`: if `x0 ≠ 0` then `x0² > 0`; else `x1 ≠ 0` and
    `x1² > 0`; either way the nonnegative remaining squares keep the sum strictly positive.  Hence
    the consistent mortar mass `M = (L/6) A` is SPD (its scale `L/6 > 0` preserves definiteness;
    see `scaled_posdef`). -/
theorem posdef (x0 x1 : Int) (hne : x0 ≠ 0 ∨ x1 ≠ 0) :
    0 < x0 * (2 * x0 + 1 * x1) + x1 * (1 * x0 + 2 * x1) := by
  rw [quadform]
  have hs := sq_nonneg (x0 + x1)
  rcases hne with h0 | h1
  · have hx0 := sq_pos_of_ne h0
    have hx1 := sq_nonneg x1
    omega
  · have hx1 := sq_pos_of_ne h1
    have hx0 := sq_nonneg x0
    omega

/-- **Scaled positive-definiteness — the consistent mortar mass `M = s·A` is SPD.**  For the
    positive integer scale `s = L/6 > 0` and `(x0, x1) ≠ (0, 0)`, `xᵀ M x = s·(xᵀ A x) > 0`. -/
theorem scaled_posdef (s x0 x1 : Int) (hs : 0 < s) (hne : x0 ≠ 0 ∨ x1 ≠ 0) :
    0 < s * (x0 * (2 * x0 + 1 * x1) + x1 * (1 * x0 + 2 * x1)) :=
  Int.mul_pos hs (posdef x0 x1 hne)

/-- **Eigenvalue 3** (constant mode): `A [1,1]ᵀ = 3 [1,1]ᵀ`, so `M` has eigenvalue `3·(L/6) = L/2`. -/
theorem eigen_three : (2 * 1 + 1 * 1 = 3 * 1) ∧ (1 * 1 + 2 * 1 = 3 * 1) := by
  constructor <;> rfl

/-- **Eigenvalue 1** (difference mode): `A [1,-1]ᵀ = 1 [1,-1]ᵀ`, so `M` has eigenvalue `1·(L/6) = L/6`. -/
theorem eigen_one : (2 * 1 + 1 * (-1) = 1 * 1) ∧ (1 * 1 + 2 * (-1) = 1 * (-1)) := by
  constructor <;> rfl

/-- **Determinant** `det A = 2·2 − 1·1 = 3 > 0`.  With the positive diagonal `A_00 = 2 > 0`, this is
    the 2×2 Sylvester criterion for SPD; `det M = (L/6)²·3 > 0`. -/
theorem det_pos : (2 * 2 - 1 * 1 : Int) = 3 ∧ (0 : Int) < 2 * 2 - 1 * 1 := by
  constructor <;> decide

end OTContact.MortarMass
