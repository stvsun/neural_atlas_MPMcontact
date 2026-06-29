/-
  OTContact.TangentPSD
  ====================

  Machine-checked (mathlib-free, over `Int`) algebraic core of the
  "symmetric SPSD 4-block tangent" of
  `solvers/contact/measure_coupling/two_body.py::assemble_two_body_contact`.

  On a single ACTIVE Gauss point the penalty traction is `t = ε_n ⟨-g_N⟩₊ n`, and (holding the
  normal `n` fixed, as the one-body and two-body assemblies both do) its consistent linearisation is

        K = ε_n (a ⊗ a) ,        a := the assembled interpolation-weight vector ⊗ n .

  In the two-body assembly (`two_body.py` lines 211-231) the per-entry coefficient is literally

        coeff = (ε_n w_q) · (s_a w_a) · (s_b w_b) ,        blk = coeff · (n ⊗ n) ,

  i.e. `K` is the outer product `ε_n w_q · (b ⊗ b) ⊗ (n ⊗ n)` of the signed weight vector
  `b_a := s_a w_a` (slave sign +1, master sign −1) with itself, tensored with `n ⊗ n`.  Both
  factors are rank-1 symmetric PSD, so `K` is symmetric PSD.  This file certifies that algebra:

    1. `outer_symm`        — `(a ⊗ a)` is symmetric: `aᵢ aⱼ = aⱼ aᵢ`.
    2. `rank1_form`        — `xᵀ (a ⊗ a) x = (a · x)²` in 2-D (the physical normal `n ∈ ℝ²`).
    3. `quad_form_nonneg`  — `(a · x)² ≥ 0` (square nonnegativity, the PSD certificate).
    4. `rank1_psd`         — `xᵀ (ε (a ⊗ a)) x ≥ 0` for `ε ≥ 0` (the scaled single-active block).
    5. `block_coeff_symm`  — the assembled coefficient `ε (s_a w_a)(s_b w_b)` is symmetric in (a,b),
                             so the 4-block tangent satisfies `K_ms = K_smᵀ` (Newton's third law).
    6. `four_block_diag_nonneg` — each diagonal coefficient `ε (s_a w_a)²` is `≥ 0` for `ε ≥ 0`.

  Everything is over `Int` so the certificate is hermetic (no real-analysis library needed): the
  squares are the only ordered-ring facts used, and they are proved from `Int.mul_nonneg` by a
  sign case split.
-/
namespace OTContact.TangentPSD

/-- Square of an integer (the scalar `a · x`). -/
def sq (z : Int) : Int := z * z

/-- Square nonnegativity over `Int` (the PSD certificate), proved from `Int.mul_nonneg` by a
    sign case split — no mathlib `mul_self_nonneg` needed. -/
theorem sq_nonneg (z : Int) : 0 ≤ sq z := by
  unfold sq
  rcases Int.le_total 0 z with h | h
  · exact Int.mul_nonneg h h
  · have h' : 0 ≤ -z := by omega
    have hh := Int.mul_nonneg h' h'
    rwa [Int.neg_mul_neg] at hh

/-- **Symmetry of the rank-1 outer product** `a ⊗ a`: the `(i,j)` entry equals the `(j,i)` entry.
    This is why the single-active contact block (and hence each of the four assembled blocks) is
    symmetric. -/
theorem outer_symm (ai aj : Int) : ai * aj = aj * ai := Int.mul_comm ai aj

/-- **Rank-1 quadratic form, 2-D** (the physical normal lives in `ℝ²`):
    `xᵀ (a ⊗ a) x = (a · x)²`.  Written out from the entries `(a⊗a)_{ij} = aᵢ aⱼ`. -/
theorem rank1_form (a0 a1 x0 x1 : Int) :
    x0 * ((a0 * a0) * x0 + (a0 * a1) * x1) + x1 * ((a1 * a0) * x0 + (a1 * a1) * x1)
      = sq (a0 * x0 + a1 * x1) := by
  unfold sq; simp only [Int.mul_add, Int.mul_comm, Int.mul_left_comm]

/-- The quadratic form is nonnegative: `(a · x)² ≥ 0`. -/
theorem quad_form_nonneg (a0 a1 x0 x1 : Int) : 0 ≤ sq (a0 * x0 + a1 * x1) :=
  sq_nonneg _

/-- **Single-active-point contact tangent is PSD**:
    `xᵀ (ε (a ⊗ a)) x = ε (a · x)² ≥ 0` whenever the penalty `ε ≥ 0`.
    This is the algebraic statement "`K = ε (a ⊗ a)` is positive semidefinite". -/
theorem rank1_psd (eps a0 a1 x0 x1 : Int) (he : 0 ≤ eps) :
    0 ≤ x0 * (eps * ((a0 * a0) * x0 + (a0 * a1) * x1))
        + x1 * (eps * ((a1 * a0) * x0 + (a1 * a1) * x1)) := by
  have key : x0 * (eps * ((a0 * a0) * x0 + (a0 * a1) * x1))
        + x1 * (eps * ((a1 * a0) * x0 + (a1 * a1) * x1))
        = eps * sq (a0 * x0 + a1 * x1) := by
    unfold sq; simp only [Int.mul_add, Int.mul_comm, Int.mul_left_comm]
  rw [key]
  exact Int.mul_nonneg he (sq_nonneg _)

/-- **4-block symmetry / Newton's third law.**  The assembled per-entry coefficient
    `coeff(a,b) = ε (s_a w_a)(s_b w_b)` (the scalar prefactor of the `n ⊗ n` block in
    `two_body.py`) is symmetric under `(a,b) ↦ (b,a)`.  Applied with `a` a slave dof and `b` a
    master dof, this is exactly `K_sm = K_msᵀ` (the slave-master block equals the transpose of the
    master-slave block). -/
theorem block_coeff_symm (eps sa wa sb wb : Int) :
    eps * ((sa * wa) * (sb * wb)) = eps * ((sb * wb) * (sa * wa)) := by
  simp only [Int.mul_comm, Int.mul_left_comm]

/-- **4-block diagonal nonnegativity.**  Each diagonal coefficient `ε (s_a w_a)²` is `≥ 0` for
    `ε ≥ 0` (a square times a nonnegative scalar): the diagonal of the assembled tangent is
    nonnegative, consistent with SPSD. -/
theorem four_block_diag_nonneg (eps sa wa : Int) (he : 0 ≤ eps) :
    0 ≤ eps * sq (sa * wa) :=
  Int.mul_nonneg he (sq_nonneg _)

end OTContact.TangentPSD
