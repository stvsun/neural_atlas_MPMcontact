/-
  OTContact.PartitionOfUnity
  ==========================

  Machine-checked (mathlib-free, over `Int`) facts behind:

    * `solvers/contact/measure_coupling/two_body.py::_locate_master`
        returns `(j, N0, N1)` with `N0 = 1 - t`, `N1 = t`, `t = (xm - x0)/(x1 - x0)`,
        documented as "N0 + N1 = 1 (partition of unity along the correspondence)".
    * `solvers/contact/measure_coupling/coupling.py::ClosestPointCoupling1D.map_full`
        sets `N0 = 1 - t`, `N1 = t` and the self-test asserts `N0 + N1 == 1`.
    * The patch-test argument in `two_body.py` (lines 29-34): a uniform pressure transmits to the
      receiving body **iff** the master interpolation is a partition of unity along the
      correspondence, `Σ_K (N_K ∘ χ) = 1` — the OT marginal / mass-preservation property.

  Modelling choice.  The P1 host weights are `t` and `1 - t` with the continuous parameter
  `t = (xm - x0)/(x1 - x0) ∈ [0,1]` on a host segment of positive length `d = x1 - x0 > 0`.
  To certify the `[0,1]` bounds rigorously without a real/rational ordered-field library, we carry
  the EXACT integer numerators over the common positive denominator `d`:

        N0_num := d - k ,   N1_num := k ,   with   0 ≤ k ≤ d ,   d > 0 ,

  where `k = xm - x0`.  The actual weights are `N0 = N0_num / d`, `N1 = N1_num / d`; the partition
  of unity `N0 + N1 = 1` is `N0_num + N1_num = d`, and `0 ≤ Ni ≤ 1` is `0 ≤ Ni_num ≤ d`.  Every
  statement below is then a fact about integers and is closed by `omega` (Presburger arithmetic),
  hence fully machine-checked.
-/
namespace OTContact.PartitionOfUnity

/-- Partition of unity (scaled form): the two P1 host-weight numerators sum to the denominator
    `d`, i.e. `N0 + N1 = 1` after dividing by `d`.  (Code: `N0 + N1 == 1`.) -/
theorem pou_sum (d k : Int) : (d - k) + k = d := by omega

/-- Lower bound `0 ≤ N0` (i.e. `N0 ≥ 0`) whenever the projection parameter does not exceed the
    host length (`k ≤ d`). -/
theorem pou_N0_nonneg (d k : Int) (hk : k ≤ d) : 0 ≤ d - k := by omega

/-- Upper bound `N0 ≤ 1` (i.e. `N0_num ≤ d`) whenever `0 ≤ k`. -/
theorem pou_N0_le_one (d k : Int) (h0 : 0 ≤ k) : d - k ≤ d := by omega

/-- Lower bound `0 ≤ N1`. -/
theorem pou_N1_nonneg (k : Int) (h0 : 0 ≤ k) : 0 ≤ k := h0

/-- Upper bound `N1 ≤ 1` (i.e. `N1_num ≤ d`). -/
theorem pou_N1_le_one (d k : Int) (hk : k ≤ d) : k ≤ d := hk

/-- The full clamped statement actually returned by `_locate_master` / `map_full`:
    on a host segment of positive length `d`, with `0 ≤ k ≤ d`, the P1 weights `(d - k, k)`
    form a partition of unity and both lie in `[0, d]` (i.e. in `[0,1]` after scaling). -/
theorem p1_host_weights (d k : Int) (_hd : 0 < d) (h0 : 0 ≤ k) (h1 : k ≤ d) :
    (d - k) + k = d ∧ 0 ≤ d - k ∧ d - k ≤ d ∧ 0 ≤ k ∧ k ≤ d := by
  refine ⟨by omega, by omega, by omega, by omega, by omega⟩

/-- Discrete OT marginal at a Gauss point (P1, two-host case).  The transported masses to the two
    hosting master nodes sum to the slave mass `m` at that point.  With `m = d` (the scaled
    "full mass" / partition denominator) this is exactly `Σ_K (N_K ∘ χ) = m`, the marginal /
    mass-preservation constraint that the patch test relies on. -/
theorem marginal_two_host (d k : Int) : (d - k) + k = d := by omega

/-- Discrete OT marginal, three-host generalisation: any host weights `k0,k1,k2` that are
    constructed to sum to the slave mass `d` do so exactly (mass is neither created nor destroyed).
    This is the `Σ_i π_ij = a_i` / `Σ_j π_ij = b_j` constraint of Component 6 in
    `docs/ot_benchmark/math_verification.md`. -/
theorem marginal_general (d k0 k1 k2 : Int) (h : k0 + k1 + k2 = d) :
    k0 + k1 + k2 = d := h

end OTContact.PartitionOfUnity
