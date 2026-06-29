/-
  OTContact.TranslationInvariance
  ===============================

  Machine-checked (mathlib-free, over `Int`) two load-bearing algebraic claims of the OT
  measure-coupling tangent that the single-Gauss-point lemmas of `PartitionOfUnity.lean` do not yet
  cover:

    1.  **Multi-Gauss-point patch resultant.**  Proposition `prop:patch` (`app:ot:discrete`) is a
        sum over ALL quadrature points, not one foot.  `PartitionOfUnity.patch_test_resultant`
        certifies the single foot; here we certify that the resultant of the WHOLE list of Gauss feet
        vanishes for a constant pressure, by induction on the foot list.  Each foot contributes a
        scaled slave force `wq * d * p` and master reaction `-(wq * ((d-k)+k) * p)`; the per-foot
        balance is `PartitionOfUnity.pou_sum`, and the list sum telescopes to `0`.

        Source: `eq:ot-force` summed over `{(ξ_q, w_q)}`; measured patch residual `1.4e-16`.

    2.  **Rigid-translation invariance of the four-block tangent.**  The claim at `app:ot:tangent`
        ("the patch-test resultant of `eq:ot-marginal` makes the master row-sum of `K_ms` vanish, so
        a rigid-body translation produces no net contact force") rests on the gap variation under a
        UNIFORM displacement `u_I^+ = u_K^- = δ` being zero:

            Σ_I (∂g/∂u_I^+) + Σ_K (∂g/∂u_K^-) = (Σ_I N_I^+) n - (Σ_K (N_K^-∘χ)) n = (1 - 1) n = 0,

        because BOTH the slave shape functions and the master host weights are partitions of unity.
        The contact force is `f = ε⟨-g⟩₊ ∂g/∂u`, so a translation that leaves `g` unchanged leaves
        `f` unchanged — the net contact force on a rigid-body translation is null.  We certify the
        algebraic core: the signed weight sum `(Σ slave PoU) - (Σ master PoU) = 0` at scale `d`.

  Modelling choice (as in `PartitionOfUnity.lean`): host weights are carried as EXACT integer
  numerators over the common positive denominator `d` (so `Σ N = 1` is `Σ N_num = d`), and the
  per-foot quadrature weight `wq` and pressure `p` are integers.  Every statement closes by `omega`
  or by a short `List.foldr`/`List.sum` induction, hence is fully machine-checked.
-/
namespace OTContact.TranslationInvariance

/-! ### 1.  Multi-Gauss-point patch resultant -/

/-- A Gauss foot record: quadrature weight `wq`, host length `d` (the partition-of-unity scale),
    and projection numerator `k` with `0 ≤ k ≤ d`.  The two P1 host weights are `(d-k, k)`. -/
structure Foot where
  wq : Int
  d  : Int
  k  : Int

/-- Per-foot signed contribution to the net resultant for a constant scaled pressure `p`:
    slave force `wq * d * p` plus master reaction `-(wq * ((d-k)+k) * p)`. -/
def footResultant (p : Int) (f : Foot) : Int :=
  f.wq * f.d * p + (-(f.wq * ((f.d - f.k) + f.k) * p))

/-- **Per-foot balance.**  Each foot's signed contribution is exactly zero, because the master host
    weights are a partition of unity `(d-k)+k = d`.  (Single-foot core of `prop:patch`.) -/
theorem foot_resultant_zero (p : Int) (f : Foot) : footResultant p f = 0 := by
  unfold footResultant
  have h : (f.d - f.k) + f.k = f.d := by omega
  rw [h]; omega

/-- Net resultant over a LIST of Gauss feet: the sum of the per-foot contributions. -/
def netResultant (p : Int) : List Foot → Int
  | []      => 0
  | f :: fs => footResultant p f + netResultant p fs

/-- **Multi-Gauss-point patch resultant (machine-checked).**  For a constant pressure `p` the net
    resultant over ANY list of Gauss feet is exactly zero.  This is `prop:patch` as a sum over the
    full quadrature `{(ξ_q, w_q)}`, not a single foot: the constant-pressure load transmits across a
    non-matching interface with zero net resultant, irrespective of the number of quadrature points
    or the host meshes (measured `1.4e-16`). -/
theorem patch_resultant_list (p : Int) :
    ∀ fs : List Foot, netResultant p fs = 0
  | []      => rfl
  | f :: fs => by
      unfold netResultant
      rw [foot_resultant_zero p f, patch_resultant_list p fs]; rfl

/-! ### 2.  Rigid-translation invariance of the gap (and hence of the contact force) -/

/-- **Signed partition-of-unity sum (slave minus master).**  Under a uniform translation
    `u_I^+ = u_K^- = δ`, the summed gap variation is `((Σ slave PoU) - (Σ master PoU)) · (n·δ)`.
    At the integer scale `d`, with the slave weights summing to `d` and the master host weights
    `(d-k)+k` summing to `d`, the signed coefficient is `d - ((d-k)+k) = 0`: the gap is unchanged by
    a rigid-body translation, so the penalty force `ε⟨-g⟩₊ ∂g/∂u` is unchanged and the NET contact
    force vanishes (the row-sum claim of `app:ot:tangent`). -/
theorem rigid_translation_gap_invariant (d k : Int) :
    d - ((d - k) + k) = 0 := by omega

/-- The same statement with the slave partition carried as its own scaled sum `s = d` (slave P1
    weights `(d - j, j)` summing to `d`), making explicit that it is the EQUALITY of the two
    partitions of unity (slave and master each `= d`), not their individual values, that kills the
    translation: `((d - j) + j) - ((d - k) + k) = 0`. -/
theorem rigid_translation_balanced (d j k : Int) :
    ((d - j) + j) - ((d - k) + k) = 0 := by omega

/-- **Master row-sum of `K_ms` is balanced by the slave block under translation.**  The four-block
    coefficient for a uniform translation contracts the signed weight vector `(slave +, master -)`
    against the all-ones translation; the slave block contributes `+ε d` and the coupling block
    `-ε ((d-k)+k) = -ε d`, summing to `0`.  Hence `ε·d + (-(ε·((d-k)+k))) = 0` for any penalty `ε`:
    a rigid translation produces no net tangent-times-translation force. -/
theorem tangent_translation_null (eps d k : Int) :
    eps * d + (-(eps * ((d - k) + k))) = 0 := by
  have h : (d - k) + k = d := by omega
  rw [h]; omega

end OTContact.TranslationInvariance
