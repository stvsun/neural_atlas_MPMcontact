/-
  OTContact.RadialSign
  ====================

  Machine-checked (mathlib-free, over `Int`) sign-exactness of the radial vs perpendicular gap.

  Source: `solvers/contact/chart_gap.py` lines 37-39 —

        "The radial gap is **not** a Euclidean distance: gap_rad = gap_perp / cos(alpha)
         ... Since 1/cos(alpha) ≥ 1 and the sign is preserved, its [active set is the same]."

  Equivalently `g_perp = g_rad · cos α` with `cos α > 0` (the chart flank condition).  The contact
  active set is `{ g < 0 }`; for the penalty/AL clamp to select the SAME active set whether it reads
  the radial gap or the true perpendicular (normal) gap, the two must have the same sign at every
  point.  Because `cos α > 0`, multiplication by `cos α` is sign-preserving, so:

    * `radial_gap_sign_agree`  — `g_rad · g_perp ≥ 0` (the two gaps never have strictly opposite
                                  signs: their product is nonnegative).
    * `active_set_iff`         — `g_perp < 0 ↔ g_rad < 0` (penetration detected by one iff by the
                                  other — the active set is identical).
    * `open_set_iff`           — `g_perp = 0 ↔ g_rad = 0` (the contact boundary coincides).

  We model `cos α` by a positive integer `c` (`0 < c`); the only fact used is that scaling by a
  positive number preserves sign, which is exactly `1/cos α ≥ 1` with sign preserved.  Working over
  `Int` keeps the certificate hermetic; the result is the discrete sign-exactness claim, independent
  of the magnitude bias `1/cos α` (which is a separate, documented, *magnitude* property, not a sign
  property — see CLAUDE.md / manual §11.2).
-/
namespace OTContact.RadialSign

/-- **Sign agreement.**  With `g_perp = g_rad · cos α` and `cos α > 0`, the product
    `g_rad · g_perp = (g_rad)² · cos α ≥ 0`: the radial and perpendicular gaps never carry strictly
    opposite signs. -/
theorem radial_gap_sign_agree (grad c : Int) (hc : 0 < c) : 0 ≤ grad * (grad * c) := by
  have heq : grad * (grad * c) = (grad * grad) * c := (Int.mul_assoc grad grad c).symm
  rw [heq]
  have hsq : 0 ≤ grad * grad := by
    rcases Int.le_total 0 grad with h | h
    · exact Int.mul_nonneg h h
    · have h' : 0 ≤ -grad := by omega
      have hh := Int.mul_nonneg h' h'
      rwa [Int.neg_mul_neg] at hh
  exact Int.mul_nonneg hsq (Int.le_of_lt hc)

/-- **Active-set exactness.**  Penetration under the perpendicular (normal) gap holds iff it holds
    under the radial gap: `g_rad · cos α < 0 ↔ g_rad < 0` for `cos α > 0`.  Hence the clamp
    `{ g < 0 }` selects the SAME active set under either measure. -/
theorem active_set_iff (grad c : Int) (hc : 0 < c) : grad * c < 0 ↔ grad < 0 := by
  rcases Int.lt_trichotomy grad 0 with hlt | heq | hgt
  · constructor
    · intro _; exact hlt
    · intro _
      have hneg : 0 < -grad := by omega
      have hp : 0 < (-grad) * c := Int.mul_pos hneg hc
      have e : (-grad) * c = -(grad * c) := by rw [Int.neg_mul]
      rw [e] at hp; omega
  · subst heq; simp
  · constructor
    · intro h
      have : 0 < grad * c := Int.mul_pos hgt hc
      omega
    · intro h; omega

/-- **Contact-boundary coincidence.**  The two gaps vanish together: `g_rad · cos α = 0 ↔
    g_rad = 0` for `cos α > 0`.  So the open/closed boundary is the same under either measure. -/
theorem open_set_iff (grad c : Int) (hc : 0 < c) : grad * c = 0 ↔ grad = 0 := by
  constructor
  · intro h
    rcases Int.mul_eq_zero.mp h with h0 | h0
    · exact h0
    · omega
  · intro h; rw [h]; simp

end OTContact.RadialSign
