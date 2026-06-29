/-
  OTContact — machine-checked algebraic core of the optimal-transport (measure-coupling)
  contact model of `solvers/contact/measure_coupling/`.

  Three families of lemmas are formalised and `lake build`-checked (mathlib-free, over `Int`):

    * `OTContact.PartitionOfUnity` — the P1 host weights N0 = 1-t, N1 = t are a partition of
      unity (N0 + N1 = 1) and lie in [0,1] for t in [0,1]; the discrete OT marginal at a Gauss
      point sums to 1.  (Code: `two_body._locate_master`, `coupling.ClosestPointCoupling1D.map_full`.)

    * `OTContact.TangentPSD` — the single-active-point contact tangent block K = eps (a ⊗ a) is
      symmetric and positive semidefinite: xᵀ K x = eps (a · x)² ≥ 0.  This is the algebraic core
      of the "symmetric SPSD 4-block tangent" of `two_body.assemble_two_body_contact`.

    * `OTContact.RadialSign` — with cos α > 0, the radial gap g_rad and the perpendicular gap
      g_perp = g_rad cos α have the same sign, so the active set { g < 0 } is identical under either
      measure (sign-exact active set).  (Code: `chart_gap.py` lines 37-39.)

  A fourth module states the genuinely measure-theoretic results (Brenier existence/uniqueness; the
  1-D monotone map T = F_B⁻¹ ∘ F_A pushes μ_A to μ_B) as formal statements with a documented `sorry`
  and a rigorous prose proof — labelled "proposed, not machine-checked".
-/
import OTContact.PartitionOfUnity
import OTContact.TangentPSD
import OTContact.RadialSign
import OTContact.BrenierProposed
