# MPM velocity-gradient audit for curved charts

**Status:** **RESOLVED — Option 3 applied.** The bug below is fixed; this document is retained as the design record. See the "Resolution (applied)" section at the end for the implementation pointer and regression coverage.
**Discovered:** During the Phase 4 / Phase 7 debug audit of the contact framework.
**Resolved:** Via Option 3 (chain-rule transform of shape-function gradients inside `_shape_functions_and_indices`). All 250 tests pass, including 5 new curved-chart tests in `tests/test_mpm_curved_chart.py`.
**Scope:** MPM correctness on non-identity chart decoders. Does **not** affect the contact framework (Phases 1–7b), any existing test, or any existing benchmark, because every MPM caller in the repository uses an identity decoder. The fix is fully backward-compatible — when no decoder bundle is supplied to `ChartMPMSolver`, the legacy identity path runs unchanged.

---

## TL;DR

The velocity gradient computed in `solvers/mpm/transfers.py::grid_to_particle` is

$$
\widetilde L_p \;=\; \sum_I v^{\text{phys}}_I \,(\nabla_{\xi} N_I)(\xi_p) \;=\; \frac{\partial v^{\text{phys}}}{\partial \xi}\bigg|_{\xi_p}.
$$

This is a **mixed-coordinate tensor**: the numerator is physical velocity, the denominator is parametric (chart-local) coordinate. The subsequent deformation-gradient update in `solvers/mpm/particles.py::update_deformation_gradient` treats it as the physical spatial velocity gradient

$$
L_p \;=\; \frac{\partial v^{\text{phys}}}{\partial x^{\text{phys}}}\bigg|_{\xi_p}.
$$

By the chain rule these two differ by a factor of `J^{-1}`, where `J = ∂φ/∂ξ` is the chart decoder Jacobian:

$$
L_p \;=\; \widetilde L_p \cdot J_p^{-1}.
$$

For an identity decoder `J = I` and `L = \widetilde L`, so every existing unit test and benchmark (all of which use identity decoders) silently returns the correct answer. The moment a curved chart decoder is wired into `SchwarzMPMSolver`, `F` accumulates the wrong tensor and the constitutive model receives a physically incorrect deformation gradient.

The same bug pattern also appears in the P2G internal-force scatter, which drops a dual `J^{-T}` factor. Both bugs share a root cause: `_shape_functions_and_indices` returns shape-function gradients in **ξ-space** rather than **physical space**, and the callers use them as if they were already in physical space.

---

## Dimensional analysis of the P2G / G2P pipeline

| Tensor | Shape | Space | Source |
|---|---|---|---|
| `particles.xi` | (N, 3) | ξ (parametric) | `particles.py:14`, "position ξ in chart-local coords" |
| `particles.v` | (N, 3) | x (physical) | `particles.py:15`, "velocity v in physical space" |
| `particles.V0` | (N,) | x (physical volume) | `particles.py:31` |
| `particles.F` | (N, 3, 3) | **x/x (physical-physical)** — by implication of the Neo-Hookean model, which uses `J = det F` as the physical volume ratio and computes `B = F F^T` as the physical left Cauchy-Green tensor | `constitutive.py:39–88` |
| `particles.stress` | (N, 3, 3) | x (physical Cauchy) | `constitutive.py:82–86` |
| `grid.positions` | (n_nodes, 3) | ξ (parametric) | `grid.py:30–67`, built by `linspace(-extent, extent, …)` where `extent` is "Half-width of the grid domain in chart-local coordinates" |
| `grid.h` | scalar | ξ (parametric) | `grid.py:44`: `self.h = 2.0 * extent / n_cells` |
| `grid.momentum` | (n_nodes, 3) | x (physical) — constructed as `m_p * v_p * N_I` with physical `v_p` | `transfers.py:122–125` |
| `grid.velocity` | (n_nodes, 3) | x (physical) — `v = p / m` | `grid.py:69–75` |
| `grid.force` | (n_nodes, 3) | **mixed** — internal force contribution carries `∇_ξ N_I` units while gravity / contact contributions are pure physical vectors | `transfers.py:134–155` |

### Shape function gradients: ξ-space by construction

From `solvers/mpm/transfers.py:75–86`:

```python
# dw_1d = -sign(dist) / h     where h = grid.h = 2*extent/n_cells  (ξ-space)
# dist is also in grid-index units (ξ / h)
dw_1d = -sign_dist / h

grad_weights[:, :, d] = (
    dw_1d[:, :, d] * w_1d[:, :, other0] * w_1d[:, :, other1]
)
```

Because `h` is the ξ-space cell spacing, `dw_1d[..., d]` has units of `1/length_ξ`, and so does `grad_weights[..., d]`. The variable thus holds `(∂N_I / ∂ξ_d)(ξ_p)` — the gradient of the shape function with respect to the chart-local coordinate.

The docstring at `transfers.py:36` even spells this out explicitly:

> "grad_weights : torch.Tensor — (N_particles, 8, 3) shape function gradients in ξ-space."

### G2P velocity gradient: ∂v_phys / ∂ξ

From `solvers/mpm/transfers.py:202–207`:

```python
vel_grad[:, d, :] = (v_at_nodes.unsqueeze(2) * grad_weights).sum(dim=1)
```

Component-wise:

$$
\text{vel\_grad}[p, d, k] \;=\; \sum_I v^{\text{phys}}_{I, d} \,(\partial N_I / \partial \xi_k)(\xi_p) \;=\; \frac{\partial v^{\text{phys}}_d}{\partial \xi_k}\bigg|_{\xi_p}.
$$

Units: `(m_phys / s) / m_ξ`. This is neither `∇_x v` nor `∇_ξ v_ξ`.

### Deformation gradient update: assumes physical spatial gradient

From `solvers/mpm/particles.py:87–98`:

```python
self.F = torch.bmm(eye + dt * vel_grad, self.F)
```

This is the standard forward Euler discretisation of `dF/dt = L · F` where `L = ∂v^phys / ∂x^phys` is the **physical** spatial velocity gradient. It presumes `vel_grad` is `L`. But what's passed in is `∂v^phys/∂ξ`, not `L`.

### Constitutive contract: `F` is physical-physical

From `solvers/mpm/constitutive.py:66–88` (Neo-Hookean):

```python
J = det(F);  B = F F^T
sigma = (1/J) * (mu * (B - I) + lam * ln(J) * I)
```

This formula only makes physical sense when `F` is `∂x^phys_current / ∂X^phys_reference` and `det F` is a true volume ratio. The code also uses `current_volume = V0 * |det F|` (`particles.py:72–74`), which presumes the same convention.

So the implicit contract is:

1. `F` lives in `phys / phys` space.
2. `vel_grad` fed to `update_deformation_gradient` should be the physical `L`.
3. The gradient that drives it should be `∂v^phys / ∂x^phys`.

The G2P routine satisfies none of these. It computes and delivers `∂v^phys / ∂ξ`.

### Dual bug: P2G internal force also drops a J

The standard weak-form internal nodal force is

$$
f^{\text{int}}_I \;=\; -\sum_p V_p \,\sigma_p \cdot \nabla_{x^{\text{phys}}} N_I(x_p).
$$

The correct chain rule gives `∇_{x^phys} N_I = J^{-T} ∇_ξ N_I`, so the expression that should appear in `transfers.py:134–141` is:

```python
f_int_I = - Σ_p V_p * sigma_p @ (J_p_inv_T @ grad_weights)
```

The current code writes

```python
stress_grad = einsum("pij, pnj -> pni", particles.stress, grad_weights)
stress_grad = Vp.unsqueeze(1).unsqueeze(2) * stress_grad
... grid.force[:, d] -= stress_grad[:, :, d]
```

without any `J^{-T}` — which drops the exact same factor that the G2P side drops. The two bugs are not independent; they are dual to each other under the same change of variables.

---

## Correctness verdict

**(B) Inconsistent.** Hidden factor of `J^{-1}` (and its dual `J^{-T}`) missing in two places:

1. **Internal force scatter** at `solvers/mpm/transfers.py:137` — should be `σ_p · (J_p^{-T} ∇_ξ N_I)`.
2. **Velocity gradient gather** at `solvers/mpm/transfers.py:207` — should be `(vel_grad) · J_p^{-1}`, or equivalently should use `∇_x N_I = J^{-T} ∇_ξ N_I` from the start.

Both collapse to identity for `J = I`, which is why no existing test fires on this.

A dimensionally correct G2P would be:

```python
# J_p, J_p_inv computed once from the decoder at ξ_p
grad_weights_phys = einsum(
    "pij,pnj->pni",
    J_p_inv.transpose(-1, -2),
    grad_weights,
)  # ∂N_I/∂x_phys at ξ_p

vel_grad_L = einsum(
    "In,pIk->pnk",
    v_grid_at_nodes,
    grad_weights_phys,
)  # = ∂v_phys / ∂x_phys  = spatial velocity gradient L

F_new = (I + dt * vel_grad_L) @ F_old
```

And the mirror change in P2G internal force scatter:

```python
stress_grad = einsum(
    "pij, pnj -> pni",
    particles.stress,
    grad_weights_phys,
)
```

---

## Empirical verification strategy

A clean affine-decoder test isolates the bug because for an affine decoder `J` is constant and all `J`-related rates of change cancel.

### Setup
- Decoder: `φ(ξ) = α · ξ` with `α = 2`, so `J = 2 I`, `J^{-1} = 0.5 I`.
- Chart extent `0.5`, so the physical body covers `[-1, 1]^3`.
- Neo-Hookean, `E = 1e5`, `ν = 0.3`, density `1000`.

### Initial condition
Apply a uniform spatial velocity field that is linear in **physical** coordinates:

`v_phys(x) = (γ · x[1], 0, 0)` with `γ = 0.1 / s` (a simple shear).

Set particle velocities: `cloud.v[p, 0] = γ * x_phys[p, 1] = γ * α * cloud.xi[p, 1]`.

### Expected vs observed

The true spatial velocity gradient is `L_true[0, 1] = γ`, all other entries zero.

- **Correct G2P** would recover `vel_grad[:, 0, 1] ≈ γ`.
- **Current code** computes `vel_grad = L_true · J = γ · 2 · e_1 ⊗ e_2`, i.e. `vel_grad[:, 0, 1] ≈ 2γ` — exactly twice the correct value.

Sweeping `α ∈ {0.5, 1.0, 2.0, 4.0}` yields a buggy ratio of `α` between observed and analytic.

### Stronger test: integrated F mismatch

Run 100 steps of `dt = 1e-4`. The exact solution under constant `L` is `F(t) = exp(L·t)`. With `γ·t = 0.01`:

- **Correct F** at end: `I + 0.01 e_1 ⊗ e_2 + O(1e-4)`.
- **Buggy F** at end: `I + 0.02 e_1 ⊗ e_2 + O(1e-4)` — factor `α = 2` inflation.

Neo-Hookean `σ_12` ≈ `μ γ t` (small-strain limit); the buggy version gives roughly twice that, with sign and magnitude very visible.

---

## Severity assessment

| Caller | Decoder | Affected? |
|---|---|---|
| `tests/test_chart_mpm_solver.py` | None — uses `ChartMPMSolver` directly, `ξ` is `x`, `J = I` | **No.** |
| `tests/test_mpm_contact.py`, `test_mpm_friction.py`, `test_contact_*.py`, `test_self_contact.py` | All use `ChartMPMSolver` directly. `J = I`. | **No.** |
| `benchmarks/contact/*_mpm.py` | All identity (`two_sphere_collision_mpm.py:5` explicitly says so). | **No.** |
| `archive/benchmarks/rabbit_poisson_fem/`, `archive/benchmarks/fracture/*` (archived) | FEM, not MPM. | **No.** |
| `benchmarks/mpm_basic/` | Empty package. | n/a |
| `solvers/mpm/schwarz_mpm.py` (`SchwarzMPMSolver`) | The **only** path that combines real `ChartDecoder`s with `ChartMPMSolver`s (`cloud.push_to_physical(self.decoders[i], ...)` at `schwarz_mpm.py:250–255`). | **Yes — latent.** |
| Contact force pathway | Contact forces are computed in physical space and scattered to the grid as per-particle physical force vectors, without going through `σ · ∇N`. They use `f_I += f_p N_I(ξ_p)`, which is dimensionally clean. | **Not affected.** |

### Current exposure
- **Zero** — no code path in the repository today actually exercises the buggy branch in a meaningful way. Every MPM simulation to date uses `J = I`.

### Latent exposure
- **High** — the very next planned experiment, once a chart decoder is trained and MPM is run on it via `SchwarzMPMSolver`, would silently produce `F` and stress off by an `α`-dependent amount per step.
- Energy & momentum statistics: `KE = m_p v²` stays correct (no `J`), but elastic strain energy and work done by stress are both wrong.

### Detectability
- **Very low.** No test, assertion, or benchmark catches it. The mistake hides perfectly inside the `J = I` regime. A coincidental decoder with `|det J| ≈ 1` would also mask most of the macroscopic error.

---

## Recommended fix (analysis only — not applied)

### Option 1 — Multiply `vel_grad` by `J^{-1}` per particle inside G2P
Compute `J_p` once per particle, invert via SVD or `linalg.solve`, multiply from the right in `grid_to_particle`. Mirror change in `particle_to_grid` for the internal-force scatter (`σ_p · J_p^{-T} ∇_ξ N_I`). Existing identity tests still pass because `J = I ⇒ J^{-1} = I`. Reuses `common/geometry.py::chart_map_and_jacobian` and `stabilized_jacobian_ops` verbatim. **Recommended.**

### Option 2 — Store `F` in ξ-space and rewrite constitutive for Riemannian MPM
Makes the solver intrinsic to a curved manifold. Research-grade rewrite; not recommended unless the project's goal is specifically meshfree-on-a-manifold MPM.

### Option 3 — Compute `grad_weights` in physical space inside `_shape_functions_and_indices`
Same correctness scope as Option 1 but with a single point of change at the source. Requires threading the decoder through `_shape_functions_and_indices`, which currently has signature `(particles, grid)`. Slightly cleaner than Option 1 because both buggy call sites (`transfers.py:137` and `transfers.py:207`) become correct automatically. **Recommended as the implementation strategy.**

### Option 4 — Runtime guard + documentation
Add a one-line guard in `ChartMPMSolver.__init__` that refuses curved decoders, plus documentation at the affected functions. Zero runtime cost, zero risk, doesn't fix the physics but prevents silent misuse. **Recommended as an immediate stopgap.**

### Recommendation
Combine **Option 4 now** (documentation + runtime warnings at the affected sites) with **Option 3 when ready** (the principled fix). Option 3 requires careful design — specifically, deciding where the decoder lives inside the MPM pipeline and how the Jacobian is cached between P2G and G2P — so it's not a 1-hour change. Option 4 is a 5-minute change that prevents a quiet correctness bug from creeping into future experiments.

---

## File-line citations

- ξ-space shape function gradients: `solvers/mpm/transfers.py:38` (`h = grid.h`), `solvers/mpm/transfers.py:75–86` (`dw_1d = -sign_dist / h` and product), docstring `solvers/mpm/transfers.py:36`.
- Grid `h` is ξ-space: `solvers/mpm/grid.py:44`, `grid.py:24–25`.
- P2G internal force uses ξ-gradients: `solvers/mpm/transfers.py:134–141`.
- G2P velocity gradient is `∂v_phys/∂ξ`: `solvers/mpm/transfers.py:202–207`.
- Particle velocity is physical: `solvers/mpm/particles.py:15`, `transfers.py:130–132`.
- Forward Euler `F` update assumes physical `L`: `solvers/mpm/particles.py:87–98`.
- Neo-Hookean presumes physical `F`: `solvers/mpm/constitutive.py:39–88`, `particles.py:72–74`.
- Only curved-decoder caller path: `solvers/mpm/schwarz_mpm.py:51–107`, `solvers/mpm/schwarz_mpm.py:249–255`.
- Identity-decoder test suite: `tests/test_chart_mpm_solver.py:134–150`.
- Identity-decoder MPM benchmarks: `benchmarks/contact/two_sphere_collision_mpm.py:5` ("identity decoder").
- Jacobian utilities available for any fix: `common/geometry.py:37–77` (`chart_map_and_jacobian`, `stabilized_jacobian_ops`).

---

## Resolution (applied)

**Fix strategy:** Option 3 (chain-rule transform inside `_shape_functions_and_indices`).

**Mechanism.** `ChartMPMSolver` grew an optional decoder bundle (`decoder`, `seed`, `t1`, `t2`, `n_vec`, `chart_scale`). When all six fields are supplied, `ChartMPMSolver._compute_J_inv_T(particles)` is called once per `step()` and computes the transpose-inverse chart Jacobian at every particle via the existing helpers:

```python
_, _, J = chart_map_and_jacobian(decoder, particles.xi, seed, t1, t2, n_vec, chart_scale)
J_inv, _, _, _ = stabilized_jacobian_ops(J.detach(), sigma_floor=1e-8, det_floor=1e-10)
J_inv_T = J_inv.transpose(-1, -2)                       # (N, 3, 3)
```

The resulting `J_inv_T` is threaded into `particle_to_grid(..., J_inv_T=...)` and `grid_to_particle(..., J_inv_T=...)`, both of which forward it to `_shape_functions_and_indices(..., J_inv_T=...)`. Inside that helper, after the ξ-space `grad_weights` are built, a single `einsum` applies the chain rule:

```python
if J_inv_T is not None:
    grad_weights = torch.einsum("pkj,pij->pik", J_inv_T, grad_weights)
# Now grad_weights = ∇_x N_I(ξ_p), i.e. physical-space gradients.
```

Downstream, the P2G internal-force line (`einsum("pij,pnj->pni", stress, grad_weights)`) and the G2P velocity-gradient line (`(v_at_nodes.unsqueeze(2) * grad_weights).sum(dim=1)`) are **unchanged** — they simply consume the now-physical `grad_weights` and are dimensionally correct by construction. One surgical point of fix, no duplicated logic.

**`SchwarzMPMSolver` wiring.** The solver's `__init__` loop that constructs per-chart `ChartMPMSolver` instances now forwards `self.decoders[i]`, `self.seeds_t[i]`, `self.t1_t[i]`, `self.t2_t[i]`, `self.nvec_t[i]`, and `self.support_r_t[i]` to each child solver, so every chart in a multi-chart atlas automatically gets the curved-chart path. The mirror change in `SchwarzMPMSolver.add_charts` propagates the bundle to dynamically spawned charts (fracture + contact-topology spawning).

**Backward compatibility.** The `J_inv_T` argument defaults to `None` on `particle_to_grid`, `grid_to_particle`, and `_shape_functions_and_indices`. The decoder bundle defaults to `None` on `ChartMPMSolver.__init__`. Every pre-fix caller — all 245 pre-fix tests and every existing MPM benchmark — uses `None` implicitly and runs the legacy ξ-space gradient path byte-for-byte unchanged. The fix is invisible unless you deliberately opt in with a decoder bundle.

**Regression coverage.** `tests/test_mpm_curved_chart.py` (5 new tests):

| Test | What it verifies |
|---|---|
| `test_affine_decoder_j_eq_2I_velocity_gradient` | With `φ(ξ) = 2ξ` and `v_phys = (γ x[1], 0, 0)`, the G2P routine recovers `L[0,1] = γ` exactly — not the buggy `2γ`. This is the clean analytic verification called out in section (c) of this audit. |
| `test_affine_decoder_j_eq_half_I_velocity_gradient` | Mirror with `α = 0.5`. Handles `J` smaller than identity. |
| `test_static_equilibrium_curved_chart` | Particles at rest with `F = I` under `α = 2` stay at rest across 10 steps (KE < 1e-12). Verifies P2G internal force and G2P update are internally consistent on a curved chart. |
| `test_identity_decoder_matches_no_decoder_path` | `ChartMPMSolver(decoder=AffineDecoder(1.0), ...)` produces byte-identical particle state after 5 steps as `ChartMPMSolver()` with no decoder. The fix does not perturb the identity path. |
| `test_contact_invariant_under_decoder` | A ball falling onto a floor SDF gives the same physical-space trajectory whether the solver was built with `decoder=None` or `decoder=AffineDecoder(1.0)`. Regression guarantee that the contact path (which lives in physical space) is unaffected by the decoder threading. |

**Full regression status after fix:** 250 passed, 1 skipped, 0 failed — previously 245 passed, so +5 from the new tests, with no regressions anywhere in the 91 contact-framework tests or the 154 pre-existing fracture / topology / FEM tests.

**Stopgap warnings removed.** The docstring warning blocks previously added to `solvers/mpm/transfers.py::grid_to_particle` and `solvers/mpm/particles.py::update_deformation_gradient` have been rewritten to reflect the new contract: "pass `J_inv_T` for curved charts; identity path is the default and preserved".

**Out of scope (documented in the plan file, not addressed here).** Slip boundary conditions on curved charts still treat the ξ-space grid faces as slip planes, which is a pre-existing convention unrelated to the velocity-gradient fix. The FLIP velocity blend in `grid_to_particle` remains pure PIC (a pre-existing simplification). Friction wiring through `SchwarzMPMSolver.configure_contact` is still external. None of these issues affect the correctness of the fix above.
