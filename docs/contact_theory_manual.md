# Contact Mechanics Theory Manual (Neural Atlas)

Mathematical formulation of the neural-atlas contact framework: the SDF gap oracle, penalty and
augmented-Lagrangian normal forces, regularized Coulomb friction, topology-aware contact detection,
contact chart spawning, self-contact, and multi-body orchestration on the chart-based MPM solver.

Companion documents:
- `contact_atlas/03_mathematical_theory.md` — variational formulation & well-posedness;
  `contact_atlas/01_brainstorm.md`, `02_implementation_plan.md` — design rationale & phase plan.
- `docs/contact_verification_manual.md` — analytical benchmarks CV-1..CV-5 + the neural-chart
  verification protocol; `docs/hertz_derivation/` — the closed-form derivations.
- `docs/mpm_velocity_gradient_audit.md` — curved-chart MPM velocity-gradient correctness (the MPM
  time integrator underpinning the contact forces).

*(Extracted from the project theory manual §1. The legacy full manual, covering the archived
Nine-Circles fracture work, is preserved at `archive/docs/theory_manual.md`.)*

---

## 1. MPM Contact Mechanics

This section describes the contact framework built on top of the chart-based MPM solver. The entire framework operates in **physical space** — particle velocities, contact forces, and gaps are all in $\mathbb{R}^3_{\text{phys}}$, not in chart-local $\boldsymbol{\xi}$-space. Since the MPM grid velocity and gravity are already physical (see the MPM solver conventions in `solvers/mpm/`), the contact force scatters through the existing P2G channel with **no Jacobian pull-back**.

### 1.1 Gap Function from a Neural SDF

For two bodies $A$ and $B$ with neural SDFs $\phi_A, \phi_B : \mathbb{R}^3 \to \mathbb{R}$ (negative inside), the **gap function** of particle $p$ on body $B$ against the obstacle $A$ is simply the SDF value:

$$g_N^p = \phi_A(\mathbf{x}_p^{\text{phys}}),\qquad g_N^p < 0 \iff \text{particle}\ p\ \text{penetrates body}\ A.$$

The Eikonal loss used during SDF training enforces $\|\nabla \phi\| \approx 1$ away from the medial axis, so the **outward unit normal** of body $A$ at the query point is obtained via one autograd call on $\phi_A$:

$$\mathbf{n}_A(\mathbf{x}) = \frac{\nabla \phi_A(\mathbf{x})}{\|\nabla \phi_A(\mathbf{x})\|}.$$

In code (`solvers/contact/gap.py::evaluate_gap`):

1. `x = x_candidates.clone().detach().requires_grad_(True)`
2. `phi = sdf_net(x)`
3. `grad_phi = torch.autograd.grad(phi, x, grad_outputs=ones_like(phi))[0]`
4. `normal = grad_phi / grad_phi.norm(dim=1, keepdim=True).clamp(min=eps)`

The function is wrapped in `torch.enable_grad()` so it is safe to call inside a caller's `torch.no_grad()` block (an explicit fix from the debug audit).

### 1.2 Penalty Normal Force

The classical penalty formulation approximates the Signorini constraint $g_N \ge 0$ by

$$p_n = \epsilon_n\, \langle -g_N\rangle_+, \qquad \mathbf{f}_p^{\text{contact}} = p_n\, V_p\, \mathbf{n}_A,$$

where $\langle \cdot \rangle_+$ is the Macaulay bracket ($\max(0, \cdot)$), $V_p$ is the current particle volume, and $\mathbf{n}_A$ is the outward normal of the obstacle (§1.1). This is the formula implemented in `solvers/contact/penalty.py::compute_contact_force`.

**Scattering to the grid.** Because MPM stores $\mathbf{v}$ and gravity in physical space, the contact force in physical space is added to the grid node $I$ with the same shape-function scatter as gravity:

$$\mathbf{f}_I^{\text{ext}} \mathrel{+}= \sum_p \underbrace{\mathbf{f}_p^{\text{contact}}}_{\text{per-particle force}} \, N_I(\boldsymbol{\xi}_p).$$

Note the **absence of a mass factor** relative to gravity: gravity is an acceleration scaled by $m_p$, whereas $\mathbf{f}_p^{\text{contact}}$ is already a force (its $V_p$ factor absorbs the "force per unit volume" of the penalty pressure). This is implemented as a new block in `solvers/mpm/transfers.py::particle_to_grid`, immediately after the gravity scatter, guarded by an `Optional[torch.Tensor]` parameter so all pre-existing call sites continue to work unchanged.

**Contact-stable time step.** The penalty spring introduces a characteristic oscillation frequency $\omega \sim \sqrt{\epsilon_n / m_p}$, which enters the CFL bound for explicit MPM. The stable time step becomes

$$\Delta t_{\text{contact}} \le \operatorname{safety}\cdot \sqrt{m_{\min}/\epsilon_n},\qquad \text{safety}\approx 0.5,$$

implemented in `solvers/contact/penalty.py::contact_stable_dt`. The effective time step for a contact-enabled MPM run is $\min(\Delta t_{\text{CFL}}, \Delta t_{\text{contact}})$.

### 1.3 Augmented Lagrangian (Uzawa) Normal Force

Pure penalty suffers from a fundamental tradeoff: stiffer $\epsilon_n$ gives smaller residual penetration but tighter $\Delta t_{\text{contact}}$. The augmented Lagrangian method (Alart & Curnier 1991; Simo & Laursen 1992) resolves this by introducing a persistent Lagrange multiplier $\lambda$ that accumulates the required contact pressure across iterations / time steps:

$$\boxed{p_{\text{aug}}(g, \lambda) = \max\!\big(0,\; \lambda - \epsilon_n\, g\big),\qquad \lambda^{k+1} = \max\!\big(0,\; \lambda^k - \epsilon_n\, g(u^{k+1})\big).}$$

**Sign convention.** We use $g < 0$ for penetration, so $-\epsilon_n\, g > 0$ under penetration and the multiplier grows. When the constraint is satisfied ($g \ge 0$), the Macaulay bracket clamps both expressions to zero. At convergence, $\lambda$ equals the true Lagrange multiplier (the exact contact pressure) and the penetration vanishes.

**MPM specialisation.** In an explicit MPM time stepper there is no inner Newton iteration within a single step; the Uzawa update is instead applied once per time step, after the gap has been re-evaluated at the new particle positions post-G2P. The multiplier persists across time steps, building restraint when penetration is sustained and decaying when contact separates. The force per particle is

$$\mathbf{f}_p^{\text{AL}} = p_{\text{aug}}(g_p, \lambda_p)\, V_p\, \mathbf{n}_p,$$

implemented in `solvers/contact/augmented_lagrangian.py::AugmentedLagrangianContact.compute_force`.

**Empirical performance.** For the ball-drop benchmark at the same $\epsilon_n = 5\times 10^4$:

| Strategy | Final min $y$ | Final penetration |
|---|---|---|
| Pure penalty | $-5.79\times 10^{-2}$ m | **5.8 cm** |
| Augmented Lagrangian | $+9.47\times 10^{-2}$ m (in flight) | **0 cm** |

The AL ball is actually airborne at the end of the simulation — the accumulated multiplier delivered enough restraint to fully reverse the ball's momentum from a moderate $\epsilon_n$.

**Stable-shape requirement.** The persistent multiplier is indexed by position in the input gap tensor, so the caller must supply a gap of stable shape (typically the full per-particle gap with zeros where inactive). A shape change between calls loses the accumulated state; the implementation emits a `RuntimeWarning` in this case to make the misuse visible.

### 1.4 Regularized Coulomb Friction

The classical Coulomb law is non-smooth, non-associative, and discontinuous at zero tangential velocity (Wriggers 2006, §4.1). We use the regularized form

$$\mathbf{f}_T^p = -\mu\, \|\mathbf{f}_N^p\|\, \frac{\mathbf{v}_T^p}{\sqrt{\|\mathbf{v}_T^p\|^2 + \epsilon_T^2}},$$

where the tangential velocity is obtained by projecting out the normal component,

$$\mathbf{v}_T = \mathbf{v} - (\mathbf{v}\cdot\mathbf{n})\,\mathbf{n},$$

and $\epsilon_T$ is a small regularization parameter (velocity scale) that smooths the stick/slip transition. In the slip limit $\|\mathbf{v}_T\| \gg \epsilon_T$ the magnitude approaches the Coulomb cone bound $\mu\|\mathbf{f}_N\|$; in the stick limit $\|\mathbf{v}_T\| \ll \epsilon_T$ the force is linear in $\mathbf{v}_T$ and goes cleanly through zero. The friction force is stateless, purely a function of the current particle velocity and the normal force magnitude delivered by §1.2 or §1.3.

**Composability.** The implementation in `solvers/contact/friction.py::compute_friction_force` takes the normal-force magnitude as a scalar-per-particle input, so it composes with both the penalty force ($\|\mathbf{f}_N^p\| = \epsilon_n \langle -g\rangle_+ V_p$) and the augmented-Lagrangian force ($\|\mathbf{f}_N^p\| = p_{\text{aug}} V_p$). The total contact force is $\mathbf{f}_N + \mathbf{f}_T$, passed as a single tensor to `particle_to_grid(contact_force=...)`.

**Sliding-block verification.** For a slab of particles held in static equilibrium against a floor under gravity $g$ and launched with horizontal velocity $v_0$, the analytic deceleration is $a = \mu g$ (independent of slab mass or penalty stiffness). The benchmark `benchmarks/contact/sliding_block_mpm.py` measures:

$$a_{\text{measured}} = 2.9423\ \text{m/s}^2,\quad a_{\text{analytic}} = \mu g = 0.3\cdot 9.81 = 2.9430\ \text{m/s}^2,$$

a ratio of **1.000 to 4 decimal places**, and the frictionless control run keeps $v_x = 1.000$ exactly across 4000 steps.

### 1.5 Topology-Aware Contact via Persistent Homology

For two bodies $A$ and $B$, the **combined SDF** is the pointwise minimum

$$\phi_{AB}(\mathbf{x}) = \min\!\big(\phi_A(\mathbf{x}),\ \phi_B(\mathbf{x})\big).$$

The sublevel set $\{\mathbf{x}: \phi_{AB} < 0\}$ is the union of the two body interiors. Its persistent homology encodes the contact state directly through the Betti numbers of the sublevel filtration at $t \to 0^-$:

| Configuration | $\beta_0$ | $\beta_2$ |
|---|---|---|
| Bodies fully separated | 2 | 0 |
| Bodies in contact (any amount) | 1 | 0 |
| Body $B$ fully enclosed in $A$ | 1 | 1 |

So the topology of the combined SDF is a **discrete state variable** for contact. A decrease in $\beta_0$ across two consecutive updates is a **first-contact event**; an increase is a **separation event**; an increase in $\beta_2$ is an **enclosure event** (see the brainstorm doc §8.3 for the full event table including self-contact $H_1$ births).

**Implementation.** `solvers/contact/contact_topology.py::ContactTopologyMonitor` builds $\phi_{AB}$ on a cubical grid at each call to `update(load_step)`, runs the existing `atlas/topo/persistence.py::compute_persistence_diagrams` + `filter_by_lifetime` + `betti_numbers_at` pipeline, compares the Betti numbers against the cached state from the previous call, and emits `ContactTopologyEvent` objects for each transition. The event carries:

- `event_type ∈ {"first_contact", "separation", "enclosure"}`
- `load_step` at which the event was detected
- `beta0_before`, `beta0_after` (and `beta2_before`, `beta2_after` for enclosure)
- `location` — the argmin of $\phi_{AB}$ on the grid, a reasonable proxy for the deepest overlap point (`None` for separation events, which have no unique location)

**Robustness.** The detection uses `filter_by_lifetime` to drop persistence pairs whose lifetime is below a threshold proportional to the filtration range, protecting against spurious events from SDF numerical noise. The Bottleneck Stability Theorem (Cohen-Steiner, Edelsbrunner & Harer 2007) guarantees that small perturbations of the SDF produce only small perturbations of the persistence diagram, so the $\beta_0$ transitions are robust across time step and grid resolution.

**Integration with the MPM loop.** `SchwarzMPMSolver.observe_contact_topology(monitor, check_interval)` registers the monitor; `step()` then calls `monitor.update(step_count)` every `check_interval` steps and appends any emitted events to `solver._contact_topology_events`. The new step diagnostic key `"contact_topology_events"` returns the events emitted during that step (empty list if the monitor wasn't due to fire).

### 1.6 Contact Chart Spawning

Once a first-contact event has been detected at a physical-space location $\mathbf{x}_c$, we want to add two new charts straddling the contact interface — one on each body — so the atlas gains resolution at the contact region. The existing `atlas/topo/chart_spawn.py::SpawnedChartPair` data structure is reused verbatim; the bridge function `solvers/contact/contact_chart_spawn.py::spawn_contact_chart_pair` computes the missing pieces:

1. **Contact normal** via autograd on body $A$'s SDF at $\mathbf{x}_c$ (same pattern as §1.1):

$$\mathbf{n}_A = \frac{\nabla \phi_A(\mathbf{x}_c)}{\|\nabla \phi_A(\mathbf{x}_c)\|}.$$

2. **Seeds** symmetrically straddling the contact interface along $\mathbf{n}_A$:

$$\mathbf{s}_+ = \mathbf{x}_c + \tfrac{1}{2} r\, \mathbf{n}_A,\qquad \mathbf{s}_- = \mathbf{x}_c - \tfrac{1}{2} r\, \mathbf{n}_A,$$

where $r$ is the requested support radius.

3. **Orthonormal frames** at each seed via `ChartSpawner._frame_from_normal` (Gram–Schmidt with a stable reference vector).

4. **Parent chart** for warm-starting the decoder: nearest existing chart in Euclidean distance to $\mathbf{x}_c$.

The returned pair is tagged `edge_type="contact"` and `activation_step=event.load_step`, and can be passed directly into `SchwarzMPMSolver.add_charts([pair])` — the existing `atlas/topo` chart-spawning machinery handles the rest (decoder instantiation, neighbor graph rebuild, color-group recomputation).

### 1.7 Self-Contact

Self-contact is the hardest case because particles on their own body are **supposed** to be inside their own SDF ($\phi_A < 0$ everywhere in the interior). A naive "particle penetrates its own SDF" detector would fire on every bulk particle at every step. The `solvers/contact/self_contact.py::SelfContactManager` uses two exclusion filters to recover a physically meaningful detection:

**Filter 1 — Surface filter.** At construction time, snapshot the initial gap $g_0^p = \phi_A(\mathbf{X}_0^p)$ for every particle. Only particles with $|g_0^p| < \text{surface\_tol}$ — i.e., particles that started on the body surface — participate in self-contact checks. Deep interior particles are classified as *bulk* and are never flagged.

**Filter 2 — Initial-gap delta filter.** At detection time, compute the current gap $g^p = \phi_A(\mathbf{x}^p(t))$ and compare to the baseline:

$$\text{active}^p = \text{is\_surface}^p\ \land\ \big(g^p < g_0^p - \delta_{\text{pen}}\big).$$

A particle is flagged only when its current gap is *at least $\delta_{\text{pen}}$ more negative* than its initial gap — i.e., it has moved notably deeper than where it started. This catches folding (a surface particle that crosses into the body interior because the body folded onto itself) without false-positiving numerical wobble around the home position.

**Force.** The force on active particles uses the delta, not the absolute gap, so bulk-surface-adjacent particles that move slightly don't produce large spurious restoring forces:

$$\mathbf{f}_p^{\text{self}} = \epsilon_n\, \langle g_0^p - g^p\rangle_+\, V_p\, \mathbf{n}_p\quad\text{for active}\ p;\quad \mathbf{0}\ \text{otherwise}.$$

**Scope and limitations.** The method uses the **reference** SDF (not a deformed one), so it is accurate for small-to-moderate deformations. For large deformations the reference SDF is no longer a good model of the deformed surface and the detector degrades. Tangential self-sliding through the standard friction module (§1.4) still works once self-contact has been identified via the normal-direction detector. The folding-slab benchmark (`benchmarks/contact/folding_slab_mpm.py`) verifies that 36 surface particles are flagged mid-fold, 5 bulk particles are never active, and no false positives occur at rest.

### 1.8 Multi-Body Orchestration: SchwarzMPMSolver Contact Integration

The multi-chart `SchwarzMPMSolver` grows two hooks for contact:

1. **`configure_contact(opponent_bodies, epsilon_n, margin)`** — stores a list of `ContactBody` opponents, the penalty stiffness, and a broad-phase margin. The solver constructs a `ContactManager` internally.

2. **`_compute_contact_forces()`** — before each per-chart MPM step, iterates over all charts $i$ with nonzero particle counts and:

    a. Pushes particles to physical space: $\mathbf{x}_p = \varphi_i(\boldsymbol{\xi}_p) + \mathbf{u}_p$.

    b. For each opponent body, runs a broad-phase distance cull (vectorized across all particles and all opponent seeds at once): skip if the closest particle-to-seed distance exceeds the sum of seed-support radii plus the margin.

    c. Runs the narrow phase: `evaluate_gap(x_phys, body.sdf_net)` gives per-particle gaps and normals in one batched autograd call.

    d. Accumulates penalty forces across all opponent bodies: $\mathbf{f}_p^{\text{total}} = \sum_B \mathbf{f}_p^{\text{contact},B}$.

The resulting per-chart force tensors are passed into each `ChartMPMSolver.step(..., contact_force=f_total)`, which routes them through the P2G scatter block of §1.2.

The step loop orders: (1) compute contact forces, (2) step each chart with its forces, (3) exchange boundary velocities between overlapping charts, (4) transfer escaped particles, (5) tick the `ContactTopologyMonitor` if one is registered.

### 1.9 Verification Summary

| Item | Verification | Tolerance | Source |
|---|---|---|---|
| SDF normal accuracy vs analytic sphere | < 1° | 1° | `tests/test_contact_detection.py` |
| Penalty force $f_T\cdot v_T < 0$ (friction opposes velocity) | All particles | — | `tests/test_mpm_friction.py` |
| Friction slip-limit magnitude | $\|f_T\| \to \mu\|f_N\|$ as $\|v_T\|\to\infty$ | $10^{-6}$ | `tests/test_mpm_friction.py` |
| Analytic one-step friction: total $f_T$ vs $-\mu \sum \|f_N\|$ | relative error | $10^{-12}$ | `tests/test_mpm_friction.py` |
| Sliding block deceleration vs $\mu g$ | ratio | **0.025%** | `benchmarks/contact/sliding_block_mpm.py` |
| Ball drop: AL penetration vs penalty at same $\epsilon_n$ | AL < penalty | 20% improvement | `tests/test_mpm_contact.py::test_al_reduces_penetration_vs_penalty` |
| Two-sphere momentum conservation | $\|p_x\|$ | $2\times 10^{-15}$ | `benchmarks/contact/two_sphere_collision_mpm.py` |
| Two-sphere velocity symmetry | $\|v_A + v_B\|$ | $7\times 10^{-16}$ | `benchmarks/contact/two_sphere_collision_mpm.py` |
| P2G contact force conservation | $\sum_I f_I = \sum_p f_p$ | $10^{-6}$ | `tests/test_mpm_contact.py::test_total_force_conservation` |
| Topology: $\beta_0 = 2$ (separated) / $1$ (overlap) | integer equality | exact | `tests/test_contact_topology.py` |
| Topology sweep: one `first_contact`, one `separation` event | count equality | exact | `benchmarks/contact/contact_topology_demo.py` |
| Self-contact: bulk particles never flagged | over 11-step fold | exact | `benchmarks/contact/folding_slab_mpm.py` |
| AL shape-change warning | `RuntimeWarning` emitted | exact | `tests/test_mpm_contact.py::test_shape_change_warns_and_resets` |
| Broad-phase vectorized vs reference parity | over 5 random trials | exact | `tests/test_contact_detection.py::test_vectorized_broad_phase_matches_reference` |

Total active test suite: **120 passing, 7 skipped** (`pytest`).
