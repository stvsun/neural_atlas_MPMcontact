# Contact Mechanics Implementation Plan

## Overview

This document provides a concrete, phased implementation plan for adding contact mechanics to the neural atlas FEM/MPM framework. Each phase is self-contained and testable, building incrementally toward full contact capability.

> **Historical note.** This is the original design plan. The contact framework was ultimately built on
> the **MPM** path (see `docs/contact_theory_manual.md`); the FEM-Robin contact path sketched below
> references modules now under `archive/solvers/fem/` (e.g. `chart_vector_fem.py`, `schwarz_vector_fem.py`,
> `robin_schwarz.py`) and is kept here as design history, not as an active spec.

---

## Phase 1: SDF Gap Oracle and Contact Manager

**Goal:** Build the core contact detection infrastructure that all subsequent phases depend on.

### 1.1 `ContactPair` Data Structure

```
File: solvers/contact/contact_pair.py (new)
```

```python
@dataclass
class ContactPair:
    body_id_A: int                    # index of body A
    body_id_B: int                    # index of body B
    chart_id_A: int                   # chart on body A near contact
    chart_id_B: int                   # chart on body B near contact
    candidate_nodes_B: torch.Tensor   # (N_c, 3) physical coords of candidates on B
    gap: torch.Tensor                 # (N_c,) signed distance phi_A(x_B)
    normal: torch.Tensor              # (N_c, 3) contact normal grad(phi_A)/|grad(phi_A)|
    active: torch.Tensor              # (N_c,) bool — True if g < 0 (penetrating)
```

### 1.2 `ContactManager` Class

```
File: solvers/contact/contact_manager.py (new)
```

**Responsibilities:**
- Maintain list of bodies (each = atlas + SDF + solver)
- Broad-phase: cull chart pairs by seed distance
- Narrow-phase: evaluate SDF gap at boundary node candidates
- Compute contact forces (penalty, later augmented Lagrangian)
- Track active set across iterations

**Interface:**

```python
class ContactManager:
    def __init__(self, bodies: List[ContactBody], epsilon_n: float, mu_friction: float = 0.0):
        ...

    def detect(self) -> List[ContactPair]:
        """Broad-phase + narrow-phase contact detection."""
        ...

    def compute_penalty_forces(self, pairs: List[ContactPair]) -> Dict[int, torch.Tensor]:
        """Return {body_id: nodal_forces} for penalty contact."""
        ...

    def update_deformed_positions(self, body_id: int, u: torch.Tensor):
        """Update deformed boundary positions for body after a solve step."""
        ...
```

**`ContactBody` wrapper:**

```python
@dataclass
class ContactBody:
    body_id: int
    sdf_net: nn.Module               # neural SDF for this body
    decoders: List[ChartDecoder]     # chart decoders
    masks: List[MaskNet]             # chart masks
    boundary_nodes_phys: torch.Tensor  # (N_bnd, 3) physical boundary node coords
    boundary_node_areas: torch.Tensor  # (N_bnd,) tributary area per node
    current_displacement: torch.Tensor # (N_bnd, 3) current displacement at boundary
```

### 1.3 SDF Gap Evaluation

```
File: solvers/contact/gap.py (new)
```

```python
def evaluate_gap(x_candidates: torch.Tensor,  # (N, 3) physical coords
                 sdf_net: nn.Module,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate gap function and contact normal via neural SDF.

    Returns:
        gap: (N,) signed distance values
        normal: (N, 3) unit normals grad(phi)/|grad(phi)|
    """
    x = x_candidates.detach().requires_grad_(True)
    phi = sdf_net(x).squeeze(-1)
    grad_phi = torch.autograd.grad(
        phi, x, grad_outputs=torch.ones_like(phi),
        create_graph=False, retain_graph=False
    )[0]
    norm = grad_phi.norm(dim=1, keepdim=True).clamp(min=1e-12)
    normal = grad_phi / norm
    return phi.detach(), normal.detach()
```

### 1.4 Broad-Phase Culling

```python
def broad_phase(body_A: ContactBody, body_B: ContactBody,
                margin: float = 0.1) -> List[Tuple[int, int]]:
    """Return chart pairs (i_A, j_B) that could be in contact."""
    pairs = []
    for i, (seed_A, r_A) in enumerate(zip(body_A.seeds, body_A.radii)):
        for j, (seed_B, r_B) in enumerate(zip(body_B.seeds, body_B.radii)):
            if torch.norm(seed_A - seed_B) < r_A + r_B + margin:
                pairs.append((i, j))
    return pairs
```

### 1.5 Tests for Phase 1

```
File: tests/test_contact_detection.py (new)
```

- **test_gap_sphere:** Two unit spheres separated by distance d. Verify `gap = d - 2r` at closest points.
- **test_gap_penetration:** Overlapping spheres. Verify `gap < 0` at penetrating points.
- **test_normal_accuracy:** Compare SDF-derived normals against analytic sphere normals.
- **test_broad_phase_culling:** Verify distant chart pairs are culled.
- **test_broad_phase_contact:** Verify proximate chart pairs survive culling.

### 1.6 Verification

```bash
pytest tests/test_contact_detection.py -v
```

---

## Phase 2: Penalty Contact for FEM

**Goal:** Add penalty contact forces to `ChartVectorFEMSolver` and demonstrate two-body contact.

### 2.1 Penalty Force Computation

```
File: solvers/contact/penalty.py (new)
```

```python
def penalty_force(gap: torch.Tensor,       # (N,) signed distance
                  normal: torch.Tensor,     # (N, 3) contact normal
                  area: torch.Tensor,       # (N,) tributary area
                  epsilon_n: float,         # penalty stiffness
                  ) -> torch.Tensor:
    """Compute penalty contact nodal forces.

    Returns:
        f_contact: (N, 3) nodal forces (only nonzero where gap < 0)
    """
    penetration = torch.clamp(-gap, min=0.0)  # max(0, -g)
    pressure = epsilon_n * penetration         # p_n = eps * max(0, -g)
    return pressure.unsqueeze(1) * normal * area.unsqueeze(1)
```

### 2.2 Penalty Tangent Stiffness

```python
def penalty_tangent(gap: torch.Tensor, normal: torch.Tensor,
                    area: torch.Tensor, epsilon_n: float
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute penalty tangent contribution K_c = eps * (n x n) * A.

    Returns:
        active_mask: (N,) bool
        K_nn: (N, 3, 3) per-node tangent contribution (outer product n*n^T*eps*A)
    """
    active = gap < 0
    K_nn = epsilon_n * torch.einsum('ni,nj->nij', normal, normal) * area.unsqueeze(1).unsqueeze(2)
    return active, K_nn
```

### 2.3 Integration into ChartVectorFEMSolver

Modify `solvers/fem/chart_vector_fem.py`:

- Add `contact_forces` parameter to `solve_nonlinear()`:
  ```python
  def solve_nonlinear(self, ..., contact_forces=None):
      # In residual computation:
      if contact_forces is not None:
          residual[bnd_dofs] += contact_forces
  ```
- Add `contact_tangent` to tangent assembly for Newton convergence.

### 2.4 Integration into SchwarzVectorFEMSolver

Modify `solvers/fem/schwarz_vector_fem.py`:

- At each Schwarz iteration, call `ContactManager.detect()` and `compute_penalty_forces()`
- Pass contact forces to each chart's `solve_nonlinear()`
- Re-detect contact after each Schwarz sweep (active set may change)

### 2.5 Benchmark: Hertz Contact

```
File: benchmarks/contact/hertz_contact.py (new)
```

Two elastic spheres pressed together. Analytical Hertz solution for:
- Contact radius: `a = (3FR/(4E*))^{1/3}`
- Max pressure: `p_0 = (6FE*^2 / (pi^3 R^2))^{1/3}`
- Approach: `delta = a^2 / R`

Verify penalty solution converges to Hertz as `epsilon_n -> infinity`.

### 2.6 Tests for Phase 2

```
File: tests/test_penalty_contact.py (new)
```

- **test_penalty_force_zero_gap:** No force when gap >= 0.
- **test_penalty_force_penetration:** Correct force magnitude and direction.
- **test_hertz_convergence:** Contact radius and pressure converge to Hertz with mesh refinement.
- **test_penalty_stiffness_sensitivity:** Solution changes appropriately with epsilon_n.

---

## Phase 3: Penalty Contact for MPM

**Goal:** Add penalty contact body forces to the MPM P2G transfer.

### 3.1 MPM Contact Force Injection

Modify `solvers/mpm/transfers.py`:

```python
def particle_to_grid(particles, grid, gravity=None, contact_force=None):
    """
    contact_force: (N_particles, 3) or None
        Additional body force per particle from contact penalty.
        Applied same as gravity: f_I += m_p * a_contact * N_I(xi_p)
    """
    # ... existing P2G code ...

    if contact_force is not None:
        # Scatter contact force to grid (same as gravity scatter)
        for each particle p with nonzero contact_force:
            f_I += contact_force[p] * N_I(xi_p)
```

### 3.2 MPM Contact Detection

For MPM, contact detection uses deformed particle positions:

```python
def mpm_contact_detect(particles_B: MaterialPointCloud,
                       decoder_B: ChartDecoder,
                       sdf_A: nn.Module,
                       chart_params_B: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect contact for MPM particles.

    1. Push particles to physical space via decoder
    2. Evaluate phi_A at physical positions
    3. Return gap, normal, and contact force per particle
    """
    x_phys = particles_B.push_to_physical(decoder_B, **chart_params_B)
    gap, normal = evaluate_gap(x_phys, sdf_A)
    return gap, normal, x_phys
```

### 3.3 Force Pull-Back to Chart-Local Grid

Contact forces computed in physical space must be pulled back to chart-local coordinates for grid assembly:

```python
def pullback_contact_force(f_phys: torch.Tensor,   # (N, 3) force in physical space
                           jac: torch.Tensor,        # (N, 3, 3) chart Jacobian dx/dxi
                           ) -> torch.Tensor:
    """Pull contact force from physical to chart-local space.

    f_xi = J^{-T} f_phys (consistent with virtual work transformation)
    """
    jac_inv_T = torch.linalg.inv(jac).transpose(-1, -2)
    return torch.bmm(jac_inv_T, f_phys.unsqueeze(-1)).squeeze(-1)
```

### 3.4 Time Step Restriction

```python
def contact_stable_dt(epsilon_n: float, mass_min: float, h: float) -> float:
    """CFL-like condition for explicit MPM with penalty contact.

    dt <= alpha * sqrt(mass_min / epsilon_n)  where alpha ~ 0.5
    """
    return 0.5 * math.sqrt(mass_min / epsilon_n)
```

### 3.5 Benchmark: Ball Drop

```
File: benchmarks/contact/ball_drop_mpm.py (new)
```

Elastic ball dropped onto rigid floor (phi_floor = y - y_floor). Verify:
- Ball bounces (velocity reversal)
- Energy approximately conserved (within penalty regularization)
- No penetration beyond `g_max ~ F_max / epsilon_n`

### 3.6 Tests for Phase 3

```
File: tests/test_mpm_contact.py (new)
```

- **test_mpm_no_contact:** Distant particle has zero contact force.
- **test_mpm_penetration_force:** Penetrating particle gets repulsive force.
- **test_mpm_ball_bounce:** Ball reverses velocity on contact.
- **test_mpm_energy_conservation:** KE + PE + contact_work approximately conserved.

---

## Phase 4: Augmented Lagrangian (Uzawa)

**Goal:** Wrap penalty contact in Uzawa iteration for exact constraint enforcement.

### 4.1 Uzawa Update

```
File: solvers/contact/augmented_lagrangian.py (new)
```

```python
class AugmentedLagrangianContact:
    def __init__(self, epsilon_n: float, max_uzawa: int = 10, tol: float = 1e-6):
        self.epsilon_n = epsilon_n
        self.max_uzawa = max_uzawa
        self.tol = tol
        self.lam = None  # multiplier, initialized on first call

    def uzawa_step(self, gap: torch.Tensor) -> torch.Tensor:
        """Update multiplier: lambda_{k+1} = max(0, lambda_k + eps * g_k)"""
        if self.lam is None:
            self.lam = torch.zeros_like(gap)
        self.lam = torch.clamp(self.lam + self.epsilon_n * gap, min=0.0)
        return self.lam

    def augmented_pressure(self, gap: torch.Tensor) -> torch.Tensor:
        """Effective pressure: p = max(0, lambda + eps * (-g))"""
        return torch.clamp(self.lam - self.epsilon_n * gap, min=0.0)

    def converged(self, gap: torch.Tensor) -> bool:
        """Check complementarity: |lambda * g| < tol and |min(0, g)| < tol"""
        active = self.lam > 0
        penetration = torch.clamp(-gap[active], min=0.0)
        return penetration.max() < self.tol if active.any() else True
```

### 4.2 Schwarz + Uzawa Nesting

```python
# Outer loop: Uzawa
for k_uzawa in range(max_uzawa):
    # Inner loop: Schwarz with current multiplier
    for k_schwarz in range(max_schwarz):
        for color_group in color_groups:
            for chart_i in color_group:
                # Compute augmented contact forces using current lambda
                p = al_contact.augmented_pressure(gap)
                f_contact = p * normal * area
                solver_i.solve_nonlinear(..., contact_forces=f_contact)
        if schwarz_converged: break

    # Update multiplier
    pairs = contact_manager.detect()
    al_contact.uzawa_step(pairs.gap)
    if al_contact.converged(pairs.gap): break
```

### 4.3 Tests for Phase 4

- **test_uzawa_convergence:** Verify penetration decreases with Uzawa iterations.
- **test_uzawa_hertz:** Hertz benchmark with moderate epsilon_n converges to analytic solution.
- **test_uzawa_vs_penalty:** Uzawa gives smaller penetration than pure penalty with same epsilon_n.

---

## Phase 5: Robin Contact Transmission

**Goal:** Extend `RobinSchwarzSolver` to handle inter-body contact via inequality Robin conditions.

### 5.1 Contact Robin Interface

Modify `solvers/fem/robin_schwarz.py`:

```python
def _update_contact_interface(self, contact_pairs):
    """Update Robin interface data for contact.

    Standard Robin: lambda = (u_A + u_B)/2,  g = delta*(u_B - u_A)/2
    Contact Robin:  lambda = max(0, lambda + eps * gap)  (projected)
                    g = g + delta * (u_B - u_A) / 2
    """
    for pair in contact_pairs:
        # Normal component: inequality constraint
        gap_n = pair.gap
        self._lambda_contact[pair.chart_id] = torch.clamp(
            self._lambda_contact[pair.chart_id] + self.epsilon * gap_n, min=0.0
        )
        # Tangential component: equality or friction (Phase 6)
        ...
```

### 5.2 Tests for Phase 5

- **test_robin_contact_parallel:** Both bodies solve in parallel, converge to same solution as multiplicative.
- **test_robin_vs_multiplicative:** Compare contact pressure between Robin and multiplicative Schwarz.

---

## Phase 6: Coulomb Friction

**Goal:** Add tangential friction forces within the augmented Lagrangian framework.

### 6.1 Friction Force Computation

```
File: solvers/contact/friction.py (new)
```

```python
def coulomb_friction(v_tangential: torch.Tensor,  # (N, 3) tangential velocity
                     p_normal: torch.Tensor,       # (N,) normal contact pressure
                     mu: float,                     # friction coefficient
                     epsilon_t: float = 1e-6,      # regularization
                     ) -> torch.Tensor:
    """Regularized Coulomb friction traction.

    t = -mu * p_n * v_t / max(|v_t|, epsilon_t)
    """
    v_norm = v_tangential.norm(dim=1, keepdim=True).clamp(min=epsilon_t)
    return -mu * p_normal.unsqueeze(1) * v_tangential / v_norm
```

### 6.2 Tangential Decomposition

```python
def decompose_tangential(delta_u: torch.Tensor, normal: torch.Tensor):
    """Decompose displacement increment into normal and tangential."""
    u_n = torch.sum(delta_u * normal, dim=1, keepdim=True) * normal
    u_t = delta_u - u_n
    return u_n.squeeze(), u_t
```

### 6.3 Tests for Phase 6

- **test_friction_stick:** Low tangential force -> zero slip.
- **test_friction_slip:** High tangential force -> slip in correct direction.
- **test_friction_block_on_ramp:** Block on inclined plane: verify critical angle = arctan(mu).

---

## Phase 7: Topology-Aware Contact

**Goal:** Extend the topology monitor to detect contact events and spawn contact charts.

### 7.1 Combined SDF Monitoring

Modify `atlas/topo/monitor.py`:

```python
def monitor_contact(sdf_A, sdf_B, grid_points, threshold=0.02):
    """Monitor combined SDF for H0 merges (contact events).

    Combined field: phi_combined = min(phi_A, phi_B)
    H0 death in persistence diagram = two components touching
    """
    phi_A = sdf_A(grid_points)
    phi_B = sdf_B(grid_points)
    phi_combined = torch.minimum(phi_A, phi_B)
    # Run persistence on phi_combined grid
    # Compare with previous step's diagram
    # Fire TopologyEvent if new H0 death detected
```

### 7.2 Contact Chart Spawning

Modify `atlas/topo/chart_spawn.py`:

```python
def spawn_contact_charts(event: TopologyEvent, body_A, body_B):
    """Spawn chart pair at contact interface.

    1. Localize contact point from topology event
    2. Seed one chart from body A's nearest chart
    3. Seed one chart from body B's nearest chart
    4. Register with edge_type='contact'
    """
    center = event.localization
    normal = compute_sdf_normal(body_A.sdf, center)

    return SpawnedChartPair(
        seed_plus=center + 0.5 * radius * normal,
        seed_minus=center - 0.5 * radius * normal,
        frame_plus=frame_from_normal(normal),
        frame_minus=frame_from_normal(-normal),
        radius=estimate_contact_radius(body_A, body_B, center),
        parent_chart_plus=nearest_chart(body_A, center),
        parent_chart_minus=nearest_chart(body_B, center),
        edge_type='contact',
        activation_step=event.load_step,
    )
```

### 7.3 Tests for Phase 7

- **test_topology_contact_detection:** Two approaching spheres trigger H0 merge event.
- **test_contact_chart_spawn:** Spawned charts have correct seeds and frames.
- **test_separation_detection:** Separating bodies trigger H0 split event.

---

## Phase 8: Self-Contact

**Goal:** Detect and resolve contact within a single body.

### 8.1 Non-Neighbor Chart Proximity Detection

```python
def detect_self_contact(body: ContactBody, overlap_graph: List[List[int]],
                        threshold: float) -> List[ContactPair]:
    """Detect self-contact between non-neighboring charts.

    For each chart pair (i, j) where j not in neighbors[i]:
        Push boundary nodes to physical space
        Compute min distance between deformed boundaries
        If min_dist < threshold: create ContactPair
    """
```

### 8.2 Modified SDF for Self-Contact

The body's own SDF gives `phi(x) = 0` at its surface, so it cannot detect self-proximity. Instead:
- Use **chart-specific local SDFs** from `atlas/sdf/train_sdf_chartwise.py`
- Chart i's local SDF `phi_i` gives distance to the surface patch covered by chart i
- Self-contact gap for chart j boundary node: `g = phi_i(x_j)` where i != j and i not in neighbors[j]

### 8.3 Tests for Phase 8

- **test_self_contact_folding:** Beam folded onto itself detects self-contact.
- **test_self_contact_non_neighbor:** Only non-neighboring chart pairs checked.

---

## Directory Structure (Final)

```
solvers/contact/                  # New contact module
    __init__.py
    contact_manager.py            # ContactManager, ContactBody
    contact_pair.py               # ContactPair dataclass
    gap.py                        # SDF gap evaluation
    penalty.py                    # Penalty forces and tangent
    augmented_lagrangian.py       # Uzawa iteration
    friction.py                   # Coulomb friction
    self_contact.py               # Self-contact detection

benchmarks/contact/               # Contact benchmarks
    hertz_contact.py              # Hertz sphere-sphere
    ball_drop_mpm.py              # Ball drop with MPM
    block_on_ramp.py              # Friction benchmark

tests/
    test_contact_detection.py     # Phase 1 tests
    test_penalty_contact.py       # Phase 2 tests
    test_mpm_contact.py           # Phase 3 tests
    test_augmented_lagrangian.py  # Phase 4 tests
    test_friction.py              # Phase 6 tests
    test_topology_contact.py      # Phase 7 tests
    test_self_contact.py          # Phase 8 tests
```

---

## Dependency Graph

```
Phase 1: Contact Detection (gap.py, contact_manager.py)
    |
    +---> Phase 2: FEM Penalty Contact (penalty.py + chart_vector_fem.py mods)
    |         |
    |         +---> Phase 4: Augmented Lagrangian (augmented_lagrangian.py)
    |         |         |
    |         |         +---> Phase 5: Robin Contact (robin_schwarz.py mods)
    |         |
    |         +---> Phase 6: Friction (friction.py)
    |
    +---> Phase 3: MPM Penalty Contact (transfers.py mods)
    |
    +---> Phase 7: Topology-Aware Contact (monitor.py, chart_spawn.py mods)
    |
    +---> Phase 8: Self-Contact (self_contact.py)
```

---

## Key Modified Existing Files

| File | Modification | Phase |
|---|---|---|
| `solvers/fem/chart_vector_fem.py` | Add `contact_forces` param to `solve_nonlinear` | 2 |
| `solvers/fem/schwarz_vector_fem.py` | Inject contact force loop into Schwarz iteration | 2 |
| `solvers/fem/robin_schwarz.py` | Add inequality Robin interface conditions | 5 |
| `solvers/mpm/transfers.py` | Add `contact_force` param to `particle_to_grid` | 3 |
| `solvers/mpm/schwarz_mpm.py` | Inject contact detection into multi-chart MPM | 3 |
| `atlas/topo/monitor.py` | Add `monitor_contact` for combined SDF H0 tracking | 7 |
| `atlas/topo/chart_spawn.py` | Add `edge_type='contact'` and contact chart spawning | 7 |

---

## Verification & Validation Strategy

| Phase | V&V Method | Acceptance Criterion |
|---|---|---|
| 1 | Unit tests: gap accuracy vs analytic SDF | Gap error < 1e-3 for trained SDF |
| 2 | Hertz contact benchmark | Contact radius within 5% of analytic |
| 3 | Ball drop: energy conservation | Energy drift < 5% over 100 steps |
| 4 | Uzawa convergence | Penetration < 1e-6 after 10 Uzawa steps |
| 5 | Robin vs multiplicative comparison | Pressure difference < 1% |
| 6 | Block on ramp: critical angle | Within 2 degrees of arctan(mu) |
| 7 | Topology event detection timing | Event fires within 5 load steps of true contact |
| 8 | Self-contact folding beam | No self-penetration > 1e-4 |
