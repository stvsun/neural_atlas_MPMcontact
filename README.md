# Neural Atlas for Contact Mechanics

A meshfree framework for **contact mechanics on complex 3D geometries** using learned
**coordinate charts** and **neural signed-distance functions**, with a chart-based Material
Point Method (MPM), persistent-homology-based contact detection, and an analytical
verification suite that doubles as the acceptance test for the neural charts.

> **Status.** The contact framework (penalty / augmented-Lagrangian / friction / topology /
> self-contact) and a closed-form **analytical verification suite (CV-1..CV-5)** are in place.
> Training the neural coordinate charts is the next step — the analytical benchmarks are written
> to verify those charts when they land (see [§10 of the verification manual](docs/contact_verification_manual.md)).
>
> The earlier **Nine-Circles brittle-fracture** work has been **archived** under
> [`archive/`](archive/) (code, tests, docs, figures) — preserved for history, not maintained.
> The repository is now focused solely on neural-atlas contact mechanics.

---

## Contact framework

A complete meshfree contact stack for the chart-based MPM, verified against analytic results or
machine precision on every path.

| Phase | Module | What it adds | Verification |
|---|---|---|---|
| 1 | `solvers/contact/gap.py`, `contact_pair.py`, `contact_manager.py` | SDF gap oracle (forward pass + autograd normal) + `ContactBody`/`ContactPair` + broad/narrow phase | Normal accuracy vs analytic sphere $< 1°$ |
| 2 | `solvers/contact/penalty.py` | Penalty force $f=\epsilon_n\langle -g\rangle_+ V_p\mathbf n$ + contact-stable $dt$ | P2G scatter conserves total contact force to machine precision |
| 3 | `solvers/mpm/schwarz_mpm.py::configure_contact` | Multi-body orchestration; vectorized broad-phase + per-chart force | Two-sphere collision: momentum conserved to $2\times10^{-15}$ |
| 4 | `solvers/contact/augmented_lagrangian.py` | Persistent Uzawa multiplier; exact enforcement at moderate $\epsilon_n$ | Ball-drop at $\epsilon_n=5\times10^4$: penalty 5.8 cm penetration, **AL 0 cm** |
| 6 | `solvers/contact/friction.py` | Regularized Coulomb $\mathbf f_T=-\mu\|\mathbf f_N\|\,\mathbf v_T/\sqrt{\|\mathbf v_T\|^2+\epsilon_T^2}$ | Sliding-block: measured $2.9423$ vs analytic $\mu g=2.9430$ m/s² (ratio 1.000) |
| 7 | `solvers/contact/contact_topology.py` | Persistent-homology events on combined SDF $\phi_{AB}=\min(\phi_A,\phi_B)$ | Two-sphere sweep: exactly one `first_contact`, one `separation` |
| 7b | `solvers/contact/contact_chart_spawn.py` | `spawn_contact_chart_pair()` bridges events to `add_charts()` | SDF normal → frame → spawn pair → solver grows 1→3 charts |
| 8 | `solvers/contact/self_contact.py` | Surface-filter + initial-gap-delta heuristic for folding | Folding-slab: 36 surface particles flagged, 5 bulk never active |

*(Phase 5 — an FEM-Robin contact-transmission path sketched in `contact_atlas/02_implementation_plan.md` — was not pursued; the framework is MPM-based.)*

**Key design decisions**
- **No Jacobian pull-back for contact forces** — MPM grid velocity/gravity are in physical space; contact forces scatter the same way (`f_I += f_p N_I(ξ_p)`, no $J^{-T}$). See `docs/contact_theory_manual.md §1.2`.
- **Penalty and AL share one force API** — both return per-particle force for `particle_to_grid(..., contact_force=...)`; swap strategies without touching the solver.
- **Friction is stateless and composable** — `compute_friction_force(v, n, ‖f_N‖, μ, ε_T)` plugs onto any normal-force scheme.
- **Topology monitor uses the combined-SDF $H_0$** — `ContactTopologyMonitor` reuses the `atlas/topo` persistent-homology pipeline for first-contact/separation/enclosure events.

| Benchmark | Demonstrates | File |
|---|---|---|
| Ball drop (penalty) | Bounded penetration & rebound | `benchmarks/contact/ball_drop_mpm.py` |
| Two-sphere collision | Symmetric multi-body momentum conservation | `benchmarks/contact/two_sphere_collision_mpm.py` |
| Ball drop (AL vs penalty) | AL reduces residual penetration 5.8 cm → 0 cm | `benchmarks/contact/ball_drop_al_mpm.py` |
| Sliding block + friction | Coulomb deceleration matches $\mu g$ to 0.025% | `benchmarks/contact/sliding_block_mpm.py` |
| Topology event sweep | $\beta_0$ transition detection | `benchmarks/contact/contact_topology_demo.py` |
| Folding slab (self-contact) | Folding without bulk false positives | `benchmarks/contact/folding_slab_mpm.py` |
| Superformula cam-drive | Nonconvex rigid-body contact (CV-5) | `benchmarks/contact/supershape_cam_drive.py` |

---

## Contact detection & computation via transition maps

Contacts are read from the **chart transition map** rather than from a level set alone. Each body
$X$ carries boundary charts $\varphi_X:\theta\mapsto x$ (analytic today; trained `ChartDecoder`s
once the neural charts land). The contact correspondence between two bodies $A,B$ is the
**boundary-to-boundary transition map**

$$\tau_{AB}:\ \theta_A\ \longmapsto\ \psi_B=\big(\varphi_B^{-1}\!\circ\varphi_A\big)(\theta_A),$$

i.e. take a surface point on $A$, $\varphi_A(\theta_A)$, into physical space and **invert $B$'s
chart** to find the matching surface parameter on $B$ (Newton inverse `common/geometry.py::invert_decoder`).

**Detect.** Sample one body's boundary chart and evaluate the partner's *inverse-chart gap*

$$g_B(p)=\lVert p-c_B\rVert-\rho_B(\psi),\qquad \psi=\angle\,Q_B^{-1}(p-c_B),$$

which for a star-shaped body is single-valued and smooth everywhere; $g_B<0 \Rightarrow$ penetration.
Scanning the chart parameter enumerates **every disjoint contact arc** — essential for nonconvex
shapes, where a single closest-point projection returns only one foot. (CV-5 shows this head-to-head:
the chart scan finds $\ge 2$ arcs while a single CPP reports 1.)

**Compute.** The contact normal comes from the **chart Jacobian**, not $\nabla\phi$:

$$\mathbf n=\frac{t_1\times t_2}{\lVert t_1\times t_2\rVert},\qquad t_\alpha=\frac{\partial\varphi}{\partial\xi^\alpha},$$

exact even at high-curvature lobe tips. The force is then the standard penalty / augmented-Lagrangian
normal force plus regularized Coulomb friction, scattered through the MPM P2G channel:
$\mathbf f=\epsilon_n\langle -g\rangle_+\,\mathbf n\ (+\ \text{friction})$.

**Why it helps (vs the SDF closest-point projection).** The transition map is single-valued and
analytic in concavities, where the Euclidean SDF closest-point projection (Liu & Sun 2020, Eq. 22)
becomes multivalued across the medial axis and the Eikonal normal degrades. **Honest caveats:** the
inverse radial gap is *not* the Euclidean perpendicular distance (biased $\sim 1/\cos\alpha$ on steep
flanks — a bounded 1-D chart refine removes it); and the production oracle `solvers/contact/gap.py`
**currently uses the neural-SDF gradient** — the transition-map chart oracle is the analytical
formulation exercised by CV-1..CV-5 and the target once neural charts replace the SDFs.

Full treatment: `docs/contact_theory_manual.md` (algorithms), `docs/contact_verification_manual.md`
§2 (kinematics) and §10 (neural-chart verification protocol), `solvers/contact/supershape.py` (CV-5).

---

## Analytical verification (CV-1..CV-5)

Closed-form contact solutions, derived in SymPy and adversarially cross-checked against Johnson's
*Contact Mechanics* and Timoshenko–Goodier, recast in the neural-atlas / transition-map framing.
They are the **acceptance targets for the neural charts**.

| Benchmark | Problem | Closed-form targets |
|---|---|---|
| **CV-1** | Hertz normal contact (3D + 2D) | $a,p_0,\delta$, force–approach, subsurface yield |
| **CV-2** | Cattaneo–Mindlin friction | stick radius $c/a$, tangential traction |
| **CV-3** | Brazilian disc | $\sigma_t=2P/\pi Dt$, stress field, diametral compliance |
| **CV-4** | Nine-disc packing | equibiaxial center, force–compression law |
| **CV-5** | Nonconvex superformula contact | multi-arc detection, chart-gap vs SDF (the transition-map test) |

See **[docs/contact_verification_manual.md](docs/contact_verification_manual.md)** for the pass
criteria, embedded figures, and the **two-level neural-chart verification protocol (§10)**. Symbolic
derivations + numpy evaluators: **[docs/hertz_derivation/](docs/hertz_derivation/README.md)** and
`postprocessing/contact_fields.py`.

---

## Quick start

```bash
pip install -e .

# Run the active test suite (contact + core MPM)
pytest -q                                   # 120 passed, 7 skipped

# Analytical CV references (self-checking, no solver needed)
python3 postprocessing/contact_fields.py            # numpy evaluators self-test
python3 docs/hertz_derivation/hertz_transition_map.py
python3 docs/hertz_derivation/brazilian_disc_atlas.py
python3 docs/hertz_derivation/nine_disc_atlas.py

# Contact benchmarks
python3 benchmarks/contact/ball_drop_al_mpm.py
python3 benchmarks/contact/sliding_block_mpm.py
python3 benchmarks/contact/two_sphere_collision_mpm.py
python3 benchmarks/contact/supershape_cam_drive.py          # + --free-A control

# Regenerate the verification figures
python3 postprocessing/plot_liusun_all.py
python3 postprocessing/plot_supershape_demo.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Contact Theory Manual](docs/contact_theory_manual.md) | Algorithms: SDF gap oracle, penalty, augmented Lagrangian, regularized Coulomb friction, topology-aware detection, contact chart spawning, self-contact, multi-body orchestration |
| [Contact Verification Manual](docs/contact_verification_manual.md) | CV-1..CV-5 closed-form benchmarks (embedded figures) + the **neural coordinate-chart verification protocol (§10)** |
| [Analytical derivations index](docs/hertz_derivation/README.md) | SymPy derivations, numpy evaluators, plotting, and how it all maps to CV-1..CV-5 |
| [MPM Velocity-Gradient Audit](docs/mpm_velocity_gradient_audit.md) | Curved-chart MPM velocity-gradient correctness (the integrator underpinning contact) |
| [Design docs](contact_atlas/) | Brainstorm, implementation plan, variational theory & well-posedness |

---

## Repository structure

```
neural_atlas_MPMcontact/
├── atlas/            # SDF training, chart construction, persistent-homology topology
├── common/           # ChartDecoder / MaskNet / MLP, geometry (Jacobians, invert_decoder), Schwarz utils
├── solvers/
│   ├── mpm/          # chart-based MPM (particles, grid, transfers, constitutive, schwarz_mpm)
│   └── contact/      # gap, penalty, augmented_lagrangian, friction, contact_topology,
│                     #   contact_chart_spawn, self_contact, contact_manager, supershape
├── benchmarks/
│   ├── contact/      # ball-drop, two-sphere, sliding-block, folding-slab, topology, supershape cam-drive
│   └── mpm_basic/    # (placeholder for MPM core benchmarks)
├── postprocessing/   # contact_fields (numpy refs), pyvista_field2d, plot_liusun_*, plot_supershape_demo, utils
├── docs/             # contact_theory_manual, contact_verification_manual, hertz_derivation/, mpm audit
├── contact_atlas/    # design docs (brainstorm, implementation plan, math theory)
├── tests/            # contact + core-MPM tests (test_neural_chart_verification.py = neural-chart harness)
├── figures/          # contact figures (embedded in the verification manual)
└── archive/          # legacy Nine-Circles fracture work — preserved, not maintained
```

---

## Next steps (neural coordinate charts)

The framework currently uses analytical charts/SDFs. To bring up the neural charts:
1. Train a neural SDF / `ChartDecoder` on each analytical shape (`atlas/sdf/train_sdf.py`, `atlas/charts/train_atlas.py`).
2. Verify it against the closed forms with the **two-level protocol** (L0 geometry, L1 mechanics) in
   `docs/contact_verification_manual.md §10`; the harness `tests/test_neural_chart_verification.py` is wired and waiting.

---

## References

- K. L. Johnson (1985), *Contact Mechanics*, Cambridge Univ. Press.
- Timoshenko & Goodier, *Theory of Elasticity*; Mindlin (1949); Hondros (1959).
- C. Liu & W. Sun (2020), "ILS-MPM," *CMAME* 369:113168.
- Alart & Curnier (1991); Simo & Laursen (1992); Wriggers (2006) — contact algorithms.
- Cohen-Steiner, Edelsbrunner & Harer (2007) — persistence-diagram stability.
