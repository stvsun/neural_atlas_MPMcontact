# 3D Mixed-Mode Cyclic Rock-Joint Shear — implementation brief (from research workflow wk6ah32tc)

All signatures confirmed against the actual codebase. Here is the implementation brief.

---

# IMPLEMENTATION BRIEF — 3D Mixed-Mode Cyclic Rock-Joint Shear Capstone

Branch: extend `numerical-cv-suite` (do NOT touch `main`). All paths absolute under `/Users/wsun/Documents/Software/neural_atlas_MPMcontact/`. Run Python via `/Users/wsun/opt/anaconda3/bin/python` for any PyVista work.

---

## 1. DEFINITIONS (chosen convention — binding for code, docs, and figure axes)

**ADOPT READING (A): one normal mode + TWO in-plane tangential shear modes.** Define the joint-local orthonormal triad once, up front, in `docs/contact_verification_manual.md` (new §11.10) and in every new module's docstring:

- **e_z = joint normal** — outward normal of the footwall (master) mid-surface, `n = (-h_x, -h_y, 1)/sqrt(1+|∇h|²)`. The **normal mode** is `(σ_n, u_n)`: opening/closing, dilation. Sign convention: **gap g_N > 0 = separation, g_N < 0 = penetration** (matches `evaluate_gap`/`vertical_gap` in this codebase); contact pressure `−t_N > 0`. State this is compression-positive for σ_n. KKT: `g_N ≥ 0, σ_n ≥ 0, g_N·σ_n = 0`.
- **e_x = in-plane primary shear** — sliding along x in the joint plane → **in-plane shear** = `(τ_x, u_tx)`. Mode-II-like in the x–z observation plane.
- **e_y = e_z × e_x = in-plane transverse shear** — sliding along y, still in the joint plane → **out-of-plane shear** = `(τ_y, u_ty)`. Mode-III-like ("out of" the x–z *observation* plane, NOT out of the joint plane).
- **mixed-mode = mixed-mode (in-plane azimuth)** — the PRIMARY sense for this study: both shear components active at azimuth θ in the joint plane, `u_t = (Δ·cosθ, Δ·sinθ)`, σ_n held by the protocol. The traction is NOT separable into independent x/y responses (Grasselli directional roughness). Report `τ–Δ` and `σ_n–u_n` curves per θ.

**Justification:** every 3D interface-element formulation (DIANA: `t_nz, t_sx, t_sy` conjugate to `δu_nz, δu_sx, δu_sy`) hard-wires this triad; Grasselli/Barton directional roughness is defined over in-plane azimuth; it keeps "normal mode" (dilation/opening) cleanly its own third mode.

**Alternative noted (do NOT use as default):** the fracture-mechanics/CZM sense **mixed-mode (normal–shear, Mode I–II)** = simultaneous opening + sliding. We expose this *only* in the optional cohesive branch (Phase D), where it is named explicitly as "mixed-mode (Mode I–II/III)" and uses the effective separation `λ = sqrt(⟨g_N⟩² + g_tx² + g_ty²)`. Flag in the manual that the existing manual already uses "in-plane" in the plane-stress sense (`docs/contact_verification_manual.md` ~lines 150, 426–431) — the new joint-shear usage must be disambiguated there to avoid collision.

---

## 2. MODULE PLAN (new files; what each contains; what it reuses)

### (a) 3D height chart `h(x,y)` — `solvers/contact/profile_height_chart_3d.py`
Lift `solvers/contact/profile_chart_2d.py` (`NeuralHeight1D`) to 2D.
- `class NeuralHeight2D(torch.nn.Module)`: `__init__(x_lo, x_hi, y_lo, y_hi, f_max=1500.0, n_freq=96, width=128, depth=4, base=0.0, plain=False)`. Fourier features `_features(xy)` → `[cos/sin(2πB·x̃), cos/sin(2πB·ỹ)]` with `B=geomspace(0.5,f_max,n_freq)` (4·n_freq features; keep n_freq modest in 2D to bound width). `forward(xy:(N,2)) -> h:(N,)`.
- `height_and_grad_2d(xy, height) -> (h:(N,), grad_h:(N,2))` — `torch.enable_grad()` block, autograd `(h_x, h_y)`. Mirrors `height_and_grad`.
- `surface_normal_3d(grad_h) -> n:(N,3)` = `(-h_x,-h_y,1)/sqrt(1+|∇h|²)`.
- `fit_height_chart_2d(xy_data, z_data, f_max, n_freq, width, depth, iters, ...) -> (chart, rmse)` — copy training loop from `fit_height_chart` (Adam + cosine anneal, MSE on random batches), flattened 2D grid.
- `class AnalyticPyramid3D`: 4-sided pyramid `h(x,y)=A·max(0,1−|x|/L−|y|/L)` with closed-form `(h_x,h_y)` for the L0/L1 verification anchor.

### (b) 3D rigid surface–surface contact + three loading-mode drivers — `benchmarks/contact/cv_numerical/rock_joint_shear_3d.py`
Generalize `benchmarks/contact/cv_numerical/rock_joint_shear.py` (rigid hangingwall, fixed footwall). Numpy core, mirrors the 2D node-to-surface law.
- `bake_height_3d(height_module, x_grid, y_grid) -> dict(x, y, h, hx, hy)` — one autograd pass, store `h` and both gradient components on the 2D grid (no autograd in inner loop). Generalizes `bake_height`.
- `_interp_2d(grid, X, Y) -> (h, hx, hy, valid)` — bilinear interp (`scipy.interpolate.RectBivariateSpline`). Generalizes `_interp`.
- `contact_forces_3d(zU, u_xy, lower, upper, eps_n, mu, eps_t) -> (Fx, Fy, Fz, diag)` — for each upper node at world `(X=ξ+u_x, Y=η+u_y, Z=zU+h_U)`: `n` from footwall via `_interp_2d`; `g_N = (Z−h_L(X,Y))·n_z`; `f_n = eps_n·⟨−g_N⟩·dξ·dη` (2D tributary area); regularized Coulomb in the tangent plane `f_t = −μ·f_n·v_t/sqrt(|v_t|²+ε_t²)` using `friction.py`'s law with the in-plane slip direction. Generalizes `contact_forces`.
- `solve_z_equilibrium(u_xy, lower, upper, eps_n, mu, W, z_bracket) -> zU` — bisection on the vertical DOF such that `F_z = W` (monotone, same as 2D `solve_y_equilibrium`).
- **THREE mode drivers** (one shared loop, slip vector differs):
  - `run_shear_inplane(...)` — `u=(Δ,0)`; in-plane shear.
  - `run_shear_outofplane(...)` — `u=(0,Δ)`; out-of-plane shear.
  - `run_shear_mixed(..., theta)` — `u=(Δcosθ, Δsinθ)`; mixed-mode (azimuth).
  All wrap a common `_run_shear_3d(lower, upper, slip_fn, sigma_n, mu, shear_total, n_inc, eps_n, protocol)` recording `history={ux, uy, dilation_z, tau_x, tau_y, sigma_n, mu_app, n_active, pen_max}`.
- **Protocol switch** `protocol ∈ {"CNL","CNS","CNV"}`: CNL holds `W=σ_n·A` fixed (solve zU); CNS replaces with `σ_n = σ_n0 + k_n·u_n` (target W updates each increment, `dσ_n = k_n·du_n`); CNV pins zU (k_n→∞). CNL is `k_n=0`.

### (c) Deformable chart-FEM mixed-mode CYCLIC driver — `benchmarks/contact/cv_numerical/rock_joint_cyclic_fem.py`
Couple two `ChartVectorFEMSolver` blocks (`solvers/fem/chart_vector_fem.py`) across the rough interface; the constitutive law is layered and composable.
- `solve_contact_pair(block1, block2, stress_fn, tangent_fn, f_ext1, f_ext2, u_bc1, bc_mask1, u_bc2, bc_mask2, interface_law, k_pen, max_iter, relax) -> (u1, u2, diag)` — coupled Newton on the concatenated `3(N1+N2)` DOF system. Each iteration: `internal_forces` + `tangent_stiffness` from both blocks (block-diagonal), plus interface force/stiffness in the off-diagonal blocks; BC and line-search exactly as `solve_nonlinear` (lines ~806–836) and `cv1_hertz_fem.py` Newton loop (lines 127–152).
- `interface_law` is a struct of composable pieces (each separately testable):
  1. **Unilateral normal** — penalty (`solvers/contact/penalty.py::compute_contact_force`) or AL (`augmented_lagrangian.py`); `K_c += eps_n·dA·(n⊗n)`.
  2. **Dilatant-frictional tangential** — extend `friction.py` regularized Coulomb with a Patton/Barton–Bandis term: mobilized angle `φ_b + i(σ_n, D)`, dilation flow `ġ_N = tan(i)·|ġ_T|`.
  3. **Plesha asperity degradation** — ODE `i = i_0·exp(−c·W_p)`, `W_p = ∫ τ·dδ_slip` accumulated frictional work; tracked per interface point, per in-plane direction (directional damage).
- `state_dataclass JointState`: per-interface-point `g_N, g_tx, g_ty, t_N, t_x, t_y, D, i, W_p, sigma_n` (the minimal record to reconstruct hysteresis + energy offline).
- `run_cyclic(block1, block2, n_cycles, amplitude, azimuth, protocol, k_n, sigma_n0, ...) -> history` — forward/reverse displacement-controlled loop; emits per-increment energy ledger (external work, elastic, frictional dissipation, damage dissipation) and per-cycle `(τ_peak(N), i(N), loop_area)`.
- **Material:** `make_linear_elastic_small_strain(E, nu)` (`solvers/fem/linear_elastic.py`) default; `make_neo_hookean(mu, K)` for finite-strain checks. Identity chart (`chart_decoder=None`) for the blocks initially; the rough geometry lives in the *interface* height chart, not the block decoder.
- **(D, optional) cohesive branch** — `solvers/contact/cohesive_mixed_mode.py`: Turon–Camanho bilinear single-damage law, effective `λ`, Benzeggagh–Kenane `G_c = G_Ic + (G_IIc−G_Ic)·B^η`, shared penalty stiffness; Roe–Siegmund cyclic damage with endurance threshold. Energy-consistent handoff cohesion→friction. Build only after the frictional branch passes its ladder.

### (d) Analysis + verification — `tests/test_rock_joint_shear_3d.py`, `tests/test_rock_joint_cyclic_fem.py`
Mirror `tests/test_chart_fem.py` patterns (solver construction, error norms, BC masks). Contains the full verification ladder of §3. Slow cyclic sweeps run as benchmark drivers, not routine pytest (follow the existing CV-1/CV-2/CV-5 convention).

### (e) Data storage — `postprocessing/joint_data_io.py`
- `save_joint_history(path, history, params, meta)` — writes `runs/rock_joint_3d/<name>/history.npz` (gitignored, parallel to `runs/rock_joint_capstone/`). Arrays: `ux, uy, u_n, t_x, t_y, t_N, sigma_n, D, i, W_p, dissipation, loop_area`, indexed `[increment]` and `[interface_point]` where spatial. Plus a sidecar `params.json` (E, ν, μ, φ_b, i_0, k_n, eps_n, protocol, azimuth, n_cycles, surface tag). Self-describing so loops/energy reconstruct offline.
- `load_joint_history(path) -> dict`.

### (f) PyVista viz — `postprocessing/surface_anim_3d.py`
NEW module (no 3D-surface PyVista exists; `pyvista_field2d.py` is 2D-only). Use the verified env recipe.
- `make_height_surface(X, Y, Z, scalars, name)` — `pv.StructuredGrid`; **always `ravel(order='F')`** with `meshgrid(indexing='ij')`.
- `update_surface(g, Z, scalars, name)` — mutate `.points[:,2]` and re-set `g[name]` by same key; reassign `g.points` to trigger `Modified()`.
- `new_plotter(...)` — `off_screen=True`, `set_background('white')`, `enable_anti_aliasing('fxaa')`, `enable_depth_peeling(8, 0.0)`.
- `add_surface(...)` — **`cmap=matplotlib.cm.get_cmap(name)` OBJECT, never a string** (matplotlib 3.4.1 gate); fixed `clim` to avoid flicker; `scalar_bar_args` dict; diverging `coolwarm` + symmetric clim for signed gap/dilation.
- `animate(frames_fn, n_frames, out, fps)` — **GIF only by default** (`open_gif(out, fps=15, palettesize=128, subrectangles=True)`); mp4 path documented but requires `pip install imageio-ffmpeg` first. Always `pl.close()`. Two-surface exploded view: translate upper grid `+z` to open the joint aperture.
- Hysteresis loops stay in **matplotlib** (`plot_rock_joint_3d.py`): `τ–Δ` loops shaded by `np.trapz` enclosed area, colored by cycle index; `τ_peak(N)`, `i(N)` decay; directional strength rose `τ_app(θ)`.

---

## 3. VERIFICATION LADDER (concrete acceptance checks, in build order)

1. **L0 — chart slope reconstruction.** Fit `NeuralHeight2D` to `AnalyticPyramid3D` samples; assert `max|∇h_recon − ∇h_exact| < 1e-3` and height RMSE < 1% of RMS. (mirrors 2D L0.)
2. **Flat-joint Coulomb limit.** Zero-roughness planar interface, CNL, shear along x: assert `τ/σ_n → tan(φ)` at slip, **zero dilation**, and `τ_y ≈ 0` (no spurious transverse force).
3. **Frame-indifference / isotropy of a flat joint.** Shear a planar joint at arbitrary azimuth θ: `τ–Δ` identical for all θ to solver tol; `+x` vs `−x` vs `+y` identical strength. Any azimuth dependence must come ONLY from roughness. Catches hard-coded axes / non-objective friction.
4. **Patton sawtooth/pyramid (L1).** Regular pyramid angle `i₀`, CNL, pre-degradation: assert `τ = σ_n·tan(φ_b + i₀)` and dilation slope `du_n/du_t = tan(i₀)` within 2%.
5. **Reduce-to-2D.** Drive the 3D model with slip confined to `s1` (`u_y=0`), compare `τ(Δ)`, dilation, per-cycle decay against the verified 2D capstone (`rock_joint_shear.py`, manual §11.9) to solver tolerance.
6. **KKT / non-penetration.** Assert `g_N ≥ 0`, `σ_n ≥ 0`, complementarity `g_N·σ_n ≈ 0`; under tension the interface opens (`σ_n→0`), carries no load.
7. **Patch test (FEM interface).** Two bonded blocks, non-matching interface discretizations: rigid-body translation → interface tractions and damage exactly zero; uniform imposed traction → constant relative displacement + spatially constant traction across the non-conforming interface.
8. **MMS O(h²).** Manufacture a smooth field with known jump across the joint, derive consistent body force + interface traction, confirm `O(h²)` displacement / `O(h)` traction under refinement. Extend `tests/test_chart_fem.py` MMS harness with the interface term.
9. **Mesh / penalty–AL convergence.** Refine interface mesh and raise penalty (or tighten AL/Uzawa): penetration → 0, stick-region slip → 0, tractions converge; AL becomes penalty-insensitive once converged; `ε_t → 0` recovers ideal stick/slip without locking.
10. **Energy / dissipation balance (PRIMARY cyclic gate).** Per increment: `external work = Δelastic + frictional dissipation (≥0) + damage dissipation (≥0)`. Per cycle: hysteresis loop area `== ∮τ·dδ_slip + Δ(damage energy)` to tight tol. Dissipation non-negative every step; no energy created on reversal. Anchor: extend CV-2 Cattaneo–Mindlin to a forward-reverse cycle and check the known Mindlin loop energy.
11. **Degradation sanity.** Damage disabled → multi-cycle loops perfectly repeatable (identical area each cycle). Damage enabled → `τ_peak(N)` and `i(N)` decay **monotonically**, saturate toward `φ_r`, never increase, never heal on reversal. Dilation reversal: on reversing shear, transient re-seating `du_n < 0` before re-dilation; accumulated dilation bounded by asperity amplitude.
12. **Frictional anisotropy (the 3D payoff).** Real Inada surface: forward vs reverse and along-`s1` vs along-`s2` show measured directional anisotropy (different peak/dilation matching JRC anisotropy); the two shear directions degrade independently; asymmetry shrinks over cycles toward `φ_r`. This is the check that distinguishes the 3D model from the 2D capstone.
13. **(D, cohesive) Mode-mixity consistency + path-independence.** Pure I/II/III recover prescribed single-mode TSLs and energies; 45° mixed path matches the B-K/power-law `G_c(B)`; proportional vs non-proportional paths to the same end state dissipate the same energy (catches the Turon–Camanho I/II coupling failure).

State explicitly in the manual which checks are **analytic** (L0, flat Coulomb, isotropy, Patton, reduce-to-2D, KKT, patch, MMS, energy balance, Cattaneo) vs **calibration-only** (real-Inada degradation rate, anisotropy magnitude).

---

## 4. STAGED BUILD ORDER (lowest-risk first)

- **Stage A — geometry (lowest risk, ~2 h).** `profile_height_chart_3d.py` + `AnalyticPyramid3D`. Gate: **L0** (check 1). Pure copy-with-Fourier-adaptation of a verified 1D module.
- **Stage B — rigid 3D contact + 3 drivers (~3 h).** `rock_joint_shear_3d.py` (`bake_height_3d`, `_interp_2d`, `contact_forces_3d`, `solve_z_equilibrium`, three mode drivers, CNL/CNS/CNV switch). Gates: **flat Coulomb (2), isotropy (3), Patton (4), reduce-to-2D (5), KKT (6)**. Numpy, no FEM — decisive and cheap before the expensive coupling.
- **Stage C — data + viz on rigid results (~2 h).** `joint_data_io.py`, `surface_anim_3d.py`, `plot_rock_joint_3d.py`. Produces the directional rose, dilation curves, and first GIF from Stage-B histories. Low-risk, unblocks visual review early.
- **Stage D — deformable FEM coupling, monotonic (~4 h).** `rock_joint_cyclic_fem.py::solve_contact_pair` with normal+frictional interface law, single forward pass. Gates: **patch (7), MMS O(h²) (8), mesh/penalty convergence (9)**. Reuses `ChartVectorFEMSolver` entirely.
- **Stage E — cyclic + degradation (highest risk, ~4 h).** Add Plesha degradation, `JointState`, `run_cyclic`, energy ledger. Gates: **energy balance (10), degradation sanity (11), Cattaneo–Mindlin cycle**. Then **frictional anisotropy (12)** on real Inada sub-domains.
- **Stage F — cohesive mixed-mode (OPTIONAL, defer).** `cohesive_mixed_mode.py`. Gate: **mode-mixity (13)**. Only after E is green.

Inada data: use `characterize_inada_joint.load_height_map` directly on the 4 CSVs (do NOT decimate via `export_profile`); extract rectangular sub-domains (e.g. 512×512 at 23.4 µm) or downsample 2–4× for prototyping. `DX_MM=0.0234` in both x and y.

---

## 5. RISKS + MITIGATIONS

- **PyVista env traps (high likelihood, low severity).** `cmap` string → ImportError; `open_movie` raises (no `imageio-ffmpeg`); F-order raveling; `framerate` vs `fps` arg swap. → Bake all four fixes into `surface_anim_3d.py` from the start; default deliverables to **GIF**; pass Colormap **objects**; `ravel(order='F')` everywhere; always `pl.close()`. Run viz with `/Users/wsun/opt/anaconda3/bin/python`.
- **Penalty conditioning in coupled FEM Newton (medium/high).** Too-stiff `eps_n` → ill-conditioned `3(N1+N2)` solve; too-soft → penetration. → Scale `eps_n ~ C·E/cell_size` (Hertz heuristic), `relax=0.5` for neural/rough surfaces, backtracking line search; offer AL (`augmented_lagrangian.py`, persistent multiplier) as the penalty-insensitive fallback once converged (check 9).
- **Energy balance is the make-or-break (high severity).** Friction + degradation + penalty can silently create energy on reversal. → Make the energy ledger the PRIMARY acceptance gate (check 10) before any expensive 3D cyclic run; instrument the solver to emit per-increment work/elastic/friction/damage from day one of Stage E; anchor against the analytic Cattaneo–Mindlin loop.
- **Dilatancy non-associativity / non-symmetric tangent (medium).** `ġ_N = tan(i)|ġ_T|` makes the contact tangent non-symmetric → Newton may stall. → Start with a symmetrized/secant tangent + line search; verify against the Patton analytic slope (check 4) before trusting full consistent linearization.
- **Spectral bias of the height chart on cusps (medium).** `NeuralHeight2D` may smooth sharp asperities (the known SDF failure this project exists to beat). → L0 on the pyramid (sharp ridge) is the guard; report slope-RMS preserved vs an ambient SDF, exactly as the 2D capstone's chart-over-level-set story.
- **Cost of full Inada grid (medium).** 2095×3130 ≈ 6.5 M points is too large to train/shear directly. → Sub-domain + downsample for the pipeline; reserve full-grid only for a final showcase figure.
- **Naming collision in the manual (low severity, high confusion).** "in-plane" already used in the plane-stress sense. → Define the joint-local triad once in a new §11.10 and cross-reference; tie every figure axis to the x–z observation plane.
- **`StructuredGrid` mutate-vs-rebuild (low).** Editing `.points` in place without reassignment may not refresh in PyVista 0.38.6. → Always reassign `g.points = pts`; update scalars by re-setting the same key (never add a new array per frame).

---

Key existing files to reuse (verified present): `solvers/contact/profile_chart_2d.py`, `benchmarks/contact/cv_numerical/rock_joint_shear.py`, `benchmarks/contact/cv_numerical/rock_joint_capstone.py`, `postprocessing/characterize_inada_joint.py`, `solvers/fem/chart_vector_fem.py`, `solvers/fem/linear_elastic.py`, `solvers/contact/{penalty,friction,augmented_lagrangian,gap}.py`, `tests/test_chart_fem.py`. Inada CSVs present at `downloads/inada_granite/{rough,smooth}_{footwall,hangingwall}.csv`.