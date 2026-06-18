# Contact Mechanics Verification Manual (Neural Atlas)

Analytical benchmarks for verifying a neural-atlas contact implementation. Every
result below is a closed form derived in SymPy and adversarially cross-checked
against Johnson, *Contact Mechanics* (1985) and Timoshenko & Goodier, *Theory of
Elasticity*. CV-1..CV-4 reproduce the contact problems of Liu & Sun (2020),
"ILS-MPM" (CMAME 369:113168); CV-5 (superformula) and CV-6 (Koch snowflake) are
additional transition-map / resolution-independence benchmarks.

This manual is the **verification** companion to:
- `docs/contact_theory_manual.md` — contact *algorithms* (penalty, AL, friction, topology);
- `contact_atlas/03_mathematical_theory.md` — contact *theory* (variational, well-posedness).

Each benchmark gives: **Goal → Formulation → Pass criteria → Artifact/Test**.
Symbolic artifacts live in `docs/hertz_derivation/`; the numpy evaluators that
reproduce them (and drive the figures) live in `postprocessing/contact_fields.py`.

---

## 1. Purpose & scope

**Primary purpose — verify the neural coordinate charts.** These benchmarks use *analytical*
charts (Monge patches, Flamant/superformula boundary parametrizations, analytical SDFs) for which
the contact gap, normal, stress field, and force–displacement response are known in **closed form**.
When the project switches to **trained neural charts** — a neural SDF (`atlas/sdf/train_sdf.py::SDFNet`)
and/or a `ChartDecoder` (`common/models.py`, trained by `atlas/charts/train_atlas.py`) replacing the
analytical chart — these closed forms become the **acceptance targets**: train the neural chart on the
same analytical shape, then check that its gap/normal/field/force match the analytical reference within
a tolerance that accounts for neural approximation error. The concrete recipe is in **§11 (Neural
coordinate-chart verification protocol)**; `postprocessing/contact_fields.py` holds the numpy reference
values and `tests/test_neural_chart_verification.py` is the (skeleton) harness.

Verification is **two-level**, so a failure localizes to geometry vs mechanics:
- **L0 — geometry/kinematics** (solver-free): does the neural chart reproduce the analytical **gap** and
  **normal** at query points? (gap RMSE, normal angle error.) This isolates the chart/SDF itself.
- **L1 — mechanics**: does a contact *solve* built on the neural chart reproduce the analytical
  **pressure / stress field / contact radius / force–displacement**?

These are small-strain, linear-elastic, quasi-static, frictionless-or-monotone references — code-
verification targets, not models of the full finite-deformation frictional MPM; they must be matched in
the appropriate limit (see §11).

---

## 2. Transition-map contact kinematics

In the neural-atlas framing, contact is read from the **chart transition map**
rather than a level set. For a surface point $x_B=\varphi_j^B(\xi_B)$ on body $B$,
the closest point on body $A$ is obtained by inverting $A$'s surface chart
(Newton `common/geometry.py::invert_decoder`), and the contact normal is the
chart-Jacobian cross product

$$ n = \frac{t_1\times t_2}{\lVert t_1\times t_2\rVert},\qquad t_\alpha=\frac{\partial\varphi^A}{\partial\xi^\alpha}. $$

The undeformed gap is the chart form of ILS-MPM Eq. 25,
$g_{n0}=(\varphi^{(j)}-\varphi^{(i)})\cdot n$.

**Honest scope.** For paraboloid (Monge) surfaces the chart-Jacobian normal equals
the SDF-gradient normal $\nabla\phi/\lVert\nabla\phi\rVert$ (ILS-MPM Eq. 15) **only at
leading order / at the apex**; the gap $g_{n0}=r^2/2R$ is the $a\ll R$ truncation
of the exact projection. The production oracle `solvers/contact/gap.py::evaluate_gap`
currently uses the **SDF gradient**, not the chart cross product — the transition-map
oracle is the proposed alternative, not the current code.

---

## 3. CV-1 — Hertz normal contact

**Goal.** Reproduce the Hertz pressure, contact size, force–approach law, and
subsurface stress for two elastic bodies.

**Formulation (3D axisymmetric).** With $1/E^*=(1-\nu_1^2)/E_1+(1-\nu_2^2)/E_2$
and $1/R=1/R_1+1/R_2$, gap $g_{n0}=r^2/2R$ and (via the MDR / Sneddon reduction):

$$ a=\Big(\tfrac{3FR}{4E^*}\Big)^{1/3},\quad p(r)=p_0\sqrt{1-r^2/a^2},\quad p_0=\tfrac{2E^*a}{\pi R}=\tfrac{3F}{2\pi a^2},\quad \delta=\tfrac{a^2}{R},\quad F=\tfrac43 E^*\sqrt{R}\,\delta^{3/2}. $$

On-axis subsurface (Johnson eq 3.45), $\nu=0.3$: max shear at $z/a=0.481$,
$\tau_{\max}=0.31\,p_0$, $\sigma_{vM}^{\max}=0.62\,p_0$; first yield at
$p_0\approx1.61\,\sigma_Y$.

**2D line contact (plane strain):** $a=2\sqrt{P'R/\pi E^*}$, $p_0=2P'/\pi a=\sqrt{P'E^*/\pi R}$
($P'$ = load per thickness). For two identical bodies $E^*=E/2(1-\nu^2)$.

**Pass criteria.**
- $a$, $p_0$, $\delta$, $F(\delta)$ to **< 1%** of the closed forms.
- $\int p\,dA = F$ to **< 0.5%**.
- subsurface max-shear depth $z/a=0.48\pm0.02$, magnitude $0.31\,p_0\pm2\%$.

**Artifact / test.** `docs/hertz_derivation/hertz_transition_map.py`;
`postprocessing/contact_fields.py::{hertz_3d_params, hertz_pressure, hertz_subsurface_axis, line_contact_params}`.

> **Correction recorded.** Liu & Sun Eq. 46 ($2a=\sqrt{F(1-\nu^2)R/\pi E}$) is a
> **factor of 4 low in width** (16× in area) vs the standard $2a=4\sqrt{P'R(1-\nu^2)/\pi E}$.
> Use the standard form; Fig. 22 plots both.

---

## 4. CV-2 — Cattaneo–Mindlin frictional partial slip

**Goal.** Reproduce the stick/slip partition under a tangential load $Q<\mu P$ on a
pre-formed Hertz contact (the elastic check the rigid-sliding $a=\mu g$ benchmark omits).

**Formulation.** Stick radius and tractions:

$$ \frac{c}{a}=\Big(1-\frac{Q}{\mu P}\Big)^{1/3},\qquad q(r)=\begin{cases}\mu p_0\sqrt{1-r^2/a^2}, & c\le r\le a\\[2pt]\mu p_0\big[\sqrt{1-r^2/a^2}-\tfrac{c}{a}\sqrt{1-r^2/c^2}\big], & r\le c\end{cases} $$

with compliance $\delta_x=\tfrac{3\mu P}{16aG^*}\big[1-(1-Q/\mu P)^{2/3}\big]$,
$1/G^*=(2-\nu_1)/4G_1+(2-\nu_2)/4G_2$.

**Pass criteria.** $c/a$ to **< 1%**; $\int q\,dA=Q$ to **< 1%**; tractions match $q(r)$.

**Artifact / test.** `docs/hertz_derivation/hertz_transition_map.py` (Part E);
`postprocessing/contact_fields.py::{cattaneo_stick_radius, cattaneo_traction}`.

---

## 5. CV-3 — Brazilian disc (diametral compression)

**Goal.** Reproduce the disc stress field, tensile-strength formula, displacement,
and diametral compliance via the Flamant-chart superposition.

**Formulation.** Field = two Flamant load-point charts + uniform biaxial tension
$T=2P/\pi Dt$ ($D=2R$) fixing a traction-free rim. Center:
$\sigma_{xx}=+2P/\pi Dt$ (tensile), $\sigma_{yy}=-6P/\pi Dt$ (compressive).
Splitting strength $\sigma_t=2P/\pi Dt$ (ASTM D3967). Flamant displacement carries
a $\ln r$ settlement and a multivalued $\theta\sin\theta$ branch; the gauge
$a+b=(1+\nu)p/\pi E$ is compatibility-fixed (not free). Point-load diametral
compliance **diverges** logarithmically; with a finite contact width $b$,
$\Delta_D=\tfrac{4P}{\pi Et}\big[\ln(2R/b)-\tfrac{1-\nu}{2}\big]$ (the prefactor and
the $-(1-\nu)/2$ are robust; the log argument carries the $b$↔contact-width calibration).

**Pass criteria.**
- center $\sigma_{xx},\sigma_{yy}$ to **< 1%**; rim traction $\approx 0$ off the poles.
- $\nabla\!\cdot\sigma=0$, $\nabla^2(\mathrm{tr}\,\sigma)=0$; $\int_{-R}^{R}\sigma_{yy}(x,0)\,dx=-P/t$.

**Artifact / test.** `docs/hertz_derivation/brazilian_disc_atlas.py`;
`postprocessing/contact_fields.py::brazilian_field`.

---

## 6. CV-4 — Nine-disc packing (isotropic confinement)

**Goal.** Reproduce the symmetric force network and per-disc stress field of a 3×3
array under isotropic compression.

**Formulation.** No diagonal contacts ($2\sqrt2 R>2R$). By full D4 symmetry +
compatibility, every contact carries the same normal force $N$, and friction is
inactive by reflection parity. Each disc therefore sees four equal inward loads $N$
at N,S,E,W, so the unit cell = two perpendicular Brazilian solutions (4 Flamant + 2
uniform-tension charts). Center is **equibiaxial in-plane compression**
$\sigma_{xx}=\sigma_{yy}=-2N/\pi Rt$. Force–compression law: $3\,\Delta_x(N)=2d$ with

$$ \Delta_x(N)=\frac{4N}{\pi Et}\ln\frac{2R}{b}+\frac{N}{\pi Et}\big[(2-\pi)\nu+\pi-6\big],\quad b=2\sqrt{\tfrac{N R_{\rm rel}}{\pi E^*}}. $$

For $R=1.7$ mm, $E=100$ GPa, $\nu=0.3$, $t=1$ mm: $N(d{=}10^{-4}\,\text{mm})\approx0.887$ N/mm.

**Pass criteria.** equibiaxial center to **< 1%**; identical interior field across the
9 discs; uniform contact force $N$ (rigid-plate limit); $N(d)$ within the modeling
caveats below.

**Caveats (record in any comparison).** (i) Equal-$N$ and identical-disc collapse are
exact only for **rigid** plates; the paper's **deformable** plates make disc–plate
contacts $\sqrt2$ wider/softer, so contact forces differ by O(few %) and the 9 discs
differ near their boundary contacts. (ii) The paper sets **initial gaps**; if present,
contacts engage progressively and the static square-lattice premise weakens.
(iii) This is **exact superposition, not a test of the transition-map machinery**
(each Brazilian sub-field is independently rim-admissible, so the charts simply add;
the 90° contacts only place the poles).

**Artifact / test.** `docs/hertz_derivation/nine_disc_atlas.py`;
`postprocessing/contact_fields.py::nine_disc_unit_cell_field`.

---

## 7. CV-5 — Nonconvex superformula contact (the transition-map test)

**Goal.** Demonstrate, on the case where superposition fails (nonconvex, non-collinear,
curved surfaces), whether the chart machinery helps: a rotating Gielis-superformula "cam"
$r(\theta)=\big(|\tfrac1a\cos\tfrac{m\theta}{4}|^{n_2}+|\tfrac1b\sin\tfrac{m\theta}{4}|^{n_3}\big)^{-1/n_1}$
drives a free nonconvex follower (rigid-body / DEM).

**Formulation (honest).** Each star-shaped boundary is an analytic chart
$\varphi_X(\theta)=c_X+R(\alpha_X)\rho_X(\theta)(\cos\theta,\sin\theta)$. The **gap**
$g_B(p)=|p-c_B|-\rho_B(\psi)$ is a *single-body inverse radial chart* $\varphi_B^{-1}$
(NOT itself a transition map); the **transition map** is the boundary-to-boundary
correspondence $\theta_A\mapsto\psi_B=(\varphi_B^{-1}\!\circ\varphi_A)(\theta_A)$. It **helps**
for nonconvex star shapes via: (1) **multi-arc contact** — a boundary scan enumerates all disjoint
penetrating arcs, where a single closest-point projection returns one foot (demonstrated head-to-head
in `test_chart_vs_single_cpp_multi_arc`); (2) **exact analytic surface normals at smooth boundary
points** (`outward_normal`, FD-matched); (3) a **globally single-valued $C^1$ gap** (star-shapedness)
vs the medial-axis non-smoothness of the true Euclidean SDF in concavities. Gap/normal are a
**matched pair** ($f=-\epsilon_n\langle g\rangle\nabla_p g$ conservative; radial mode) with an optional
bounded 1D refine (refined mode).

**Honest caveats.**
- The radial gap $\neq$ perpendicular distance: it **overestimates** by $g_{\rm rad}=g_{\rm perp}/\cos\alpha+O(\text{gap}^2)$
  at the radial foot ($\alpha$ = angle between the ray and the surface normal); near a lobe the true
  nearest foot is a *different* boundary point, so the radial gap can exceed the perpendicular distance
  several-fold (comparison figure). The (fixed) refine recovers the perpendicular distance to ~1% but
  re-incurs medial-axis non-smoothness; it is **verification-only**, not wired into the integrator.
- The radial gap is $C^1$-smooth but is **not** a distance/Eikonal function: $\lVert\nabla g\rVert$ can
  reach $\sim10^4$ in deep concave valleys (the radial normal rotates fast there).
- For $n<1$ shapes the lobe extrema are **cusps** (one-sided normals); the demo's penalty uses the
  radial-gap-gradient normal, so "exact tip normals" is a property the chart *exposes*, not one the
  running force exercises.
- Required precondition: **radial visibility** of the contact region from each center (star-shapedness).
- Scope is rigid-body (contact *kinematics*, not the deformable solver). The driven contact is an
  **impulsive strike** that launches the follower — the displacement magnitude is penalty-dependent
  (a stiffer $\epsilon_n$ reduces it); the work-energy residual ($\sim$1–4%) is explicit-integration
  error that shrinks with $\epsilon_n$ and $dt$. The gap is an inverse radial chart, not an SDF replacement.

**Pass criteria (all verified).**
- star-shaped inside + $g\approx0$ on the boundary; matched $\nabla g$ vs FD < 1e-4 (rel).
- area/inertia → $\pi R^2$, $\tfrac12 mR^2$ in the circle limit (< 1e-4).
- **free-A control conserves momentum** to < 1e-8 (measured 5e-15 lin / 1e-13 ang).
- driven cam: follower displaced > 0.3 and spun; work-energy $\Delta KE_B\approx$ energy injected.
- **multi-arc**: on a wedged pose the chart scan reports $\ge 2$ disjoint contact arcs while a single
  closest-point projection reports 1 (head-to-head); the driven run reaches $\ge 2$ (de-duplicated).
- flank radial gap $\ge$ perpendicular gap; the (fixed) refine recovers the perpendicular distance to ~1%.
- analytic `dradius_dtheta` finite at cusp angles (no clip artifact); tangent magnitude bounded.

**Artifact / test.** `solvers/contact/supershape.py`;
`benchmarks/contact/supershape_cam_drive.py` (+ `--free-A`);
`postprocessing/plot_supershape_demo.py`; `tests/test_supershape_contact.py` (9 tests).

---

## 8. CV-6 — Koch snowflake fractal contact (resolution-independence)

**Goal.** Showcase the decisive advantage of the coordinate-chart / transition-map representation
over a level-set: contact between *self-similar (fractal)* boundaries, where a uniform SDF grid is
restricted by the resolution needed to render the pattern, while the recursive chart resolves contact
to any depth on demand.

**Formulation.** The Koch snowflake is an IFS (4 contraction maps, scale 1/3, applied recursively) —
an exact, **O(1)-storage, resolution-independent** chart $\varphi(s)$ evaluable to any depth $n$.
Contact uses a **pruned recursive descent** (the IFS doubles as a bounding hierarchy). The inside/outside
**detection** test prunes to **O(depth) per query** (measured $\approx$21 nodes, bounded in $n$) — this is
the resolution-independent primitive. The *exact* signed gap + normal (`nearest_boundary`) is verified
equal to the brute-force polyline distance but costs more — its descent visits the boundary detail near
the query, which on a fractal grows $\sim O(3^n)$ for a fixed-distance point (intrinsic: $\sim 3^n$ tiny
segments cluster near any boundary point); it is evaluated only at the penetrating points detection already
flagged, at a finite practical level ($n=3$: $\approx$110 nodes). The snowflake is
NOT star-shaped for $n\ge2$ (verified), so the *parametric* chart is used, not the radial gap (CV-5).
The outward normal is the **gradient of the signed distance** — $\pm(\mathbf{x}-\mathbf{foot})/\lVert\cdot\rVert$,
sign-corrected by inside/outside — which equals the segment normal on a segment interior and is the
*true* gap-ascent direction at a vertex/spike foot (the bare segment normal is wrong there). The normal
exists at any finite level, but is undefined *at* a vertex and at the fractal limit.

**Why it beats the level-set — the honest cost decomposition.** Compare on the right axes:

| | storage / build | per query | resolution |
|---|---|---|---|
| recursive IFS chart | **$O(1)$** (4 maps) | $O(\text{depth})$, measured $\approx$21 nodes | **any depth on demand** |
| uniform SDF grid | $O(9^n)$ cells | $O(1)$ lookup | capped at finest cell |
| adaptive/narrow-band SDF | $O(4^n)$ boundary cells | $O(1)$ lookup | capped at max subdivision |
| fixed-capacity neural SDF | $O(\text{params})$ | $O(1)$ eval | capped by capacity / spectral bias |

A precomputed SDF is *cheaper per query* (one lookup) — the chart does **not** win on per-query speed. The
chart's decisive, defensible wins are **storage** ($O(1)$ vs exponential) and **resolution-independence**
(no maximum depth baked in). An adaptive/octree SDF narrows the storage gap to $O(4^n)$ but does not close
it (still exponential, still depth-capped). A fixed-capacity neural SDF represents any *finite* pre-fractal
level fine, but cannot keep refining self-similar detail without growing its parameter count — so it is
likewise resolution-capped; the chart stores the generating rule instead. (The neural-SDF refinement
ceiling is argued here, not yet measured — see §11.6.)

**Honest scope.** At the true fractal limit there is no tangent/normal — contact mechanics is
well-posed only at a finite pre-fractal level. The penalty *force model* struggles with thin fractal
spikes (near-point contact → weak penalty → a free body can tunnel); this is a **force-model**
limitation, not a chart one — chart *detection* still fires correctly, and the outward-normal sign is
verified separately (`test_contact_normal_repulsive`, sampling the whole interior incl. vertex feet).
The rigid-body engine conserves linear momentum (action–reaction pairing, $<10^{-8}$); note this is a
*bookkeeping* check that holds regardless of the normal, so it is **not** evidence the normal is correct.
The time-resolved dynamics demo therefore uses a robust **bounded regime**: a prescribed spinning Koch
cam drives a spring-loaded **flat-faced** follower (a thin 1-DOF plate) under overdamped (quasi-static)
relaxation. A flat face touches only the cam's right boundary (outward normals $\approx+x$), so contact
is shallow (max penetration $\approx0.06R$), there is no fat-body overlap, and tunnelling is structurally
impossible — the plate *rides* the fractal cam, tracing its lift curve. Free fractal-on-fractal collision
(two fat snowflakes) is explicitly out of scope.

**Pass criteria (all verified — 8 tests).**
- exact IFS geometry ($3\cdot4^n{+}1$ vertices; perimeter $\propto(4/3)^n$).
- inside-test = brute-force point-in-polygon (0 mismatches at $n=4$).
- **resolution-independent per-query cost**: inside-query nodes plateau at $\approx$21 (bound $<80$) for
  $n$ up to 12, $\ll 4^n$, $\ll 9^n$ — vs $O(1)$-storage IFS, $O(9^n)$ uniform / $O(4^n)$ adaptive SDF.
- not star-shaped for $n\ge2$ (no radial shortcut).
- **exact signed gap magnitude** = brute-force polyline distance (machine precision, $n=4$) — guards the
  pruning correctness (a prior bug squared a negative slack and skipped the nearest segment).
- repulsive normal **everywhere**: $g<0$ inside; the gap increases along $+\mathbf{n}$ for the full
  interior grid, including the $\approx$85% of points whose nearest foot is a vertex.
- contact micro-arcs grow with depth; the rigid-body engine conserves momentum ($<10^{-8}$).

**Artifact / test.** `solvers/contact/koch.py`; `benchmarks/contact/koch_gears_drive.py`
(+ `--free-A` control); `postprocessing/plot_koch_demo.py`; `tests/test_koch_contact.py` (8 tests).

---

## 9. Master verification table

| Item | Analytical target | Tolerance | Artifact |
|---|---|---|---|
| Hertz contact radius | $a=(3FR/4E^*)^{1/3}$ | < 1% | hertz_transition_map.py |
| Hertz peak pressure | $p_0=3F/2\pi a^2$ | < 1% | " |
| Hertz force–approach | $F=\tfrac43 E^*\sqrt R\,\delta^{3/2}$ | < 1% | " |
| Pressure integral | $\int p\,dA=F$ | < 0.5% | " |
| Subsurface max shear | $z/a{=}0.48$, $0.31p_0$ | ±2% | " |
| 2D line half-width | $a=2\sqrt{P'R/\pi E^*}$ | < 1% | " |
| C–M stick radius | $c/a=(1-Q/\mu P)^{1/3}$ | < 1% | hertz_transition_map.py (E) |
| Brazilian center | $+2P/\pi Dt$, $-6P/\pi Dt$ | < 1% | brazilian_disc_atlas.py |
| Brazilian global balance | $\int\sigma_{yy}\,dx=-P/t$ | < 0.5% | " |
| Nine-disc center | $-2N/\pi Rt$ (equibiaxial) | < 1% | nine_disc_atlas.py |
| Nine-disc force law | $3\Delta_x(N)=2d$ | caveated | " |
| Supershape star/gap | inside $\Leftrightarrow g<0$; $g\approx0$ on boundary | machine | supershape.py |
| Supershape gap/normal | $\nabla g$ matched, conservative | < 1e-4 rel | supershape.py |
| Supershape free-A momentum | linear & angular conserved | < 1e-8 | supershape_cam_drive.py |
| Supershape multi-arc | $\ge 2$ contact arcs (vs 1 for single CPP) | count | supershape_cam_drive.py |
| Koch IFS geometry | $3\cdot4^n{+}1$ verts, perimeter $(4/3)^n$ | exact | koch.py |
| Koch inside-test | = brute-force point-in-polygon | 0 mismatch | koch.py |
| Koch storage / per-query | $O(1)$ IFS vs $9^n$/$4^n$ SDF; nodes/query plateau | $\approx$21 ($<80$) @ $n\le12$ | koch.py |
| Koch normal (incl. vertex feet) | $+\mathbf{n}$ is gap-ascent over full interior | repulsive everywhere | koch.py |
| Koch engine momentum | free-free conserved (bookkeeping) | $<10^{-8}$ | koch_gears_drive.py |
| Equilibrium / compatibility | $\nabla\!\cdot\sigma=0$, biharmonic | machine | all |

---

## 10. Figures

Analytical re-plots of the Liu & Sun figures (paper style), embedded below. Regenerate the
Hertz/Brazilian/nine-disc set with `python3 postprocessing/plot_liusun_all.py` and the superformula
demo with `python3 postprocessing/plot_supershape_demo.py`. Line plots leave commented
numerical-overlay slots for the implementation's results.

### CV-1 — Hertz normal contact
Geometry & BCs (Fig.12), vertical-stress bulb $\sigma_{yy}$ (Fig.13), and contact tractions
$\sigma_n,\sigma_t$ including Cattaneo–Mindlin partial slip (Fig.14, CV-1/CV-2).

<img src="../figures/liusun_fig12_hertz_schematic_pub.png" width="30%"> <img src="../figures/liusun_fig13_hertz_sigma_yy_pub.png" width="34%"> <img src="../figures/liusun_fig14_hertz_traction_pub.png" width="33%">

Vector PDF: [Fig.14 traction](../figures/liusun_fig14_hertz_traction_pub.pdf).

### CV-3 — Brazilian disc
Geometry (Fig.21), contact radius $a$–$F$ — standard Hertz vs the paper's Eq. 46 (Fig.22), and the
$\sigma_{yy}$ disc field (Fig.23).

<img src="../figures/liusun_fig21_brazilian_schematic_pub.png" width="30%"> <img src="../figures/liusun_fig22_brazilian_aF_pub.png" width="33%"> <img src="../figures/liusun_fig23_brazilian_sigma_yy_pub.png" width="33%">

Vector PDF: [Fig.22 a–F](../figures/liusun_fig22_brazilian_aF_pub.pdf).

### CV-4 — Nine-disc packing
Geometry (Fig.15) and the max-principal $S_1$ + shear $\sigma_{xy}$ over the 3×3 array (Fig.16;
identical per disc by symmetry, contact concentrations at N/S/E/W).

<img src="../figures/liusun_fig15_nine_disc_schematic_pub.png" width="32%"> <img src="../figures/liusun_fig16_nine_disc_S1_Sxy_pub.png" width="64%">

### CV-5 — Nonconvex superformula contact
Cam-drive animation (the rotating cam drives the free follower; red = active contact), the chart-vs-SDF
comparison (smoothness advantage **and** radial bias), the run summary, and a static 3-frame montage.

<img src="../figures/supershape_cam_drive.gif" width="48%"> <img src="../figures/supershape_chart_vs_sdf_pub.png" width="48%">

<img src="../figures/supershape_summary_pub.png" width="55%"> <img src="../figures/supershape_frames_pub.png" width="44%">

### CV-6 — Koch snowflake fractal contact
Headline cost decomposition (LEFT: storage — $O(1)$ IFS chart vs $O(9^n)$ uniform / $O(4^n)$ adaptive SDF;
RIGHT: per-query — the SDF's $O(1)$ lookup vs the chart's *bounded* $\approx$21-node descent), the growth
of contact micro-arcs with depth, and the exact boundary at levels 1/3/5 vs a grid-limited SDF.

<img src="../figures/koch_cost_scaling_pub.png" width="62%"> <img src="../figures/koch_contact_count_pub.png" width="36%">

<img src="../figures/koch_geometry_pub.png" width="98%">

**Time-resolved contact dynamics.** A prescribed spinning Koch cam drives a spring-loaded 1-DOF Koch
follower (overdamped/quasi-static relaxation — unconditionally stable for stiff contact, so the follower
*rides* the fractal cam profile without tunnelling). Left: the animation (red = chart-detected contact
set); right: the follower displacement traced against cam angle (the fractal cam lift curve).

<img src="../figures/koch_contact_dynamics.gif" width="52%"> <img src="../figures/koch_follower_displacement_pub.png" width="44%">

The earlier static-cam view (net contact force arrow on a *fixed* follower) is retained as a schematic:

<img src="../figures/koch_spinning_contact.gif" width="45%">

---

## 11. Neural coordinate-chart verification protocol

This is the operational recipe for using CV-1..CV-6 to verify a **trained neural chart** once the
project replaces the analytical charts. The principle: *train the neural object on the same analytical
shape, then compare its output to the closed form.*

### 11.1 Substitution map (analytical → neural)

| Analytical object (now) | Neural replacement (under test) | Trained by |
|---|---|---|
| analytical SDF / implicit shape | neural SDF `atlas/sdf/train_sdf.py::SDFNet` (3-in → 1 scalar) | `atlas/sdf/train_sdf.py` |
| analytical gap $g_{n0}$, normal | `solvers/contact/gap.py::evaluate_gap` (neural SDF + autograd normal) | — |
| volumetric chart map $\xi\mapsto x$ | `common/models.py::ChartDecoder` (frame-anchored, 3→3) | `atlas/charts/train_atlas.py` |
| chart inverse $x\mapsto\xi$ | `common/geometry.py::invert_decoder` (Newton; needs seed/frame/scale) | — |

**Note — `ChartDecoder` is volumetric, not a boundary $\varphi(\theta)$.** It maps a reference cube
$\xi\in[-r,r]^3$ to $x\in\mathbb R^3$ anchored to a seed + tangent frame; there is no $\theta$ input. The
directly-comparable neural object for these 2-D contact shapes is therefore the **neural SDF** (primary
path). A boundary parametrization $\varphi(\theta)$ is a *derived* object — the image of the chart's
in-plane reference locus ($\xi_3=0$) — so a boundary-to-boundary check requires supplying the seed/frame
and sweeping the in-plane coordinate. The references are `postprocessing/contact_fields.py` (CV-1..4) and
`solvers/contact/supershape.py` (CV-5).

**Dimensionality.** `SDFNet`/`evaluate_gap` are 3-D (in_dim=3, normal $(N,3)$); CV-3/4/5 are 2-D — embed
the shape at $z=0$, compare only the in-plane normal, and assert $|n_z|$ is small.

### 11.2 Two-level recipe (run per benchmark)

**L0 — geometry/kinematics (solver-free; do this first).** Sample query points $\{x_k\}$ near the contact
region; evaluate the neural chart's gap $g^{\rm nn}(x_k)$ and normal $n^{\rm nn}(x_k)$ and compare to the
analytical reference:
- gap: $\mathrm{RMSE}(g^{\rm nn}-g^{\rm ana})/L \le \tau_g$  ($L$ = body size);
- normal: $\max_k \angle(n^{\rm nn}_k, n^{\rm ana}_k) \le \tau_n$;
- zero level set: $|g^{\rm nn}|$ on the analytical boundary $\le \tau_g$.

**L1 — mechanics.** Run the contact solve with the neural chart in place and compare the derived
quantities (pressure, stress field, contact radius, force–displacement, stick radius) to the CV pass
criteria, with the looser neural tolerances below.

**Compare like-for-like.** A neural **SDF** returns the *Euclidean* gap/normal, so compare it to the
*Euclidean* reference (for CV-5 use `supershape.closest_point_refine` / a dense closest-point scan, **not**
the radial gap, which is a different quantity — see §7). A neural **ChartDecoder** is compared
*boundary-to-boundary*: $\max_\theta\lVert\varphi^{\rm nn}(\theta)-\varphi^{\rm ana}(\theta)\rVert$ and the
tangent/normal from its Jacobian.

### 11.3 Tolerances (neural vs analytical)

Analytical-vs-analytical checks are machine precision (§9). Neural charts carry training error, so use:

| Quantity | Analytical tol (§9) | Neural tol (realistic) | Driver of the looser bound |
|---|---|---|---|
| gap (L0) | machine | $\tau_g \sim 10^{-3} L$ | SDF Eikonal training error (~1e-3) |
| normal angle (L0) | machine | $\tau_n \sim 1\text{–}2^\circ$ | $\lVert\nabla\phi\rVert\ne1$ off the surface |
| pressure / stress field | < 1% | ~5% | gap/normal error + discretization |
| contact radius $a$ | < 1% | ~3–5% | $a$ is gap-sensitive |
| force / approach | < 1% | ~5–10% | compounded |
| momentum (free control) | < 1e-8 | < 1e-8 | dynamics-only; neural-independent |

$\tau_g\sim10^{-3}L$ is a **target/assumption** (the typical neural-SDF Eikonal residual), **not yet
measured** on these shapes — the trainer currently fits only the rabbit. Record the actual `final_eikonal`
once an SDF is trained on the sphere/disc/superformula and tighten $\tau_g$ to it. The harness default
`TAU_GAP_REL = 2e-3` is a deliberately relaxed starting value.

### 11.4 Neural-specific failure modes each benchmark catches

- **Non-unit $\lVert\nabla\phi\rVert$ (broken Eikonal):** inflates the gap and rotates normals → caught by L0 on **any** benchmark (CV-1 sphere is the cleanest).
- **Normal degradation near the medial axis / in concavities:** **CV-5** is the discriminating test — the analytical superformula chart is exact in the concave valleys where a neural SDF normal degrades; L0 gap/normal error there (vs the dense Euclidean reference) directly measures it.
- **Over-smoothed sharp features (cusps/edges):** the neural SDF rounds lobe tips → CV-5 L0 tip-normal/gap error grows. (Multi-arc *count* is an analytical-chart boundary-scan property, not a quantity a single-foot neural SDF produces; for a neural SDF the concavity catcher is the L0 gap/normal error, not an arc count.)
- **Chart Jacobian ill-conditioning / non-bijective decoder:** caught by the L0 inverse-chart residual and `stabilized_jacobian_ops` condition number (CV-1/CV-5).
- **Contact-radius / force drift:** small gap bias → systematic $a$, $p_0$, $F(\delta)$ offset → caught by L1 (CV-1, CV-3, CV-4).

### 11.5 Harness

`tests/test_neural_chart_verification.py` is the skeleton: one test per benchmark, each currently
`pytest.skip(...)` until a neural chart is provided. **Wiring status:** only the **SDF L0** path is
runnable today (CV-1 sphere and CV-5 superformula, compared to the dense Euclidean reference). The
**decoder L0** path and **all L1** paths are stubs — L1 needs a neural contact *solve* (no neural contact
solver exists yet) and decoder L0 needs a boundary-evaluable map (§11.1 note). Activate a path by
implementing its loader and removing the skip.

| Benchmark | Neural object | L0 (geometry) | L1 (mechanics) | Failure mode caught |
|---|---|---|---|---|
| CV-1 | neural sphere/cyl SDF | gap, normal near contact ✓wired | $a,p_0,F(\delta)$ | Eikonal, gap drift |
| CV-2 | + friction | (uses CV-1 chart) | $c/a$, $q(r)$ | normal error → traction |
| CV-3 | neural disc SDF | gap/normal on rim | $\sigma$ field, $\sigma_t$ | rim normal accuracy |
| CV-4 | neural disc SDF | per-disc gap/normal | equibiaxial center, $N(d)$ | multi-contact normals |
| CV-5 | neural supershape SDF | Euclidean gap/normal **in concavities** (vs dense ref) ✓wired | cam-drive match | **medial-axis normal degradation, cusp smoothing** |
| CV-6 | neural SDF on Koch level-$n$ | gap/Eikonal RMSE vs `koch.nearest_boundary` at rising $n$ | (out of scope) | **self-similar detail beyond fixed capacity (refinement ceiling)** |

### 11.6 Worked example (train → L0 → L1)

The one missing executable piece is an SDF trained on an **analytical shape** — the shipped
`atlas/sdf/train_sdf.py` fits only the rabbit. The intended sequence:

1. **Train** a neural `SDFNet` on the analytical shape by regressing to its closed-form signed distance
   (sphere $\phi=|x|-R$; disc; superformula via `supershape.closest_point_refine` for the Euclidean
   distance) with an Eikonal term. Save `<shape>_sdf.pt` and **record the final Eikonal residual** — that
   number sets $\tau_g$ (§11.3).
2. **Wire the loader**: make `load_neural_sdf("sphere")` (in `tests/test_neural_chart_verification.py`)
   return the trained `SDFNet`.
3. **Run L0** (geometry, solver-free):
   `pytest tests/test_neural_chart_verification.py -k cv1 -rs` — evaluates `evaluate_gap` and asserts the
   gap RMSE / normal angle against the Euclidean reference. This alone validates the chart in isolation.
4. **Run L1** (mechanics): once a neural contact *solve* exists, drop the neural SDF into the contact path
   and assert $a,p_0,F(\delta)$ (CV-1), $\sigma_t$ (CV-3), $N(d)$ (CV-4), cam-drive (CV-5) against
   `contact_fields.py` / the CV pass criteria with the §11.3 tolerances.

The right-hand sides (analytical references) are already runnable now:
`python3 postprocessing/contact_fields.py` (numpy self-test) and the three
`docs/hertz_derivation/*.py` scripts (symbolic self-checks). Steps 1 (analytical-shape SDF trainer) and
the L1 neural solver are the "later" neural-chart work this manual is written to verify.

**CV-6 refinement-ceiling experiment (proposed, not yet run).** The CV-6 claim that a *fixed-capacity*
neural SDF cannot keep resolving self-similar detail is currently argued, not measured. The clean way to
*earn* it on these analytical shapes: train one `SDFNet` of fixed width to regress the exact Koch signed
distance (sign from `koch.inside`, magnitude from `koch.nearest_boundary`) at increasing level $n$, and
plot the gap RMSE / Eikonal residual rising as $n$ grows past what the network can represent. That
measured curve — rather than an asserted ceiling — is the rigorous chart-vs-neural-SDF showcase, and it
also produces the $\tau_g$ number §11.3 still lacks. Until then the ceiling is stated as a conjecture with
its mechanism (finite parameters + spectral bias), and the chart's *measured* wins (storage,
resolution-independence) carry CV-6.

### 11.7 Level-set-free detection via radial boundary charts (implemented)

The SDF detector (`gap.py::evaluate_gap`) needs an *ambient* scalar field $\phi:\mathbb R^3\!\to\!\mathbb R$.
A contact **detection** algorithm needs no such field: a star-shaped body can be carried level-set-free
by a **radial boundary chart** $\rho:S^2\!\to\!\mathbb R^+$ (a height field on the sphere of directions —
*not* an ambient SDF), with center $c$ and orientation $Q$ (body axes as columns). In the body frame
$d=Q^\top(x-c)$, $r=\lVert d\rVert$, $\hat d=d/r$:

$$ \text{gap}(x)=r-\rho(\hat d),\qquad n=\widehat{\,Q\,\nabla_d F\,},\quad \nabla_d F=\hat d-\nabla_S\rho/r, $$

where $\nabla_S\rho=(I-\hat d\hat d^\top)\nabla_{\hat d}\rho$ is the **tangential** (projected) surface
gradient — the projection is load-bearing for any non-sphere (a raw autograd gradient of $\rho$ carries a
spurious radial component). This is the 3-D lift of the 2-D `supershape.py` oracle. The normal is the
*matched / conservative* one (penalty force $=$ gradient of $\tfrac12\epsilon_n\langle F\rangle^2 V$).
Implementation: `solvers/contact/chart_gap.py` (`RadialChart`, `SphereRho`, `SuperquadricRho`,
`evaluate_gap_chart`, `closest_point_refine_chart`).

**What is proven (acceptance vs the SDF benchmark).** Tested in `tests/test_chart_gap.py` and the head-to-head
`benchmarks/contact/chart_vs_sdf_detection.py`:

| Claim | Result |
|---|---|
| Sphere chart $=$ sphere SDF (gap + normal, all depths, $Q\ne I$) | machine precision ($\Delta\text{gap}=0$, $1-\cos<10^{-15}$) |
| **Active set** $\{\text{gap}<0\}$ $=$ true SDF's, star-shaped body | exact (sign identity) |
| Swap detector in the live MPM path (two-sphere collision) | **bit-identical** $v$, impulse, penetration ($\Delta$ between the sdf and chart runs $=0$) |
| Radial normal $\to$ true surface normal at contact | $4.6^\circ$ at 10 % off-surface $\to 0.03^\circ$ at contact |
| Hertz CV-1 contact circle (chart vs SDF) | identical; matches the closed-form lens radius |

**What is NOT claimed (the honest bound).** The radial gap is **not** a Euclidean distance:
$\text{gap}_\text{rad}=\text{gap}_\perp/\cos\alpha+O(\text{gap}^2)$ ($\alpha=$ ray-vs-normal angle), so its
*magnitude* is conservative-large, $\lvert\text{gap}_\text{rad}\rvert\ge\lvert\text{gap}_\perp\rvert$ always
(an exterior point reads a larger gap; a penetrating point a slightly *deeper* penetration — a marginally
stiffer penalty, not a softer one). The sign is exact (hence the active-set equivalence); the *value*
equals the SDF only on the sphere and in the $\text{gap}\to0$ limit. `closest_point_refine_chart` recovers
the true perpendicular gap + surface normal (verification only — the foot jumps across the medial axis,
non-smooth, would leak energy; never used by the integrator).

**Preconditions / scope.** Valid only for **star-shaped** bodies (every ray from $c$ hits the boundary
once). A plane/half-space is not star-shaped about any finite center, so **floors stay on the SDF path**
(`FloorSDF`). Per-particle MPM detection makes multiple disjoint contact *patches* free (they appear as
clusters of penetrating particles — no boundary-arc enumeration), but the *obstacle chart* must still be
star-shaped. The detector is swapped behind one flag: `ContactBody(detector="chart", chart=...)`, dispatched
by `contact_manager.body_gap_normal` (shared by `detect_mpm` and `schwarz_mpm._compute_contact_forces`);
penalty / friction / augmented-Lagrangian forces and the broad-phase culler are unchanged.

**Staging.** Shipped now (Stage 0): *analytic* $\rho$ (sphere, superquadric) — fully provable without
training, as above. Deferred: **Stage 1** a trained neural $\rho_\theta:S^2\!\to\!\mathbb R^+$ (the neural
chart that *replaces* the neural SDF; same `evaluate_gap_chart` path, the projected-gradient step becomes
load-bearing; validate to the §11.3 neural tolerances, not machine precision). **Stage 2** a multi-chart
atlas fallback (`ChartDecoder` + `invert_decoder`, signed-height gap with a Jacobian-metric correction,
transition maps at overlaps) for **non-star-shaped** bodies, gated behind a star-shaped certificate.

---

## 12. Scope & limitations

- **Small-strain, linear-elastic, quasi-static (CV-1..4).** Hertz/Brazilian/C–M are infinitesimal
  half-space/plane theory; the MPM is finite-deformation and frictional. Agreement is
  expected only in the small-load, pre-damage, frictionless-or-monotone limit.
- **Plane stress vs plane strain.** In-plane stresses ($\sigma_t$, center values) are
  identical in both; only $\sigma_{zz}$ and compliances differ ($E\to E/(1-\nu^2)$,
  $\nu\to\nu/(1-\nu)$). State the choice when comparing displacements.
- **Singularities.** Point-load fields diverge as $1/r$; figures clamp $r$ to
  $\sim 1\%R$ and use percentile color limits (the physical contact is a finite Hertz patch).
- **Transition-map machinery: CV-1..4 do NOT exercise it** (symmetric/collinear/90° cases reduce
  to linear superposition of known closed forms). **CV-5 is the case that does** — nonconvex,
  non-collinear, curved-surface contact where a single closest-point projection fails and the
  boundary-chart correspondence enables multi-arc detection. Even there, the *gap* is a single-body
  inverse radial chart (biased vs perpendicular distance), not the SDF replacement.
- **CV-5 is rigid-body** — it tests the contact *kinematics*, not the deformable solver; the radial
  gap requires star-shapedness and radial visibility.
- **Carried corrections.** Eq. 46 factor-4 width; two-body modulus $E^*=E/2(1-\nu^2)$;
  compatibility-fixed Flamant gauge; equibiaxial (not hydrostatic) disc center.

---

**References.** K. L. Johnson, *Contact Mechanics* (CUP, 1985); Timoshenko & Goodier,
*Theory of Elasticity* (McGraw-Hill); Mindlin (1949); Hondros (1959);
Liu & Sun (2020), *CMAME* 369:113168.
