# CV-7 — Formulation, Verification & Plan brief (from workflow w6gbw4c58, audited)

All grounded. Here is the synthesized CV-7 brief with all 12 audit fixes applied.

---

# §11.12 CV-7 — Rough-Joint Contact via Learned Coordinate Charts: Formulation, Verification, and Plan

> **Scope note (read first).** CV-7 spans two distinct geometries and two distinct solvers, which the literature
> drafts repeatedly conflated. They are kept rigorously separate throughout this section:
> - **§11.9 capstone** — the *real Inada granite* 1-D profile, sheared by the height-chart driver
>   (`rock_joint_shear.py`, `profile_chart_2d.py`). Headline numbers: chart reconstruction **2.3 µm** vs SDF
>   **107 µm**; mean asperity angle **19.8° (chart) / 12.5° (SDF) / 19.4° (true)**; **shear strength under-predicted
>   61 %** by the SDF; **dilation barely affected (0.5 %)** because total dilation is set by large-scale waviness the
>   SDF *does* capture.
> - **§11.11 genuine decoder shear** — a *band-limited synthetic* rough surface carried by a trained 3-D
>   boundary-fitted `RoughBlockDecoder`, sheared by chart-FEM (`rock_joint_decoder_shear.py`). Headline numbers
>   (frictionless): **dilatancy under-predicted 98 %** by the level set (0.004 vs 0.171), **shear strength
>   under-predicted 35 %**.
>
> **Never attach the 98 % dilation number to the Inada capstone** (where the measured value is 0.5 %), and never
> attach 61 % strength to the synthetic decoder run. (Audit C1.)

---

## (A) Fourier-Feature Mathematical Formulation

The neural-atlas thesis specialized to a surface-of-a-graph: geometry is a learned *function on the surface's own
parameter*, not an ambient level set. The single design choice that makes this work — and the heart of the CV-7
"atlas beats level set" claim — is **Fourier-feature input encoding**, which hands the network high frequencies
directly and defeats spectral bias.

### A.1 The three banks used in CV-7

There are **three** encodings in the codebase, with different banks. (Audit C4: the 1-D bank is deterministic
geometric, *not* Gaussian-random.)

**(i) 1-D height chart** `NeuralHeight1D` (`profile_chart_2d.py`) — the Inada-capstone chart. *Deterministic
geometric* bank:

$$
\gamma(x)=\big[\cos(2\pi B\,\tilde{x}),\ \sin(2\pi B\,\tilde{x})\big]^{\top}\in\mathbb{R}^{2K},
\qquad
\tilde{x}=\frac{2(x-x_{\mathrm{lo}})}{x_{\mathrm{hi}}-x_{\mathrm{lo}}}-1\in[-1,1],
$$

with $B=\mathrm{geomspace}(0.5,\,f_{\max},\,K)$, $K=192$, $f_{\max}=1500$, width $128$, depth $4$, tanh; a learnable
`base` absorbs DC. Geometric spacing gives **log-uniform (equal-log-decade) coverage of scales** — *not* uniform
coverage of $[0,f_{\max}]$ (Audit C4). The chart maps

$$
z=h_\theta(x)=\texttt{base}+\mathrm{MLP}\big(\gamma(x)\big).
$$

**(ii) 2-D surface chart** `NeuralRho2D`-analog `surface_chart_3d.py` (the §11.10 3-D extension). *Gaussian-random*
bank (Tancik et al. 2020, Eq. 3): $B\in\mathbb{R}^{K\times2}$, $B_i\sim\mathcal N(0,\sigma^2 I_2)$, $\sigma=8.0$,
$K=256$, seed $0$, encoded at $2\pi$:

$$
\gamma(\mathbf u)=\big[\cos(2\pi\,\tilde{\mathbf u}B^{\top}),\ \sin(2\pi\,\tilde{\mathbf u}B^{\top})\big],
\qquad z=h_\theta(\mathbf u)=\texttt{base}+\mathrm{MLP}(\gamma(\mathbf u)).
$$

**(iii) 3-D decoder relief** `RoughBlockDecoder` (`rough_block_decoder.py`) — the §11.11 genuine-shear geometry.
Gaussian bank $B\sim\mathcal N\!\big(0,(k_{\max}/3)^2 I_2\big)$, $k_{\max}=6.0\Rightarrow\sigma=2.0$, $K=20$,
seed $0$, width $64$, depth $3$, and crucially encoded at **$\pi$ (not $2\pi$)** because $\xi_{xy}\in[-1,1]^2$:

$$
\mathrm{relief}(\xi_x,\xi_y)=\mathrm{MLP}\Big(\big[\cos(\pi\,\xi_{xy}B^{\top}),\ \sin(\pi\,\xi_{xy}B^{\top})\big]\Big).
$$

The boundary-fitted ChartDecoder map ($\xi\in[-1,1]^3\to\mathbb R^3$) is

$$
\mathbf x=D(\boldsymbol\xi)=\mathbf s+\xi_x\mathbf t_1+\xi_y\mathbf t_2+\xi_z\mathbf n
+\mathrm{relief}(\xi_x,\xi_y)\cdot\mathrm{ramp}(\xi_z)\cdot\hat{\mathbf n},
\qquad
\mathrm{ramp}(\xi_z)=\begin{cases}\tfrac{\xi_z+1}{2}&\text{rough TOP}\\[2pt]\tfrac{1-\xi_z}{2}&\text{rough BOTTOM}.\end{cases}
$$

The chart-Jacobian is

$$
J=\frac{\partial D}{\partial\boldsymbol\xi}
=\Big[\,\mathbf t_1+\tfrac{\partial\mathrm{relief}}{\partial\xi_x}\,\mathrm{ramp}\,\hat{\mathbf n},\;\;
\mathbf t_2+\tfrac{\partial\mathrm{relief}}{\partial\xi_y}\,\mathrm{ramp}\,\hat{\mathbf n},\;\;
\mathbf n+\tfrac{\mathrm{relief}}{2}\,\hat{\mathbf n}\,\Big]\in\mathbb R^{3\times3},
$$

since $\partial\,\mathrm{ramp}/\partial\xi_z=\pm\tfrac12$ is **constant** — it is the third column's
$\mathrm{relief}/2\cdot\hat{\mathbf n}$ term that couples the relief into $\det J$ and sets the foldover amplitude
limit (Audit C10). `verify_decoder` checks $\det J>0$ on every element and *reports* (does not enforce) the worst
$\kappa(J)$ before any contact is attempted.

### A.2 Spectral-bias rationale

A plain-coordinate MLP $h_{\text{plain}}(x)=\mathrm{MLP}(x)$ learns frequencies in order of increasing wavenumber
(Rahaman et al. 2019): its neural-tangent kernel is Laplace-like, with spectral density decaying in the high-frequency
tail, so sharp asperities are smoothed or never learned. This is exactly the representation an **ambient neural SDF is
built from**, hence its asperity-smoothing failure (the `plain=True` ablation in all three modules reproduces it).

Fourier prepending (Tancik et al. 2020) replaces the Laplace kernel with a stationary band-limited kernel. For the
**deterministic 1-D geometric bank**, the correct linear-readout kernel is (Audit C4 — the draft's spurious
$+\sin(\cdot)$ term breaks stationarity and is dropped):

$$
\Theta(x,x')=\frac{1}{K}\sum_{k=1}^{K}\cos\!\big(2\pi f_k\,(x-x')\big).
$$

For the **2-D Gaussian-random bank**, $B_i\sim\mathcal N(0,\sigma^2 I_2)$ gives the stationary RBF/Bessel kernel

$$
\mathbb E_B\big[\cos(2\pi(\mathbf u-\mathbf v)^{\top}B)\big]=J_0\big(2\pi\sigma\|\mathbf u-\mathbf v\|\big),
$$

flat in spectral density up to a cutoff $\sim2\pi\sigma$. All sub-cutoff frequencies are learned in parallel.

### A.3 Measured numbers (kept honest and separated)

| Geometry / solver | Method | Reconstruction RMSE | Source |
|---|---|---|---|
| **Inada profile** (§11.9, 1-D height-chart shear) | **Fourier height chart** | **2.3 µm** (66k params) | manual §11.9 lines 706–708 |
| Inada profile | Ambient neural SDF (level set) | **107 µm** (83k params, **47×** worse) | manual §11.9 |
| **Synthetic band-limited** (§11.11, 3-D decoder shear) | **Fourier ChartDecoder** | **< 8 %** of surface RMS (gate; ~2 %) | `rough_block_decoder.py` line 145 |
| Synthetic band-limited | Plain-coordinate decoder | smoothed (spectral bias) | `plain=True` ablation, line 65 |

The synthetic decoder target is `band_limited_rough_surface(n_modes=6, k_min=0.6, k_max=2.2)` — **6 modes**, not the
48/50 the drafts claimed (Audit C3). The ~2 %/<8 % reconstruction is the `rough_block_decoder.py` self-test gate
(`rmse/rms_surf < 0.08`), **not** the `surface_chart_3d.py` synthetic self-test (which asserts a different `<8 %` on
an unrelated surface). Cite each number to its own module (Audit C3).

---

## (B) Interpenetration Prevention (Gap, KKT, Penalty/AL)

### B.1 Gap function

The §11.11 decoder driver detects against the **rigid mating rough surface** $z_{\text{up}}(X,Y)=z_p+h(X-u_x,Y)$ at
the deformed top-node positions $(X,Y,Z)$, using **finite differences** (not autograd, not the height chart's
`surface_normal`) for the surface gradient (Audit C8):

$$
h_x=\frac{h(X-u_x+\varepsilon,Y)-h_0}{\varepsilon},\quad
h_y=\frac{h(X-u_x,Y+\varepsilon)-h_0}{\varepsilon},\quad \varepsilon=10^{-4},
$$
$$
\sec=\sqrt{1+h_x^2+h_y^2},\quad
\mathbf n=(h_x,h_y,-1)/\sec\ \ (\text{upper-body outward/downward normal},\ n_z<0),
$$
$$
g_{\text{vert}}=z_{\text{up}}-Z,\qquad g_N=g_{\text{vert}}\cdot(-n_z).
$$

The 1-D height-chart contract (`profile_chart_2d.py`) is the analogous graph evaluation with the *small-slope-exact*
perpendicular projection $g_n\sim g_{\text{vert}}/\sqrt{1+h'^2}$. This is a **biased proxy, not the closest-point
distance** (Audit C9): the slope/normal fidelity is the contact-relevant story, the gap is a small-slope proxy.

### B.2 Signorini (KKT) conditions

$$
g_N\ge0,\qquad t_N\ge0,\qquad g_N\,t_N=0,
$$

with $t_N=\varepsilon_n\,\langle-g_N\rangle_+\ge0$ the (non-negative) normal-pressure magnitude. The force on the
penetrating lower node is (Audit C6 — sign corrected to match the code):

$$
\mathbf F_N=f_n\,\mathbf n,\qquad f_n=\varepsilon_n\,\langle-g_N\rangle_+\,A_{\text{top}}\ge0,
$$

where $\mathbf n$ is the upper body's **downward** outward normal ($n_z<0$). Hence $F_{N,z}=f_n n_z<0$ pushes the
lower node *down/out* of the upper body, and the platen feels the upward reaction
$-\sum f_z>0$ (`normal_force`). The draft's $\mathbf F_N=-f_n\mathbf n$ was wrong.

### B.3 Penalty and Augmented Lagrangian

Penalty stiffness and tributary area (verified against `rock_joint_decoder_shear.py`):

$$
\varepsilon_n=\frac{20\,E}{h},\qquad A_{\text{top}}=\frac{(2L)^2}{|\text{top nodes}|},\qquad
f_n=\varepsilon_n\,\langle-g_N\rangle_+\,A_{\text{top}}.
$$

> **Audit C7.** The CV-7 driver is an **implicit quasi-static Newton** solve — there is *no time step*. The CFL bound
> $\Delta t\le\text{safety}\sqrt{m_{\min}/\varepsilon_n}$ (`contact_stable_dt`, `penalty.py`) belongs to the
> **explicit MPM** path only and is *not* a CV-7 constraint. It is documented in §11.7/§D as MPM-only.

The Newton convergence test *is* in the driver: $\|R_{\text{free}}\|<\text{tol}\,(1+\varepsilon_n A_{\text{top}})$.

For tighter non-penetration without $\varepsilon_n\to\infty$, the augmented-Lagrangian (Uzawa) path
(`augmented_lagrangian.py`, the separate MPM-side API) keeps a persistent multiplier:

$$
p_{\text{aug}}=\max(0,\ \lambda-\varepsilon_n g_N),\qquad \lambda^{k+1}=\max(0,\ \lambda^k-\varepsilon_n g_N(u^{k+1})).
$$

---

## (C) Surface Traction (Normal + Coulomb Friction + FEM Assembly)

### C.1 Normal traction

$$
t_N=\varepsilon_n\,\langle-g_N\rangle_+\ \ [\text{Pa}],\qquad
\mathbf F_N^i=t_N\,A_{\text{top}}\,\mathbf n^i=f_n^i\,\mathbf n^i .
$$

### C.2 Friction — two *different* laws (do not conflate)

> **Audit C5.** The §11.11 chart-FEM driver is **quasi-static**; there is no velocity, no $\varepsilon_T$, no
> $v_T/\sqrt{|v_T|^2+\varepsilon_T^2}$. Friction is a **prescribed-direction, full-slip Coulomb drag**: the upper
> surface slides $+x$, so the slip direction is the fixed world vector $\hat{\mathbf t}=[1,0,0]$ projected onto the
> tangent plane:

$$
\mathbf t_{\text{proj}}=\frac{\hat{\mathbf t}-(\hat{\mathbf t}\!\cdot\!\mathbf n)\mathbf n}
{\lVert\hat{\mathbf t}-(\hat{\mathbf t}\!\cdot\!\mathbf n)\mathbf n\rVert},
\qquad
\mathbf F_T=\mu\,f_n\,\mathbf t_{\text{proj}},\qquad \mu=0.4.
$$

The **regularized-velocity** Coulomb law
$\mathbf f_T=-\mu|f_N|\,v_T/\sqrt{|v_T|^2+\varepsilon_T^2}$ (`friction.py`) is the **MPM** API only — the CONTACT
draft's own grounding admits it is "NOT USED in CV-7." Present it strictly as the separate explicit-MPM path (§D).

Total per-node contact force: $\mathbf F_{\text{contact}}=\mathbf F_N+\mathbf F_T$ (driver line 94).

### C.3 FEM assembly (verified against the driver)

Residual and Newton step:

$$
R(u)=K u-f_{\text{ext}},\quad f_{\text{ext}}=f_{\text{contact}},\qquad K_{\text{total}}\,\Delta u=-R .
$$

Penalty contact tangent on active top nodes (vertical/normal approximation):

$$
K_c[3i+2,\,3i+2]\mathrel{+}=\varepsilon_n A_{\text{top}},\qquad
K_{\text{total}}=K_{\text{elastic}}+K_c .
$$

Armijo line search ($c=10^{-4}$): accept if $\|R(u^{k+1})\|<(1-c\,\text{step})\|R(u^k)\|$. Elastic $K$ from
`ChartVectorFEMSolver.tangent_stiffness` with `make_linear_elastic_small_strain`. Free DOFs exclude the fixed bottom
face; solved with `scipy.sparse.linalg.spsolve`. **All of $\varepsilon_n=20E/h$, $A_{\text{top}}$, $\mu=0.4$,
$g_N=g_{\text{vert}}(-n_z)$, $F_n=f_n\mathbf n$, the line-search constant, the convergence test, and the contact
tangent match `rock_joint_decoder_shear.py` exactly** (Audit "CORRECT" list).

---

## (D) Transition-Map Contact Verification + Feasibility

### D.1 Formulation — the height-chart query *is* the transition map

For a single-valued surface $z=h(x)$ the boundary chart is the graph $\varphi(x)=(x,h(x))$, so the domain coordinate
$x$ is automatically the inverse-chart coordinate — no inversion, no Newton refinement, no closest-point projection.
The gap is a direct evaluation $g_{\text{vert}}=z-h(x)$ and the normal is the chart-Jacobian normal
$\mathbf n=(-h',1)/\sqrt{1+h'^2}$ (autograd, `height_and_grad`). This is the *single-valued* simplification of the
radial detector of CV-5 (`radial_chart_2d.py`, $\rho:S^1\!\to\!\mathbb R^+$, $g=r-\rho(\psi)$).

> **Honesty (Audit C9).** "One-shot gap + normal" is true *for the chart query*, but the gap fed to the solver is the
> small-slope vertical proxy, **not** the perpendicular distance. The fidelity story is **slopes/normals**
> (19.8° vs 12.5°), not the gap magnitude. `closest_point_refine_chart` (`chart_gap.py` line 306) recovers the true
> perpendicular distance for *verification only* — never in the integrator (its foot jumps across the medial axis,
> leaking energy).

### D.2 What is actually wired (Audit C8)

The §11.11 decoder driver computes the gap **inline** against a hard-coded analytic
`band_limited_rough_surface` via finite differences. It does **not** route through any `ContactBody(detector="chart")`
/ `contact_manager.body_gap_normal` dispatch — that chart-detector dispatch is a **separate, not-yet-exercised path**.
Claims of "already wired" / "bit-identical chart-vs-SDF in the contact manager" are aspirational and must be marked as
planned (see Phase 3).

### D.3 Patton anchor — the L1 closed form (Audit C2)

For two mating sawtooth faces of asperity angle $i$ under plain Coulomb (base angle $\phi_b$, $\mu=\tan\phi_b$):

$$
\boxed{\ \mu_{\mathrm{app}}=\tan(\phi_b+i)=\frac{\tan i+\mu}{1-\mu\tan i}\ }
$$

The denominator is $1-\mu\tan i$ (matches manual line 695), **not** the draft's broken $(1/\mu)\tan i$. Numerical
check ($\phi_b=20^\circ,\ i=15^\circ,\ \mu=\tan\phi_b$): true $\tan(35^\circ)=0.7002$; the broken form gives $0.8584$.
Measured: **0.00 %** error (real closed form — only plain Coulomb + geometry go in, Patton comes out), dilation rate
$=\tan i$ to within 1 %.

### D.4 Feasibility

- **Well-posed** for single-valued surfaces (every $x\mapsto$ one $h$); the real Inada topography is single-valued, so
  the height-chart covers the capstone. Overhangs/re-entrant geometry need the full transition-map atlas (out of scope).
- **Star-shapedness is *not* required** — that is the CV-5 radial constraint; single-valuedness is the relevant,
  weaker condition here.
- **Fourier-frequency tuning is required** ($f_{\max}$, $\sigma$ chosen from the measured power spectrum). This is a
  *tunable, transparent* ceiling — unlike the *intrinsic* spectral bias of an ambient SDF.

---

## (E) Staged Plan to Make CV-7 a Success Story

> **Status discipline (Audit C11, C12).** Phases 0 and 5 are **NOT BUILT**. All `<0.1 %` residual, `<2 %`
> energy-balance, and MPM "within 10 %" figures below are **targets contingent on Phase 0/5**, not measured results.
> Currently measured: friction Newton residual **0.3–1 %**, cyclic energy balance **~1.5×** (frictionless converges
> cleanly). Do not present post-Phase-0 targets as achieved.

| Phase | Goal | Acceptance (target unless noted) | Key risk |
|---|---|---|---|
| **0. Semismooth-Newton / AL solver** *(critical path, not built)* | Resolve the 0.3–1 % friction residual from Coulomb non-smoothness | Monotone residual → **<0.1 %**; cyclic energy balance **∈[0.95,1.05]** (vs current ~1.5×); Patton stays **0.00 %** | Clarke-Jacobian looseness can stall; AL adds artificial damping. Prototype on 2-D Patton first |
| **1. Two-block rough FEM** | Extend one-deformable-on-rigid → two mutually deformable decoders | MMS rate **1.95–2.05** on both decoders (patch machine-zero); $\det J\in[0.90,1.10]$ coupled; energy balance **∈[0.98,1.02]** | Interface quadrature/normal misalignment. Start flat (identity decoder), swap rough in one at a time |
| **2. Full multi-scale Inada + band-limit study** | Coarse/mid/fine spectral cutoffs; dilatancy-vs-roughness scaling law | Reconstruction **<5 % RMS** all cutoffs; $\det J>0.85$; smooth dilatancy(RMS, slope, Hurst) curve | Over-refinement degrades $\det J$. Adaptive refinement near asperities; monitor FFT of $h$ |
| **3. Transition-map detection wired into FEM contact** | Route chart gap/normal through the contact-manager dispatch | Active sets identical to SDF on 100k points (sign machine-precise, $<10^{-5}$ magnitude); per-query $<2\times$; smoother normals at asperity tips | Nested decoder inversion cost/convergence. Cache inverse + interpolate; test on identity decoder first |
| **4. Cyclic CNL/CNS + energy accounting** | Constant-load / constant-stiffness protocols; full energy ledger | $W_{\text{ext}}=W_{\text{fric}}+\Delta U_{\text{el}}+W_{\text{dil}}$ to **<2 %** all protocols; closed friction loops; Plesha degradation monotone | Feedback overshoot → spurious energy jumps. Implement one ledger term at a time on flat interface |
| **5. Finite-deformation ChartMPM** *(not built)* | Explicit dynamic rough-joint shear; quasi-static cross-check | Dilatancy & $\mu_{\text{app}}$ **within 10 %** of FEM at CFL $\ll0.3$; momentum balance **<0.1 %**; one animation | $\det J$ hourglassing on curved domains; noisy per-particle active set. Fine penalty + small $dt$; identity-decoder first |
| **6. Publication figure + narrative** | Atlas vs level set vs flat benchmark, single page | One 3×3 grid (reconstruction, stress fields, cyclic loops, dilatancy trends); all numbers cited per their correct geometry | Dense figure / over-broad narrative. Lead with physics, defer methods to appendix |

### E.1 Paper narrative (honest)

**Title.** *Contact Mechanics on Rough Geometry via Learned Coordinate Charts: A Direct-Shear Study of Real Rock
Joints.*

**Thesis.** The apparent friction angle $\mu_{\text{app}}=\tan(\phi_b+i)$ is *emergent* from asperity geometry, not
imposed. A boundary-fitted neural chart resolves asperity slopes; an ambient neural SDF smooths them (spectral bias),
under-predicting strength and dilatancy.

**Headline results, attributed to their correct experiment (Audit C1):**
- **Method anchor (Patton, sawtooth):** $\mu_{\text{app}}=\tan(\phi_b+i)=(\tan i+\mu)/(1-\mu\tan i)$ to **0.00 %**.
- **Real Inada capstone (§11.9):** chart **2.3 µm** vs SDF **107 µm** (47×); slopes **19.8°/12.5°** (true 19.4°);
  **strength −61 %** with the SDF; **dilation barely affected, −0.5 %**.
- **Genuine synthetic decoder shear (§11.11):** **dilatancy −98 %** (frictionless, 0.004 vs 0.171) and **strength
  −35 %** with the level set.

**Honest limitations (must appear in the paper):** band-limited roughness (6 modes at the resolved decoder scale);
single decoder-pair realization; single-valued geometry only (no overhangs); semismooth-Newton **not yet** implemented
(current friction residual 0.3–1 %, cyclic energy balance ~1.5×); MPM cross-check (Phase 5) and the chart-detector
contact-manager dispatch (Phase 3) **not built**; the contact gap is a small-slope vertical proxy (slopes/normals are
the fidelity metric, not the gap magnitude).

---

### Audit fixes applied (all 12)
- **C1** dilation numbers split: 0.5 % stays §11.9 Inada; 98 %/0.004-vs-0.171 stays §11.11 synthetic decoder (frictionless). **C2** Patton denominator corrected to $1-\mu\tan i$ (with the 0.7002-vs-0.8584 check). **C3** "6 modes" (not 48/50); 2.2 % cited to `rough_block_decoder.py`, not the surface_chart self-test. **C4** 1-D bank labeled deterministic geometric / log-uniform; NTK corrected to $\frac1K\sum\cos(2\pi f_k\Delta)$. **C5** CV-7 FEM friction = prescribed-direction full-slip Coulomb; velocity-regularized law marked MPM-only. **C6** $\mathbf F_N=f_n\mathbf n$ sign fixed. **C7** CFL bound removed from CV-7, marked MPM-only. **C8** decoder driver uses inline finite-difference analytic surface; chart-detector dispatch marked not-yet-exercised. **C9** gap is a small-slope proxy, slopes are the fidelity story. **C10** $\partial\mathrm{ramp}/\partial\xi_z=\pm\tfrac12$ constant → $\mathrm{relief}/2\cdot\hat{\mathbf n}$ Jacobian term; $\kappa$ reported not enforced. **C11/C12** Phases 0 and 5 and all `<0.1 %`/`<2 %`/`within-10 %` numbers marked as targets, not measured (current: 0.3–1 % residual, ~1.5× balance).

**Files referenced (all absolute):** `/Users/wsun/Documents/Software/neural_atlas_MPMcontact/benchmarks/contact/cv_numerical/rock_joint_decoder_shear.py`, `.../solvers/contact/profile_chart_2d.py`, `.../solvers/fem/rough_block_decoder.py`, `.../solvers/contact/penalty.py`, `.../solvers/contact/friction.py`, `.../solvers/contact/chart_gap.py`, `.../docs/contact_verification_manual.md` (§11.9 lines 693–731, §11.10, §11.11 lines 855–909).