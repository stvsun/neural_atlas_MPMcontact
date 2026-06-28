All section labels and line ranges in the change lists are valid against the current manuscript. I have verified the load-bearing claims (module docstrings, headline numbers, the closest-point/convex-cost/arclength-smearing identities, section locators). I can now write the consolidated plan.

---

# CMAME Manuscript Reframing Plan — Optimal-Transport Transition Maps for Contact

**File:** `/Users/wsun/Documents/Softwares/neural_atlas_MPMcontact/.claude/worktrees/unruffled-pike-35fc89/paper/main.tex`

## New thesis statement

> We formulate computational contact as an **optimal-transport coupling of two boundary measures**: the chart transition map $\tmap=\map{B}^{-1}\circ\map{A}$ is the Brenier map between the slave and master boundary measures, so the gap and the contact traction follow as continuous, mass-preserving **fields** assembled mortar-consistently, not as node-collocated tributary-lumped penalties. One framework spans two regimes — the closest-point projection is the convex-cost / partial-contact limit, the global arclength-monotone rearrangement the conforming / rough-surface limit — which resolves the "not a projection" tension and unlocks a genuine **two-body deformable–deformable** boundary-value problem (symmetric SPSD 4-block tangent, FD-verified). The validated "representation, not the enforcement scheme, controls fidelity" rough-joint result (level set under-predicts strength 61% / dilatancy 98%) is retained as the empirical payoff of the same coupling on a conforming interface.

**Recommended title (use in `.tex`):**
> **Optimal-transport transition maps for computational contact: a mortar-consistent measure coupling of deformable boundaries on a neural atlas**
> running: *Optimal-transport measure-coupling contact*

**Alternates (cover note only):**
- (A) *From closest-point projection to optimal transport: a unified measure-coupling transition map for two-body contact on a neural atlas* — foregrounds the unification that resolves the current "not a projection" line.
- (B) *Contact as a measure coupling: optimal-transport transition maps between deformable boundaries on a neural atlas.*

---

## Section-by-section plan (manuscript order)

### Title & front matter (lines 103–135)
*Role: announce OT measure coupling as the formulation thesis in sentence one; rough-joint = payoff, not lead.*
- **retitle** (103–105): replace with the OT-led title above; update `\titlerunning`.
- **add-citation** keywords (133–134): reorder to lead with formulation — `optimal transport \and measure coupling \and mortar method \and transition map \and coordinate atlas \and neural signed-distance function \and Fourier-feature charts \and rock-joint dilatancy`.

### Abstract (lines 117–132)
*Role: OT formulation is sentence one; two-body deformable result is the new headline; Inada stays as payoff.*
- **reframe** opener (118–121): demote representation to a supporting clause; open with *"We formulate computational contact as an optimal-transport coupling of two boundary measures… the contact correspondence is the Brenier map that minimises a transport cost, and the gap and traction follow as continuous, mass-preserving fields… On a learned atlas this coupling is read from $\tmap=\map{B}^{-1}\circ\map{A}$, the optimal-transport map between the boundary measures of $A$ and $B$."* Keep one representation sentence.
- **revise-text** (122–124): recast the chart-composition sentence as the discrete measure coupling + two OT limits: *"$f_I=\sum_q w_q N_I(p_N\nbf+\bm t_T)$, master interpolation a partition of unity along the correspondence, $\sum_K(N_K\!\circ\!\chi)=1$ — the OT mass marginal — so a constant pressure transmits exactly across a non-matching interface (patch test, $10^{-16}$)… for a localised convex indenter the convex-cost OT map degenerates to the closest-point projection ($g=\norm{\bm p-\bm c}-R$); for two mated rough surfaces it is the global arclength-monotone (Brenier) rearrangement; the closest-point limit is required for partial contact, where the arclength map smears the load."*
- **add-result** (after 124): two-body headline — *"two-sided, transpose-coupled, symmetric positive-semidefinite four-block consistent tangent (FD-verified to $3.4\times10^{-11}$)… deformable Hertz between two elastic blocks: $a$ to $2.75\%$, $p_0$ to $5.82\%$ (mesh-converged), patch test $1.4\times10^{-16}$, force balance $1.3\times10^{-19}$; a $3\times3$ elastic disc array: centre mean stress $0.58\%$, force balance $3.7\times10^{-15}$."*
- **revise-text** (124–132): keep Fourier-feature + matched-baseline sentence intact but reorder to *follow* the OT block; keep the variational-sequence + six-benchmark + 61%/98% payoff sentence, re-leading with *"As an application that turns on geometric fidelity…"* — numbers unchanged.

### Section 1 — Introduction, Related Work, Contributions (lines 138–355)
*Role: establish OT measure coupling as the central formulation thesis; representation thesis becomes support; add the OT related-work paragraph and the two-body contribution.*
- **reframe** (146–148, 164–166): split the single claim into two coupled claims — (i) the transition map is the Brenier map / contact is a measure coupling yielding continuous mass-conserving gap+traction fields; (ii) representation, not enforcement, controls fidelity. Introduce $\tmap$ as "the optimal-transport coupling of two learned boundary measures." Keep 61%/98% verbatim.
- **revise-text** (154–159): after the $(\gn,\nbf)$ sentence add *"Reading $(\gn,\nbf)$ independently at each slave node is the node-to-segment choice; we instead read it from an optimal-transport coupling of the two boundary measures, so $\gn$ is a continuous field carrying a mass marginal."*
- **reframe** "Closest-point projection and its deficiencies" (220–230): keep the documented fragilities (cite `taylor1991patch`, `zavarise2009nts`, `zavarise2009patchtest`); replace the "NOT a projection" sentence with: closest-point projection *is* the convex-cost limit of $\tmap$ ($g=\norm{\bm p-\bm c}-R$), arclength-monotone the conforming limit; the marginal forbids the per-node collapse that fails the patch test. Recast the `liu2020implicit` single-foot note as "the measure-coupling reading removes the single-foot collapse."
- **add-subsection** (after 230): `\paragraph{Optimal transport and measure-based contact.}` — Monge–Kantorovich + Brenier polar factorization → unique mass-preserving gradient-of-convex map; 1-D monotone rearrangement (McCann); monographs Villani/Santambrogio; connect to mortar / dual-multiplier (`puso2004mortar`, `wohlmuth2011variational`) as the same variationally-consistent family; unbalanced OT (`chizat2018unbalanced`) as the active-set screen; the discrete coupling is a mortar assembly whose master interpolation is the OT mass marginal.
- **revise-text** "Smoothing the master surface…" (232–249): change "the pairing follows from chart composition and inversion rather than any projection" → "follows from the optimal-transport coupling of the two boundary measures (of which the covariant closest-point projection is the convex-cost limit)…"; add half-clause that OT supplies the correspondence mortar builds by segmentation.
- **reframe** Contributions list (315–336): reorder —
  - **C1 (lead):** contact detection as OT measure coupling read from $\tmap$; Brenier map; gap+traction continuous fields with mass marginal; closest-point and monotone-rearrangement as the two limits.
  - **C2:** mortar-consistent assembly $f_I=\sum_q w_q N_I(p_N\nbf+\tbf_T)$, $\sum_K(N_K\!\circ\!\chi)=1$, patch test $1.4\times10^{-16}$, node-lumped penalty fails.
  - **C3 (new headline):** two-body deformable–deformable BVP; symmetric SPSD 4-block tangent $K_{ms}=K_{sm}^{\top}$ FD-verified; CV-8 ($a$ 2.75%, $p_0$ 5.82%, force balance $\sim10^{-19}$) + CV-9a (mean 0.58%, force balance $3.7\times10^{-15}$).
  - **C4:** Fourier-feature charts (kept). **C5:** strong–weak–Galerkin + conservative normal / sign-exact radial gap (kept). **C6:** verification ladder + Inada payoff (kept). Keep the closing "chart and level set interchangeable at the force layer."
- **revise-text** (326–328): clarify the $1/\cos\alpha$ enlargement is the *single-body radial chart* property; *"the two-body measure-coupling gap $\gn=(\xbf_s-\xbf_m)\cdot\nbf$ uses the true surface normal and is unbiased."*
- **revise-text** organisation (347–354): name OT structure in the Sec. `\ref{sec:tmap}` sentence; add two-body Hertz + array to the Sec. `\ref{sec:verif}` sentence.

### Section 2 — The neural coordinate atlas (lines 358–451)
*Role: the chart carries the boundary MEASURE on which the OT map lives; the $(d{-}1)$-dim domain makes the coupling a closed-form 1-D transport. Light touches only.*
- **add-text** (after `\eqref{eq:chart}`, ~360–385): one sentence — *"each chart carries a boundary measure $\mu_X=(\map{X})_\#\,d\theta$, the object the contact pairing of Sec.~\ref{sec:tmap} couples between two bodies."*
- **add-text** (400–413): append to the radial/height/decoder sentence — because each is a scalar on a 1-D parameter line, the boundary measures are coupled by a *closed-form* 1-D OT map (monotone rearrangement $F_B^{-1}\!\circ F_A$), which keeps the coupling tractable.
- **revise-text** (426–432): change "turning a coordinate change into a contact-pairing operator." → add *"Sec.~\ref{sec:tmap} identifies this pairing as an optimal-transport (Brenier) coupling of $\mu_A,\mu_B$, of which $\tmap=\pi_B\circ\map{A}$ is the closest-point / convex-cost branch."*
- **revise-text** caption Fig. `atlas-extension` panel (b) (437–449): add parenthetical *"(the non-uniform reparametrisation is the optimal-transport rearrangement of $\mu_A$ onto $\mu_B$)."*
- **add-text** Remark `cor:capacity` (503–509): the dimension drop also reduces the coupling to a 1-D transport with a closed-form solution.

### Section 3 — Fourier-feature representation (lines 453–509)
*Role: Fourier features make the boundary measure sharp; smoothing the chart smooths the transported measure.*
- **add-text** (~463–476): one clause to the granite-profile sentence — *"since $\mu_X$ inherits $h_X'$ through its arclength density $\sqrt{1+\abs{h_X'}^2}$, a low-passed chart transports a smoothed measure and under-resolves the contact correspondence."* (Mechanistic link between the §11 payoff and the OT formulation.)

### Section 3/govern — Governing equations & variational formulation (lines 512–624)
*Role: carry OT from prose into the weak form; $\delta W_c$ is a measure coupling, gap variation is two-sided through $\chi$.*
- **reframe** (after `\eqref{eq:weak}`, 568–576): add a sentence naming the coupling — a correspondence $\chi:\Gamma_c^A\to\Gamma_c^B$ carries the slave measure onto the master measure mass-preservingly; $\tmap$ is one such correspondence; cite `wohlmuth2011variational`, `puso2004mortar`, `brenier1991polar`, `villani2009optimal`.
- **add-subsection** (between 576–578, "Contact as a measure coupling on the boundary"): boundary measures $d\mu^\alpha=dA$; transport plan with marginals; quadratic-cost optimal plan = Monge/Brenier map; $\gn=(\xbf-\chi(\xbf))\cdot\nbf=\partial_n f$ (Kantorovich potential); displayed two-limit case split (closest-point vs arclength-monotone); note arclength transports all slave mass onto all master mass → smears partial contact.
- **add-equation** (`\eqref{eq:contact-vw}`, 762–766): two-sided gap variation $\delta\gn(\xbf)=\nbf\cdot(\delta\ubf^A(\xbf)-\delta\ubf^B(\chi(\xbf)))$ — master test displacement at the transported point; note slave/master are transpose-coupled images of one traction field (continuum form of the symmetric two-body tangent).

### Section 4 — Formulation of frictional contact by the transition map (lines 625–853) — MAJOR REWRITE
*Role: the mathematical core. Define $\tmap$ as the OT/Brenier map; closest-point and arclength-monotone as the two limits; recast $\delta W_c$ as a discrete measure coupling; add the two-body 4-block tangent.*
- **reframe** opening (625–646): introduce $\tmap$ as the OT map of the two boundary measures; redirect the projection critique to pathologies of the *unconstrained pointwise* search that the mass-marginal repairs; state closest-point = convex-cost limit, monotone rearrangement = conforming limit.
- **add-subsection** `\subsection{The transition map as an optimal-transport coupling}\label{sec:tmap:ot}` (insert ~646/648): (i) boundary measures $d\mu_A=\sqrt{1+h_A'^2}\,dx$; (ii) Monge problem, quadratic cost, Brenier theorem → unique monotone map; (iii) 1-D Brenier map $\tmap=F_B^{-1}\circ F_A$, marginal $F_B(\tmap(\theta_A))=F_A(\theta_A)$; (iv) displayed two-limit case split; close with $\gn=\partial_n f$, $\grad f=\xbf_s-\tmap(\xbf_s)$.
- **revise-text** `\eqref{eq:tmap-def}` (662–671): keep equation; replace "the same operation an atlas uses to pass between two overlapping charts" with "$\map{B}^{-1}$ supplies whichever transition map the regime selects: closest-point $\pi_B$ for partial contact, monotone rearrangement for a conforming interface; both are limits of the OT coupling of \ref{sec:tmap:ot}."
- **add-equation** at `\eqref{eq:radial-gap}` (683–686) + Remark `prop:active-set` (737–742): identify the radial gap as the closest-point OT limit; state the arclength-smearing failure mode for partial contact explicitly.
- **add-equation** at `\eqref{eq:contact-vw}` (763–787): the discrete measure-coupling force $\fbf_I=\sum_q w_q N_I(\theta_A^{(q)})(p_N\nbf+\tbf_T)|_q$ and the transpose-coupled master reaction $-(N_K\circ\chi)(\dots)$; displayed $\sum_K(N_K\circ\chi)\equiv1$ as the discrete OT marginal → exact constant-pressure transmission.
- **add-subsection** `\subsection{Contact as a measure coupling}\label{sec:weak:mortar}` (after 787): (i) consistent Galerkin force $\fbf_I=\sum_J M_{IJ}\tbf_J$, $M_{IJ}=\int N_I N_J\,ds$, per-P1 $M=\frac{L}{6}[[2,1],[1,2]]$ vs diagonal tributary lumping; (ii) consistent tangent $\bm K_c=\sum_J M_{IJ}\varepsilon_n(\nbf_J\otimes\nbf_J)$ with nonzero adjacent off-diagonal; (iii) patch-test proposition (PoU + OT marginal + Gauss exactness ⇒ machine precision, measured $1.4\times10^{-16}$, 67× tighter than lumped); (iv) dropped $p_N\,\partial\nbf/\partial\chi$ = small-rotation approximation, converged solution tangent-independent.
- **add-subsection / move-from-futurework** `\subsection{Two deformable bodies}\label{sec:tmap:twobody}`: two-sided gap $\xbf_m(\chi)=\sum_K(N_K\circ\chi)\xbf_K$; variations $N_I\nbf$, $-(N_K\circ\chi)\nbf$; symmetric SPSD 4-block tangent with $\bm K_{ms}=\bm K_{sm}^{\top}$ (the never-before-assembled master-coupling block); verification SymPy $dR/du\equiv K$, FD $3.45\times10^{-11}$, force balance $4.4\times10^{-16}$; forward-pointer to CV-8/CV-9a with honest caveats. Converts the 902–904 concession into a delivered result. Reference `solvers/contact/measure_coupling/two_body.py`.
- **revise-text** `\eqref{eq:contact-tangent}` (767–787): present the per-node tributary tangent as the *lumped diagonal* of the consistent measure-coupled tangent ($A\to M_{IJ}$).
- **revise-text** Algorithm `alg:contact-detection` (823–851): detector returns $(\gn,\nbf)$ **plus** the correspondence $\chi$ and host PoU weights $(N_K\circ\chi)$, $\sum_K=1$; master inversion realises the selected OT limit; for two deformable bodies the scan is two-sided, emitting the master reaction via $-(N_K\circ\chi)$.

### Section 5 — Finite element formulation (lines 854–926)
*Role: the discrete contact element IS the discrete measure coupling; replace node-lumped penalty with consistent mortar element; add two-body 4-block tangent.*
- **revise-text** section lead (854–862): replace "node-to-surface" with segment-to-segment mortar coupling (Gauss-integrated nodal tractions, consistent non-lumped mass).
- **add-equation** `\eqref{eq:contact-residual}`–`\eqref{eq:contact-block}` (906–924): consistent mortar element $f_I=\sum_J M_{IJ}\tbf_J$, $M=\frac{L}{6}[[2,1],[1,2]]$, $\bm K_c=\sum_J M_{IJ}\varepsilon_n(\nbf_J\otimes\nbf_J)$; node-lumped penalty recovered by mass lumping $M_{IJ}\to A_I\delta_{IJ}$.
- **add-result** (after `\eqref{eq:contact-block}`, 919): patch-test paragraph — $\sum_K(N_K\circ\chi)=1$, line-load error $3.6\times10^{-15}$, two-body net resultant $3.6\times10^{-17}$, transmit error $1.4\times10^{-16}$; node-lumped sawtooth fails. Cite `wohlmuth2011variational`.
- **add-equation / move-from-futurework** new subsection (after 924, "Two deformable bodies: the symmetric 4-block consistent tangent"): the four blocks verbatim from `two_body.py` (lines 19–22), $K_{ms}=K_{sm}^{\top}$, transpose-coupled forces; rigid-master $K_{ss}$ is the $\partial\xbf_m=0$ restriction. Reference `two_body.py`.
- **add-result** (same subsection): FD $3.45\times10^{-11}$, SymPy $dR/du=K_c$, force balance $4.4\times10^{-16}$, symmetric atol $10^{-12}$, min eig $>-10^{-9}$.
- **add-equation** `\eqref{eq:detector}` (893–897): third detector branch — deformable-master closest-point OT map $(\gn,\nbf)=((\xbf-\chi(\xbf))\cdot\nbf,\nbf)$, $\chi=\pi_B\circ\map{A}$; three branches return the same pair, only the deformable-master branch activates $K_{sm},K_{mm}$.
- **revise-text** dropped-terms note (922–924): two-body dropped terms are $\partial\nbf$ and $\partial\chi$ (correspondence fixed within each solve, recomputed between Newton iterations); converged solution tangent-independent (identical $a_{\mathrm{fem}}$ at 40 and 120 iterations).
- **revise-text** closing (900–904): the mortar treatment is now supplied; two-deformable frictionless solved at full Newton (CV-8/CV-9a); keep honest residual (contact-edge noise; friction still calls for semismooth-Newton).

### Section 6 — Numerical examples (lines 927–1964)
*Role: the CV ladder verifies the OT measure coupling vs conventional node-lumped penalty (OT-vs-conventional table); NEW two-body subsection is the headline; rock joint = conforming/monotone-rearrangement OT limit (payoff).*
- **reframe** preamble (929–952): the constraint solver is fixed across the whole ladder — every case solved by the same discrete OT measure coupling; each CV run head-to-head against conventional node-lumped penalty on identical meshes/loads/tolerances; convex cases = closest-point limit, rough/cusped = monotone-rearrangement limit.
- **add-text** (943–948): append the discrete-measure-coupling formula + patch-test consequence; keep the sign convention.
- **reframe** Hertz (954–976): closest-point/convex-cost OT limit; report **conventional 1.59% vs OT 0.50%** ($a$-relerr, 3.2×), honest floor ~0.4–0.5% from edge slope; advantage is contact-edge mesh resolution, not a tangent effect.
  - **add-equation** at `eq:hertz_a` (961–964): `eq:hertz_ot_limit` — $\chi(p)=\arg\min_y c(p,y)=\pi_B(p)$, $g_N=\norm{p-c}-R$.
- **reframe** Cattaneo–Mindlin (1038–1061): **conventional 11.15% vs OT 0.03%** ($c/a$, 370×, refined $n=3200$) — the suite's strongest like-for-like margin; keep honest note that 11% is coarse-mesh node-quantization of the moving slip front.
- **reframe** nine-disc (1159–1181): four pole contacts = four OT couplings; **conventional 0.11% vs OT 0.077%** (1.4×, smallest margin); soften the "contacts only place the load poles" caveat into a forward pointer to CV-9a where confinement is an output; note the per-pole load $N$ emerges (cross-checked to 5 digits) and D4 symmetry is recovered.
- **revise-text** transition (1269–1273): OT meets/beats conventional across the convex/superposable regime (1.4×–370×); cusp/fractal cases are the second OT regime (monotone-rearrangement / grid-independent).
- **revise-text** superformula (1275–1326): **conventional 1.99e-2 vs OT 5.3e-13** (grid-independent, float64 floor); keep chart-vs-SDF reconstruction (3.8e-3 L vs 8e-3 L); recast multi-arc scan as the OT correspondence reporting multiple disjoint mass-overlap arcs.
- **revise-text** Koch (1422–1435): **conventional frozen vs OT 2.6e-16** (convergent); increments $[3.36e{-}1,7.09e{-}2,8.01e{-}3,5.94e{-}4,2.43e{-}5]$ as demonstrated super-geometric Cauchy convergence; keep $O(1)$ recursive-chart storage.
- **revise-text** performance summary `tab:cv_summary` (1585–1637): add **"Conventional vs OT (same mesh)"** column — CV-1 1.59%→0.50% (3.2×), CV-2 11.15%→0.03% (370×), CV-4 0.11%→0.077% (1.4×), CV-5 1.99e-2→5.3e-13 (grid-indep), CV-6 frozen→2.6e-16 (convergent), CV-7 1.5e-4→3.5e-14 (machine prec). Keep L1-vs-closed-form and chart-vs-SDF columns.
- **add-subsection** `\subsection{Two deformable bodies: the measure coupling as a two-sided contact operator}\label{sec:ex_twobody}` (after 1637): rigid-master ladder never exercises the master tangent; two-sided gap + 4-block tangent ($K_{ms}=K_{sm}^{\top}$, FD 3.45e-11, force balance 4.4e-16); OT-map dichotomy (arclength smears partial; closest-point for convex). **CV-8** deformable Hertz: $a$ 2.75% / $p_0$ 5.82% (nx=192), mesh convergence ($a$: 5.14→4.14→2.96→2.75%; $p_0$: 6.34→5.68→5.91→5.82%), patch test 1.4e-16 (67×), sliding centroid-monotone (max backstep 0.0), force balance 1.27e-19, 52/161 active, $a/W=0.228$. **CV-9a** $3\times3$ array (12 mortar pairs): full Newton relax=1.0 (32 iters, 0 backtracks), mean stress 0.58%, anisotropy 0.20–0.22% (D4), force balance 3.71e-15. Honest residuals (~3–6% edge noise; single-realization, plane-strain CST small-strain, frictionless, rigid outer walls).
  - **add-equation** (`eq:twobody-gap`, `eq:twobody-tangent`, `eq:twobody-force`): two-sided gap + variations, 4-block tangent, transpose-coupled forces with $\sum_K(N_K^-\circ\chi)=1$.
  - **add-result** Table `tab:twobody`: CV-8 + CV-9a gate/measured/threshold rows.
  - **add-figure**: `fig:cv8-bvp` (two-body BVP, non-matching interface), `fig:cv8-patch` (mortar smooth vs lumped sawtooth in receiving body), `fig:cv9-array` ($3\times3$ array colored by max-principal stress).
- **revise-text** rock-joint roadmap (1652–1666) + opener (1647): position the rock joint as the conforming/monotone-rearrangement OT limit (the complementary regime to the convex Hertz closest-point limit); keep 61%/98% and the Patton anchor verbatim.

### Section 7 — Discussion, limitations, conclusions (lines 1965–2129)
*Role: close on the OT thesis; move CV-8/CV-9a out of future work; keep representation result as payoff; update capability matrix and honest limitations.*
- **reframe** opening (1965–1972): two-statement opener — (1) $\tmap$ is the OT measure coupling (two-sided, mass-conserving, exact constant-pressure transmission, solves two-body BVP); (2) representation controls fidelity (empirical consequence). Add the unification sentence resolving "not a projection."
- **add-subsection** `\label{sec:disc:ot}` "The transition map as a measure coupling" (after 1973): (a) OT map / two limits / arclength-smears-partial; (b) mortar-consistent assembly + patch test 1.4e-16 (67×); (c) two-body symmetric SPSD 4-block tangent, FD 3.45e-11, force balance 4.4e-16.
- **add-result** Table `tab:comparison` (1988–2007): add a two-body / N-body row (CV-8 $a$ 2.75% / $p_0$ 5.82% / patch 1.4e-16 / force balance $\sim10^{-19}$; CV-9a mean 0.58% / anisotropy 0.22% / force balance 3.7e-15) contrasting OT measure coupling vs node-lumped penalty.
- **revise-text** preconditions (2009–2027): distinguish the single-body radial $1/\cos\alpha$ bias from the unbiased measure-coupling gap $\gn=(\xbf_s-T(\xbf_s))\cdot\nbf_s$.
- **revise-text** limitations (2039–2049): add OT-path honest residuals — ~3–6% contact-edge noise in CV-8 $a/p_0$ ($p_0$ plateaus ~5.8%, a genuine floor); single-realization, plane-strain CST small-strain, frictionless, rigid outer walls (CV-9a); dropped $d\nbf/d\chi$ small-rotation approximation, converged solution tangent-independent.
- **revise-text** orthogonality (2042–2048, 2076–2082): constraint-enforcement upgrade is orthogonal to *both* representation and measure-coupling formulation; the tangential treatment plugs into the two-sided OT mortar tangent already assembled.
- **reframe** conclusions opener (2084–2100): prepend the OT sentence (mass-conserving two-sided operator, patch test 1.4e-16, two limits), then keep representation + NTK/spectral-bias + Inada sentences.
- **move-from-futurework** (2102–2112): add a sentence stating the two-body BVP is *solved, not deferred* (CV-8 + CV-9a numbers); restrict remaining future work to (i) multi-chart decoder atlas for overhangs, (ii) **3-D deformable Hertz** (the dimensional step, not the two-body step), (iii) finite-strain/dynamic + mortar/AL semismooth friction.
- **revise-text** 3-D Hertz future-work (2105–2108): re-scope to "extending the *two-body* OT coupling to full 3-D deformable Hertz" so it does not read as still-open two-body work.
- **revise-text** topo-geometric paragraph (2114–2121): add clause naming the partition-of-unity blend as "the same partition of unity that makes the inter-body OT marginal exact."
- **revise-text** closing line (2111–2112): broaden to "the transition map — read as an optimal-transport coupling of the two boundaries — is the mass-conserving, two-sided operator that resolves what the level set smooths and couples two deformable bodies the level set cannot."

### Appendices & Supplementary (lines 2131–2816)
*Role: supply the rigorous OT backbone; register the reproducibility surface; recast the linearization appendix as the slave-slave block of the two-body tangent.*
- **add-subsection** new `\section{The optimal-transport measure coupling and the two-body contact tangent}\label{app:ot}` (between 2495–2497): four parts — A (measure-coupling gap + Brenier $\tmap=F_B^{-1}\circ F_A$, marginal, unbiased normal vs $1/\cos$ radial); B (the two OT limits, resolving "not a projection," arclength-smears-partial); C (discrete measure coupling = mortar assembly, $M=\frac{L}{6}[[2,1],[1,2]]$, $\sum_K(N_K\circ\chi)=1$); D (symmetric 4-block tangent, $K_{ms}=K_{sm}^{\top}$, SymPy all 64 entries, patch-test resultant vanishes).
- **add-result** `\subsection{Verified two-body and N-body results}\label{app:ot:results}`: gate table (SymPy resultant [0,0]; FD 3.45e-11; force balance 4.4e-16; CV-8 full mesh sequence 5.14→2.75% / 6.34→5.82%, patch 1.4e-16, force balance 1.27e-19; CV-9a converged full-Newton, mean 0.58%, anisotropy 0.22%, force balance 3.71e-15) + explicit limitations.
- **revise-text** `app:closedforms` preamble (2144–2149): these closed forms also serve as acceptance targets for the two-body OT solves of `app:ot` (combined $1/E^\ast$; equibiaxial nine-disc as N-body OT output).
- **add-text** `app:ninedisc` (2384–2389): `app:ot` reaches the equibiaxial state as a genuine $3\times3$ N-body OT contact ($N$ emergent), mean stress 0.58%, force balance 3.7e-15, rigid-outer-wall limit.
- **add-result** `tab:closedforms` (2391–2413): two rows for the two-body OT targets (deformable Hertz half-width <10% meas. 2.75%; N-body equibiaxial mean <5% meas. 0.58%) with honestly-looser tolerances.
- **add-text** `app:repro` Code availability (2508–2513): itemize `solvers/contact/measure_coupling/` (`coupling.py` MonotoneCoupling1D / ClosestPointCoupling1D / SinkhornCoupling1D, `gap_field.py`, `traction.py`, `assembly.py`, `two_body.py`); drivers `cv8_deformable_ot.py`, `cv9_nbody_array_ot.py`; SymPy verifier `docs/hertz_derivation/two_body_mortar_tangent.py`; `tests/test_cv9_nbody_ot.py`.
- **add-subsection** `app:repro` Algorithm summary (2515–2547): third procedure `CoupleTwoBody` — build $\chi$, evaluate $g_N$ + traction at Gauss points, scatter $f_I^+/f_K^-$ via OT marginal, assemble symmetric 4-block tangent.
- **add-text** `app:repro` closing (2549–2554): third reproduction note — two-body $a/p_0$ carry a ~3–6% contact-edge floor; the non-matching patch test is exact to $1.4\times10^{-16}$ because the OT marginal holds independently of the mesh.
- **reframe** `app:linearization` opening (2749–2761): name the one-body penalty tangent as the slave-slave block $K_{ss}$ of the general two-body OT tangent (`app:ot`); a deformable counterface adds $K_{sm}=K_{ms}^{\top}$ and $K_{mm}$.
- **add-text** `app:linearization` (2799–2802): the two-body consistent object is the symmetric SPSD 4-block tangent (SymPy 64-entry match, FD 3.45e-11, force balance 4.4e-16).
- **add-text** `app:supp` preamble (2557–2566): the three deferred studies are single-body / rigid-counterface; the two-body deformable–deformable coupling is `app:ot` (so the supplementary is not read as the full OT contribution).
- **add-citation** `app:supp:p4` energy ledger (2668–2701): cross-reference that $W_{\mathrm{fric}}$ is the discrete realisation of the OT traction $t_T$ (shared bound $\abs{t_T}\le\mu p_N$); cite unbalanced-OT for the band-screen active set.

---

## New citations to add to refs.bib
| key | reference | grounds |
|---|---|---|
| `brenier1991polar` | Brenier, *Polar factorization and monotone rearrangement of vector-valued functions*, CPAM 1991 | $\tmap$ = Brenier map = gradient of convex potential; the unification |
| `villani2009optimal` | Villani, *Optimal Transport: Old and New*, Springer 2009 | Monge–Kantorovich backbone |
| `santambrogio2015optimal` | Santambrogio, *Optimal Transport for Applied Mathematicians*, Birkhäuser 2015 | 1-D monotone rearrangement $T=F_B^{-1}\circ F_A$ |
| `mccann1995existence` | McCann, *Existence and uniqueness of monotone measure-preserving maps*, Duke 1995 | conforming/rough-surface OT limit (`MonotoneCoupling1D`) |
| `chizat2018unbalanced` | Chizat, Peyré, Schmitzer, Vialard, *unbalanced/scaling algorithms for OT*, Math. Comp. 2018 | active-set screen (mass=0, `unbalanced=True`) |
| `peyre2019computational` *(optional)* | Peyré & Cuturi, *Computational Optimal Transport*, FnT ML 2019 | entropic Sinkhorn → monotone limit (`SinkhornCoupling1D`) |

**Reuse existing** `puso2004mortar` / `puso2004segment`, `wohlmuth2011variational`, `hueber2005primaldual` / `popp2009finite`, `konyukhov2013computational`, `laursen2002computational` — connect the OT mass-marginal partition-of-unity to the already-cited mortar / dual-multiplier literature; no new entries needed. (Note: use a single, consistent key per work — the editor lists used both `villani2009ot`/`villani2009optimal` and `santambrogio2015ot`/`santambrogio2015optimal` and `mccann1995monotone`/`mccann1995existence`; standardize on one set, e.g. `villani2009optimal`, `santambrogio2015optimal`, `mccann1995existence`.)

## New / modified figures and tables
**New figures**
- **OT-limits schematic** (Sec. `\ref{sec:tmap}`, forward-referenced from Intro C1 and recast 220–230): two panels — left, convex indenter, closest-point map sends central slave patch to matching master patch (partial); right, mated rough profiles, arclength-monotone matches equal cumulative-arclength quantiles (conforming). Generator analogue: `postprocessing/plot_transition_map_manual.py`.
- **`fig:cv8-bvp`** — TikZ two-body deformable Hertz, non-matching interface node spacing, $2a$ marked (parallel to `fig:cv1-bvp`).
- **`fig:cv8-patch`** — mortar (uniform $\sigma_{yy}$) vs lumped (sawtooth) in the receiving body; source `runs/cv8_deformable_ot`.
- **`fig:cv9-array`** — $3\times3$ elastic disc array colored by max-principal stress, 12 mutual-OT interfaces, equibiaxial centre; source `runs/cv9_nbody_array_ot`.

**New tables**
- **`tab:twobody`** (Sec. `\ref{sec:ex_twobody}`) and **`tab:ot:gates`** (`app:ot:results`) — CV-8 + CV-9a gate/measured/threshold rows (mirror manual §11.8.1).
- Two rows added to **`tab:closedforms`** for the two-body OT targets with honestly-looser tolerances.

**Modified tables**
- **`tab:cv_summary`** — add the "Conventional vs OT (same mesh)" column (six CV rows).
- **`tab:comparison`** — add the two-body / N-body OT-vs-node-lumped row.

**Modified caption:** `fig:tm-concept` — add the OT reading (foot $\map{B}(\psi_B)$ = OT partner under closest-point limit; signed gap = normal projection of $\xbf_s-\tmap(\xbf_s)$). `fig:atlas-extension(b)` — add the "optimal-transport rearrangement of $\mu_A$ onto $\mu_B$" parenthetical.

---

## Narrative-consistency through-lines
These five sentences must read **identically in meaning** in the abstract, contributions, formulation, and conclusions:
1. **The identity.** "$\tmap=\map{B}^{-1}\circ\map{A}$ is the optimal-transport (Brenier) map between the two boundary measures." (Abstract 122 / C1 / `sec:tmap:ot` / `sec:disc:ot`.)
2. **The two limits.** "Closest-point projection is the convex-cost / partial-contact limit; the arclength-monotone rearrangement is the conforming / rough limit — one framework. The arclength map smears a partial contact." (Same four places — never call the projection the *opposite* of $\tmap$.)
3. **The discrete coupling.** "$f_I=\sum_q w_q N_I(p_N\nbf+\bm t_T)$, master interpolation a partition of unity along the correspondence $\sum_K(N_K\circ\chi)=1$ (the OT mass marginal) ⇒ constant pressure transmits exactly across a non-matching interface, patch test $1.4\times10^{-16}$, 67× tighter than node-lumped."
4. **The two-body unlock.** "Two-sided, transpose-coupled, symmetric SPSD 4-block tangent $K_{ms}=K_{sm}^{\top}$ ⇒ genuine two-body deformable–deformable BVP." Same headline numbers everywhere: CV-8 $a$ 2.75% / $p_0$ 5.82% / patch 1.4e-16 / force balance 1.27e-19; CV-9a mean 0.58% / anisotropy 0.22% / force balance 3.71e-15; FD tangent 3.45e-11.
5. **The payoff subordination.** "Representation, not the enforcement scheme, controls fidelity" is the *empirical consequence* of the OT coupling on a conforming interface; the 61% strength / 98% dilatancy numbers are unchanged and always introduced as the payoff, not the lead.

---

## Genuinely NEW vs recontextualized

**Genuinely new (advancing former future work):**
- The two-body deformable–deformable BVP: CV-8 deformable Hertz + CV-9a N-body array, the symmetric SPSD 4-block tangent ($K_{ms}=K_{sm}^{\top}$), the master-coupling blocks $K_{sm}/K_{mm}$ that the rigid-master suite never assembled. (New Sec. `\ref{sec:ex_twobody}`, `app:ot`, contribution C3.)
- The OT-vs-conventional head-to-head table column (CV-1…CV-7 conventional vs OT on identical meshes).
- The mortar-consistent measure-coupling element ($M=\frac{L}{6}[[2,1],[1,2]]$, $f_I=\sum_J M_{IJ}t_J$) replacing the node-lumped penalty element, with the measured patch test $1.4\times10^{-16}$.

**Recontextualized (existing content, re-read through OT):**
- The transition map $\tmap$, the radial/height gaps, the conservative normal, the multi-arc scan — all reframed as instances/limits of the OT coupling, not new math.
- The Fourier-feature charts, the strong–weak–Galerkin sequence, the six-benchmark ladder, the Inada rough-joint result (61%/98%) — unchanged numbers, repositioned as the enabling substrate and the empirical payoff.
- The closest-point projection — recast from "the rejected alternative" to "the convex-cost OT limit."

**Honest residuals to keep (do not soften):**
- CV-8 ~3–6% contact-edge noise; $p_0$ plateaus ~5.8% — a genuine floor (infinite Hertz edge-pressure slope over one element), not a vanishing discretization error.
- Both CV-8/CV-9a: single-realization, plane-strain CST (small-strain Tri2D), frictionless, rigid outer walls (CV-9a / T4).
- Dropped $d\nbf/d\chi$ geometric tangent term = standard small-rotation approximation; converged solution tangent-independent (a convergence-rate lever, not an accuracy lever).
- Cyclic energy balance only ~1.5× (Coulomb non-smoothness; needs semismooth-Newton/AL) — keep as remaining limitation.
- Single-valued surfaces only (no overhangs); rigid blocks for the rock joint; band-limited roughness.

---

## Tensions the reframing introduces, and resolutions

1. **"NOT a projection" (related work 226–230, formulation opening 642–646; echoed in atlas 426–432).** This is the central tension. **Resolve uniformly:** the closest-point projection is *not the opposite* of $\tmap$ but its **convex-cost / partial-contact limit**; the arclength-monotone rearrangement is the **conforming limit**. Every "rather than a projection" must become "of which the projection is the convex-cost limit." Grounded in `ClosestPointCoupling1D` docstring ("This IS the composite transition map $\tau_{AB}=\pi_B\circ\phi_A$… the correct OT map for non-conforming / partial contact") and `cv1_ot_gap.py:15-20`.

2. **Multi-arc remark vs single-foot projection.** The paper sells multi-arc detection as the chart's advantage over the single closest-point foot — which seems to clash with "projection IS the OT limit." **Resolve:** the multi-arc capability is a property of the *measure coupling* (many-to-many by the mass overlap), and the closest-point limit is the *partial/convex* specialization that returns one localized foot when the contact is localized. They are not in conflict once both are framed as OT: the convex limit *should* return one arc; the conforming/cusped regime returns many.

3. **Radial $1/\cos\alpha$ bias vs unbiased OT gap.** Promoting the OT gap could let a reader charge the documented single-body radial $1/\cos$ enlargement against the two-body path. **Resolve** (Intro 326–328, precond 2009–2027): explicitly state the $1/\cos$ enlargement is the single-body inverse-radial-chart property; the two-body measure-coupling gap $\gn=(\xbf_s-T(\xbf_s))\cdot\nbf_s$ uses the true surface normal and is unbiased.

4. **Future-work contradiction.** Lines 902–904 and 2102–2108 list two-deformable / two-body contact as future work, but CV-8/CV-9a deliver it. **Resolve:** move-from-futurework everywhere; re-scope the remaining 3-D Hertz item as "extending the *two-body* OT coupling to 3-D" (the dimensional step, not the two-body step).

5. **Bibliography key collisions.** The editor JSON proposes inconsistent keys (`villani2009ot`/`villani2009optimal`, `santambrogio2015ot`/`santambrogio2015optimal`, `mccann1995monotone`/`mccann1995existence`). **Resolve:** standardize on one key per work before insertion to avoid duplicate/undefined `\cite` keys.

---

**Verification note:** All headline numbers (CV-8 2.75% / 5.82% / patch 1.4e-16 / force balance 1.27e-19; CV-9a 0.58% / 3.71e-15; FD 3.45e-11; force balance 4.4e-16; 67× lumped) and the closest-point-as-convex-cost-OT-limit / arclength-smears-partial claims were checked verbatim against `solvers/contact/measure_coupling/{coupling,two_body}.py`, `benchmarks/contact/cv_numerical/cv1_ot_gap.py`, and `docs/ot_benchmark/final_report.md` in the working tree. All section labels and line ranges (`sec:tmap` 625, `sec:discr` 854, `sec:verif` 927, `sec:ex_twobody` to insert after 1637, `sec:disc` 1965, `app:repro` 2497, `app:linearization` 2749) are valid against the current `paper/main.tex`.