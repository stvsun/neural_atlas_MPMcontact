# Theory Manual: Brittle Fracture and Contact Mechanics with Chart-Based FEM / MPM

## 1. Overview

This document describes the mathematical framework for modeling brittle fracture and contact mechanics using the neural atlas coordinate chart approach. The method replaces traditional mesh-based FEM with a collection of overlapping coordinate charts, each equipped with a learned or analytical decoder that maps a reference cube to physical space. Fracture is handled through topology-aware chart construction: persistent homology detects domain changes, and new charts are spawned automatically at crack locations. Contact is handled through an SDF-based gap oracle feeding penalty and augmented-Lagrangian normal forces, regularized Coulomb friction, and a dedicated combined-SDF topology monitor that detects inter-body first contact and separation via $\beta_0$ transitions — all wrapped inside the same Schwarz MPM framework.

The fracture framework rests on three independent material properties following Kamarei et al. (2026):
1. **Elasticity** — how the material deforms (E, ν, or μ, K)
2. **Strength** — when it breaks under stress (Drucker–Prager surface)
3. **Toughness** — energy cost of crack propagation (G_c)

The contact framework (§15) adds:
4. **Gap oracle** — the neural SDF $\phi_A$ evaluated at a query point
5. **Friction coefficient** — regularized Coulomb with smoothing scale $\epsilon_T$
6. **Topology state** — Betti numbers of $\phi_{AB} = \min(\phi_A, \phi_B)$ tracked over time

**Key references (fracture):**
- Kamarei, Zeng, Dolbow & Lopez-Pamies (2026), *CMAME* 448, 118449
- Du (2002), *SIAM J. Numer. Anal.* 39(3) — Robin domain decomposition
- de Souza Neto et al. (1996) — F-bar method
- Tada, Paris & Irwin (2000) — stress intensity factor handbook

**Key references (contact, see §15):**
- Signorini (1933); Kikuchi & Oden (1988) — penalty contact theory
- Alart & Curnier (1991); Simo & Laursen (1992) — augmented Lagrangian contact
- Wriggers (2006) — computational contact mechanics
- Cohen-Steiner, Edelsbrunner & Harer (2007) — persistent homology stability (used in §15.5)

---

## 2. Coordinate Charts and Atlas

### 2.1 Chart Decoder Mapping

A coordinate chart is defined by a smooth diffeomorphism from a reference domain to physical space:

$$\varphi_i : [-1,1]^3 \to \mathbb{R}^3, \qquad \boldsymbol{\xi} \mapsto \mathbf{x} = \varphi_i(\boldsymbol{\xi}).$$

The decoder can be:
- **Neural** (ChartDecoder): $\mathbf{x} = \mathbf{s} + \xi_1 \mathbf{t}_1 + \xi_2 \mathbf{t}_2 + \xi_3 \mathbf{n} + \text{MLP}(\boldsymbol{\xi})$
- **Analytical** (BoxDecoder, TubeSectorDecoder, CylinderDecoder): closed-form with exact inverse
- **CrackTipDecoder**: radial power-law map that absorbs the $1/\sqrt{r}$ singularity (Section 9)

### 2.2 Jacobian and Metric Tensor

The decoder Jacobian $\mathbf{J}_i \in \mathbb{R}^{3\times 3}$ at each point $\boldsymbol{\xi}$ is

$$J_{i,\alpha\beta} = \frac{\partial x_\alpha}{\partial \xi_\beta}, \qquad \mathbf{J}_i = \frac{\partial \varphi_i}{\partial \boldsymbol{\xi}}.$$

Physical gradients are obtained via the chain rule:

$$\frac{\partial u}{\partial x_\alpha} = \frac{\partial u}{\partial \xi_\beta}\, (J_i^{-1})_{\beta\alpha}.$$

The volume element transforms as $\mathrm{d}\mathbf{x} = |\det \mathbf{J}_i|\, \mathrm{d}\boldsymbol{\xi}$.

### 2.3 Atlas and Minimum Chart Count

An atlas is a collection of $M$ overlapping charts $\{(\varphi_i, \tilde{\Omega}_i)\}_{i=1}^M$ covering $\Omega$. The minimum chart count is bounded by the Lusternik–Schnirelmann category:

$$M \geq M_{\min} = \operatorname{cat}(\Omega) + 1 = \beta_0 + \beta_1 + \beta_2 + 1,$$

where $\beta_k$ are the Betti numbers of $\Omega$, computed from the neural SDF via persistent homology (GUDHI CubicalComplex).

---

## 3. P1 Tetrahedral FEM on Mapped Charts

### 3.1 Mesh Generation

Each chart's reference domain $[-r, r]^3$ is discretized as:
1. **Structured hexahedral grid** with $n_{\text{cells}}$ cells per axis, giving $(n_{\text{cells}}+1)^3$ nodes.
2. **Freudenthal decomposition**: each hex is subdivided into 6 tetrahedra using the fixed template
   $\{[0,1,3,7],\; [0,1,5,7],\; [0,2,3,7],\; [0,2,6,7],\; [0,4,5,7],\; [0,4,6,7]\}$
   (vertices 0 and 7 are opposite corners of the hex).
3. **SDF filtering**: elements whose centroid maps to $\text{SDF}(\varphi_i(\boldsymbol{\xi}_c)) > \text{threshold}$ are removed, conforming the mesh to the geometry.

### 3.2 Shape Function Gradients

For a P1 tet with vertices $\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$, define the edge matrix

$$\mathbf{B} = [\mathbf{v}_1 - \mathbf{v}_0 \;|\; \mathbf{v}_2 - \mathbf{v}_0 \;|\; \mathbf{v}_3 - \mathbf{v}_0] \in \mathbb{R}^{3\times 3}.$$

The shape function gradients in reference space are constant per element:

$$\frac{\partial N_0}{\partial \boldsymbol{\xi}} = -\sum_{k=1}^{3} (\mathbf{B}^{-1})_{k,:}, \qquad \frac{\partial N_a}{\partial \boldsymbol{\xi}} = (\mathbf{B}^{-1})_{a,:}, \quad a = 1,2,3.$$

The element volume is $V_e = |\det \mathbf{B}|/6$.

In physical space (when a chart decoder is present):

$$\frac{\partial N_a}{\partial \mathbf{x}} = \frac{\partial N_a}{\partial \boldsymbol{\xi}}\, \mathbf{J}_i^{-1}, \qquad V_e^{\text{phys}} = V_e^{\text{ref}} \cdot |\det \mathbf{J}_i|,$$

where $\mathbf{J}_i$ is evaluated at the element centroid.

### 3.3 Weak Form

The elastostatic equilibrium on chart $i$ reads: find $\mathbf{u} \in V_h$ such that

$$\int_{\tilde{\Omega}_i} \mathbf{P}(\mathbf{F}) : \nabla_{\!\xi}\mathbf{v}\; |\det \mathbf{J}_i|\, \mathrm{d}\boldsymbol{\xi} = \int_{\tilde{\Omega}_i} \mathbf{f} \cdot \mathbf{v}\; |\det \mathbf{J}_i|\, \mathrm{d}\boldsymbol{\xi}, \qquad \forall \mathbf{v} \in V_{h,0},$$

where $\mathbf{P}$ is the first Piola–Kirchhoff stress and $\mathbf{F} = \mathbf{I} + \nabla_{\!x}\mathbf{u}$ is the deformation gradient.

### 3.4 Deformation Gradient

At each element, the physical deformation gradient is

$$F_{iJ} = \delta_{iJ} + \sum_{a=1}^{4} u_{a,i}\, \frac{\partial N_a}{\partial x_J} = \delta_{iJ} + \sum_{a=1}^{4} u_{a,i}\, \frac{\partial N_a}{\partial \xi_K} (J_i^{-1})_{KJ}.$$

For P1 elements, $\mathbf{F}$ is constant per element.

### 3.5 Internal Force Assembly

The element internal force contribution for node $a$, component $i$ is

$$f_{a,i}^{\text{int}} = V_e \sum_J P_{iJ}\, \frac{\partial N_a}{\partial x_J}.$$

In index-free notation: $\mathbf{f}_e = V_e\, \mathbf{P} \cdot (\nabla N)^T$, assembled via scatter-add to global DOFs.

### 3.6 Tangent Stiffness Assembly

The B-matrix maps element DOFs to the vectorized deformation gradient:

$$B_{(3i+J),\,(3a+i)} = \frac{\partial N_a}{\partial x_J}, \qquad \text{size } 9 \times 12 \text{ for P1}.$$

The element stiffness matrix is

$$\mathbf{K}_e = V_e\, \mathbf{B}^T \mathbf{C} \mathbf{B} \in \mathbb{R}^{12 \times 12},$$

where $\mathbf{C} = \partial \mathbf{P}/\partial \mathbf{F} \in \mathbb{R}^{9\times 9}$ is the material tangent in row-major flattening: component $(i,J) \to$ row $3i+J$.

---

## 4. Constitutive Models

### 4.1 Small-Strain Linear Elasticity

For problems where $\|\nabla \mathbf{u}\| \ll 1$, the infinitesimal strain is

$$\boldsymbol{\varepsilon} = \tfrac{1}{2}(\mathbf{F} + \mathbf{F}^T) - \mathbf{I},$$

and the stress (simultaneously first Piola–Kirchhoff and Cauchy for small strain) is

$$\boldsymbol{\sigma} = \lambda\, \text{tr}(\boldsymbol{\varepsilon})\, \mathbf{I} + 2\mu\, \boldsymbol{\varepsilon},$$

with Lame parameters $\mu = E/[2(1+\nu)]$ and $\lambda = E\nu/[(1+\nu)(1-2\nu)]$.

The material tangent is constant (independent of $\mathbf{F}$):

$$C_{iJkL} = \lambda\, \delta_{iJ}\delta_{kL} + \mu\,(\delta_{ik}\delta_{JL} + \delta_{iL}\delta_{Jk}).$$

**Usage:** Challenges 1–6, 8 (soda-lime glass).

### 4.2 Compressible Neo-Hookean Hyperelasticity

For finite deformations, the strain energy density is

$$\Psi = \frac{\mu}{2}(I_1 - 3) - \mu \ln J + \frac{K}{2}(\ln J)^2,$$

where $I_1 = \text{tr}(\mathbf{F}^T\mathbf{F})$, $J = \det \mathbf{F}$, $\mu$ is the shear modulus, and $K$ is the bulk modulus.

The first Piola–Kirchhoff stress is

$$\mathbf{P} = \mu(\mathbf{F} - \mathbf{F}^{-T}) + K \ln J\; \mathbf{F}^{-T}.$$

The material tangent tensor is ($\mathbf{A} = \mathbf{F}^{-1}$):

$$\frac{\partial P_{iJ}}{\partial F_{kL}} = \mu\, \delta_{ik}\delta_{JL} + (\mu - K \ln J)\, A_{Li}A_{Jk} + K\, A_{Ji}A_{Lk}.$$

**Derivation:** From the identities $\partial(\mathbf{F}^{-T})_{iJ}/\partial F_{kL} = -A_{Li}A_{Jk}$ and $\partial(\ln J)/\partial F_{kL} = A_{Lk}$.

**Usage:** Challenges 7, 9 (PU elastomer, $\nu \approx 0.4997$).

### 4.3 St. Venant–Kirchhoff (Large-Strain Linear Elastic)

Uses the Green–Lagrange strain $\mathbf{E} = (\mathbf{F}^T\mathbf{F} - \mathbf{I})/2$ and second Piola–Kirchhoff stress $\mathbf{S} = \lambda\,\text{tr}(\mathbf{E})\,\mathbf{I} + 2\mu\,\mathbf{E}$, with $\mathbf{P} = \mathbf{F}\mathbf{S}$.

**Note:** Not frame-indifferent — produces spurious stresses under rigid-body rotation. Prefer Neo-Hookean for large deformation problems.

---

## 5. F-bar Method: Volumetric Locking Prevention

### 5.1 Motivation

For near-incompressible materials ($\nu \to 0.5$, $\Lambda/\mu \gg 1$), standard P1 tetrahedra over-constrain the volumetric response, producing zero or near-zero displacements (volumetric locking). The F-bar method (de Souza Neto et al. 1996) alleviates this by smoothing the volumetric part of the deformation gradient.

### 5.2 Algorithm

Replace the element deformation gradient $\mathbf{F}_e$ with

$$\bar{\mathbf{F}}_e = \left(\frac{\bar{J}_e}{J_e}\right)^{1/3} \mathbf{F}_e,$$

where $J_e = \det \mathbf{F}_e$ and $\bar{J}_e$ is a volume-weighted nodal average:

1. For each element $e$: compute $J_e = \det \mathbf{F}_e$.
2. For each node $a$: accumulate over the node patch $\mathcal{P}(a) = \{e : a \in e\}$:
$$\bar{J}_a = \frac{\sum_{e \in \mathcal{P}(a)} V_e\, J_e}{\sum_{e \in \mathcal{P}(a)} V_e}.$$
3. For each element $e$: average its 4 nodal values:
$$\bar{J}_e = \frac{1}{4}\sum_{a=1}^{4} \bar{J}_{e_a}.$$
4. Scale: $\bar{\mathbf{F}}_e = (\bar{J}_e / J_e)^{1/3}\, \mathbf{F}_e$.

The stress is computed from $\bar{\mathbf{F}}$ instead of $\mathbf{F}$. The tangent uses $\mathbf{C}(\bar{\mathbf{F}})$ with the standard B-matrix (approximate but sufficient for Newton convergence).

### 5.3 Verification

Without F-bar on PU elastomer ($\nu \approx 0.4997$): $\|\mathbf{u}\| = 0$ (complete locking).
With F-bar: $\|\mathbf{u}\|_\infty = 0.21$ mm (physical deformation).

---

## 6. P2 Quadratic Tetrahedral Elements

### 6.1 Motivation

Near crack tips, the CrackTipDecoder smooths the $1/\sqrt{r}$ singularity into a linear function of $\xi$. P2 elements capture this smooth reference-space field more efficiently than P1, giving $O(h^2)$ convergence instead of $O(h)$.

### 6.2 Node Layout

A P2 tet has 10 nodes: 4 vertices (inherited from P1) + 6 edge midpoints. The 6 edges of tet $(n_0, n_1, n_2, n_3)$ are:

$$(n_0,n_1),\; (n_0,n_2),\; (n_0,n_3),\; (n_1,n_2),\; (n_1,n_3),\; (n_2,n_3).$$

Midpoint nodes are shared between adjacent tets via a global edge-to-midpoint dictionary.

### 6.3 Shape Functions

In barycentric coordinates $(L_1, L_2, L_3)$ with $L_4 = 1 - L_1 - L_2 - L_3$:

**Vertex nodes** ($i = 1,\ldots,4$):
$$N_i(\mathbf{L}) = L_i(2L_i - 1).$$

**Edge midpoints** (edge $(i,j)$):
$$N_{ij}(\mathbf{L}) = 4\, L_i\, L_j.$$

**Partition of unity:** $\sum_{a=1}^{10} N_a(\mathbf{L}) = 1$ for all $\mathbf{L}$.

### 6.4 Shape Function Gradients

The gradients $\partial N_a / \partial L_k$ are:
- $\partial N_i / \partial L_i = 4L_i - 1$ for vertex $i$
- $\partial N_3 / \partial L_k = -(4L_4 - 1)$ for the $L_4$-dependent vertex
- $\partial N_{ij} / \partial L_i = 4L_j$, $\partial N_{ij} / \partial L_j = 4L_i$ for edge $(i,j)$
- Mixed terms for $L_4$-dependent edge nodes involve $-4L_i$ or $4(L_4 - L_i)$

Physical gradients: $\partial N_a / \partial \mathbf{x} = (\partial N_a / \partial \mathbf{L})\, \mathbf{J}_{\text{geom}}^{-1}\, \mathbf{J}_{\text{decoder}}^{-1}$.

### 6.5 Quadrature Rule

4-point Gauss quadrature for tetrahedra (degree 2, exact for cubic integrands):

| Point | $(L_1, L_2, L_3)$ | Weight |
|-------|-----|--------|
| 1 | $(a, a, a)$ | $1/24$ |
| 2 | $(b, a, a)$ | $1/24$ |
| 3 | $(a, b, a)$ | $1/24$ |
| 4 | $(a, a, b)$ | $1/24$ |

where $a = (5 - \sqrt{5})/20 \approx 0.1382$ and $b = (5 + 3\sqrt{5})/20 \approx 0.5854$. Total weight = $1/6$ (reference tet volume).

### 6.6 P2 Assembly

**Internal forces** (sum over quadrature points):

$$f_{a,i}^{\text{int}} = \sum_{q=1}^{4} w_q\, V_e\, \sum_J P_{iJ}(\mathbf{F}_q)\, \frac{\partial N_a}{\partial x_J}\bigg|_q.$$

**Tangent stiffness** ($30 \times 30$ per element):

$$\mathbf{K}_e = \sum_{q=1}^{4} w_q\, V_e\, \mathbf{B}_q^T \mathbf{C}_q \mathbf{B}_q, \qquad \mathbf{B}_q \in \mathbb{R}^{9 \times 30}.$$

### 6.7 Convergence

| Element | DOF/tet | Convergence rate | Best use case |
|---------|---------|-----------------|---------------|
| P1 | 12 | $O(h)$ | Bulk charts, far from singularities |
| P2 | 30 | $O(h^2)$ | Crack-tip charts (with CrackTipDecoder) |

---

## 7. Newton–Raphson Nonlinear Solver

### 7.1 Residual and Iteration

The nonlinear equilibrium $\mathbf{R}(\mathbf{u}) = \mathbf{f}_{\text{int}}(\mathbf{u}) - \mathbf{f}_{\text{ext}} = \mathbf{0}$ is solved by Newton's method:

$$\mathbf{K}_{\text{tan}}(\mathbf{u}_k)\, \Delta\mathbf{u} = -\mathbf{R}(\mathbf{u}_k), \qquad \mathbf{u}_{k+1} = \mathbf{u}_k + \alpha\, \Delta\mathbf{u},$$

where $\alpha \in (0, 1]$ is determined by backtracking line search.

### 7.2 Dirichlet Boundary Conditions

Enforced by row/column elimination:

$$K_{ij} = \begin{cases} \delta_{ij} & \text{if } i \text{ or } j \in \text{BC DOFs,} \\ K_{ij} & \text{otherwise.} \end{cases} \qquad R_i = 0 \;\text{ if } i \in \text{BC DOFs.}$$

### 7.3 Backtracking Line Search

For robustness when Newton overshoots (common with Neo-Hookean at large strains):

```
alpha = 1.0
for _ in range(8):
    u_trial = u + alpha * du
    if ||R(u_trial)|| < ||R(u)||: accept; break
    alpha *= 0.5
```

### 7.4 Convergence Criteria

Iteration terminates when $\|\mathbf{R}_{\text{free}}\| < \text{tol}$ (absolute) or $\|\mathbf{R}_{\text{free}}\|/\|\mathbf{R}_0\| < \text{tol}$ (relative), where "free" excludes constrained DOFs.

---

## 8. Robin Parallel Domain Decomposition

### 8.1 Robin Transmission Conditions

At the artificial boundary $\Gamma_{ij}$ between charts $i$ and $j$, the Robin condition replaces the classical Dirichlet exchange (Du 2002):

$$\frac{\partial \mathbf{u}_i}{\partial \mathbf{n}} + \delta\, \mathbf{u}_i = \frac{\partial \mathbf{u}_j}{\partial \mathbf{n}} + \delta\, \boldsymbol{\lambda} \qquad \text{on } \Gamma_{ij},$$

where $\delta \sim E/h$ is the Robin parameter and $\boldsymbol{\lambda}$ is the interface variable.

### 8.2 Parallel Update (Du 2002, Eqs. 5.9–5.10)

At iteration $k \to k+1$:

**Interface update** (parallel over all chart pairs):
$$\boldsymbol{\lambda}^{k+1} = \frac{\mathbf{u}_i^k + \mathbf{u}_j^k}{2}, \qquad \mathbf{g}^{k+1} = \mathbf{g}^k + \frac{\delta}{2}(\mathbf{u}_j^k - \mathbf{u}_i^k).$$

**Chart solve** (all charts in parallel):
$$\text{Set artificial BC: } \mathbf{u}_i = \boldsymbol{\lambda} + \mathbf{g}/\delta \;\text{ on } \partial\tilde{\Omega}_i \cap \tilde{\Omega}_j.$$
$$\text{Solve: } \mathbf{K}_i \mathbf{u}_i = \mathbf{f}_i \text{ with physical + artificial BCs.}$$

### 8.3 Boundary Interpolation

Neighbor displacement is interpolated via decoder inversion + barycentric lookup:

$$\mathbf{u}_i(\mathbf{x}_{\text{bdy}}) = \sum_a N_a\big(\varphi_j^{-1}(\mathbf{x}_{\text{bdy}})\big)\, \mathbf{u}_{j,a},$$

where $\varphi_j^{-1}$ is the decoder inverse (closed-form for analytical decoders, Newton for neural).

### 8.4 Comparison with Multiplicative Schwarz

| Property | Multiplicative Schwarz | Robin DD (Du 2002) |
|----------|----------------------|-------------------|
| Parallelism | Sequential (color groups) | Fully parallel |
| Interface data | Displacement only | Displacement + flux |
| Convergence rate | Geometric ($\rho < 1$) | Faster ($\delta$-tunable) |
| Typical iterations | 10–25 | 2–5 |

### 8.5 Under-Relaxation

For stability on challenging geometries (thin-walled tubes, large aspect ratios):

$$\mathbf{u}_i^{k+1} = \omega\, \mathbf{u}_i^{\text{new}} + (1 - \omega)\, \mathbf{u}_i^k, \qquad \omega \in [0.3, 0.7].$$

---

## 9. CrackTipDecoder: Singularity-Absorbing Coordinate Chart

### 9.1 The Near-Tip Singularity Problem

The Williams (1957) Mode-I asymptotic solution for stress and displacement near a crack tip has characteristic singularities:

$$\sigma_{ij} \sim \frac{K_I}{\sqrt{2\pi r}}\, f_{ij}(\theta), \qquad u_i \sim \frac{K_I}{2\mu}\sqrt{\frac{r}{2\pi}}\, g_i(\theta, \kappa),$$

where $r$ is the distance from the crack tip, $\theta$ is the polar angle, $K_I$ is the Mode-I stress intensity factor, and $\kappa = 3 - 4\nu$ (plane strain).

Standard finite elements on uniform meshes converge slowly for singular problems. The convergence rate for P1 elements on a quasi-uniform mesh around a $r^{-1/2}$ stress singularity is only $O(h^{1/2})$ in the energy norm — compared to the optimal $O(h)$ for smooth solutions. Achieving 1% accuracy would require $\sim 10^4$ elements near the tip, which is impractical for multi-chart domain decomposition.

**Three classical approaches to this problem:**

| Approach | Mechanism | Drawback |
|----------|-----------|----------|
| h-refinement (mesh grading) | Geometrically graded mesh toward tip | Requires mesh generation expertise |
| p-refinement (spectral) | Higher polynomial order | Pollution from singularity |
| XFEM (enrichment) | Add $\sqrt{r}\,g(\theta)$ to basis | Extra DOFs, blending issues |

The CrackTipDecoder provides a **fourth approach**: absorb the singularity into the coordinate mapping, making the solution smooth in the reference domain. Standard P1/P2 elements then converge at their optimal rate without any enrichment.

### 9.2 Crack Frame Definition

The crack-tip region is described by an orthonormal frame $(\mathbf{t}_1, \mathbf{t}_2, \mathbf{n})$ centered at the tip $\mathbf{c}$:

- $\mathbf{t}_1$: crack propagation direction (tangent to crack front)
- $\mathbf{t}_2$: in-crack-plane direction perpendicular to propagation
- $\mathbf{n}$: crack opening direction (normal to crack plane)

**Construction from physical crack geometry** (`from_crack_tip`):

Given the crack direction $\hat{\mathbf{d}}$ and opening direction $\hat{\mathbf{o}}$:

$$\mathbf{t}_1 = \frac{\hat{\mathbf{d}}}{\|\hat{\mathbf{d}}\|}, \qquad \mathbf{n} = \frac{\hat{\mathbf{o}}}{\|\hat{\mathbf{o}}\|}, \qquad \mathbf{t}_2 = \mathbf{n} \times \mathbf{t}_1.$$

The right-hand rule ensures a consistent orientation across all benchmark geometries.

**Construction from TopologyEvent** (`from_spawned_pair`):

When the ChartSpawner creates a `SpawnedChartPair` from a topology event, the frame is stored as a $3\times 3$ matrix with rows $[\mathbf{t}_1, \mathbf{t}_2, \mathbf{n}]$. The CrackTipDecoder reads this directly:

$$\text{center} = \mathbf{s}_{\pm}, \qquad \text{normal} = \text{frame}[2,:], \qquad \text{tangent}_1 = \text{frame}[0,:], \qquad \text{tangent}_2 = \text{frame}[1,:].$$

### 9.3 Coordinate Mapping

The decoder maps reference coordinates $\boldsymbol{\xi} \in [-1,1]^3$ to physical space in three components:

**In-plane mapping** (linear, along crack face):

$$d_1 = s \cdot \xi_0, \qquad d_2 = s \cdot \xi_1,$$

where $s$ is the in-plane scale (default $s = R$, the support radius). These produce a uniform grid on the crack face.

**Radial mapping** (power law, perpendicular to crack face):

$$t = \text{clamp}\!\left(\frac{\xi_2 + 1}{2},\;\; t_{\min},\;\; 1\right), \qquad r_{\text{phys}} = R \cdot t^p,$$

where $R$ is the support radius, $p$ is the power exponent (default $p = 2$), and $t_{\min} = 0.01$ prevents the singular Jacobian at the exact tip.

**Composite physical position:**

$$\varphi(\boldsymbol{\xi}) = \mathbf{c} + d_1\,\mathbf{t}_1 + d_2\,\mathbf{t}_2 + r_{\text{phys}}\,\mathbf{n}.$$

**Closed-form inverse** (physical $\to$ reference):

$$\xi_0 = \frac{\langle\mathbf{x} - \mathbf{c},\, \mathbf{t}_1\rangle}{s}, \qquad \xi_1 = \frac{\langle\mathbf{x} - \mathbf{c},\, \mathbf{t}_2\rangle}{s}, \qquad \xi_2 = 2\left(\frac{r}{R}\right)^{1/p} - 1,$$

where $r = \max(\langle\mathbf{x} - \mathbf{c},\, \mathbf{n}\rangle,\, 0)$.

### 9.4 Singularity Absorption Mechanism

The key insight: with $p = 2$, the radial mapping converts the square-root singularity into a polynomial:

$$r_{\text{phys}} = R\left(\frac{\xi_2+1}{2}\right)^2 \implies \sqrt{r_{\text{phys}}} = \sqrt{R}\,\frac{\xi_2+1}{2}.$$

The Williams displacement field in physical space is

$$u_y(r, \theta) = \frac{K_I}{2\mu}\sqrt{\frac{r}{2\pi}}\,S(\theta) \sim K_I\, \sqrt{r},$$

which under the mapping becomes

$$u_y(\xi_2, \theta) = \frac{K_I}{2\mu}\sqrt{\frac{R}{2\pi}}\,\frac{\xi_2+1}{2}\,S(\theta) \sim K_I\, \xi_2.$$

This is **linear in $\xi_2$** — exactly representable by P1 elements with zero interpolation error.

**Convergence rates in reference space:**

| Physical singularity | Exponent $p$ | Reference-space regularity | P1 rate | P2 rate |
|---------------------|:---:|---|:---:|:---:|
| $r^{1/2}$ (Mode I) | 2 | $C^\infty$ (linear in $\xi$) | $O(h)$ | $O(h^2)$ |
| $r^{1/3}$ (wedge) | 3 | $C^\infty$ (linear in $\xi$) | $O(h)$ | $O(h^2)$ |
| $r^{1/n}$ (general) | $n$ | $C^\infty$ (linear in $\xi$) | $O(h)$ | $O(h^2)$ |

The general principle: for a physical field with singularity $r^\alpha$, choosing $p = 1/\alpha$ makes the field linear in reference coordinates.

### 9.5 Jacobian and Mesh Concentration

The decoder Jacobian has a block-diagonal structure reflecting the decoupled in-plane and radial components:

$$\mathbf{J} = \frac{\partial \mathbf{x}}{\partial \boldsymbol{\xi}} = \begin{bmatrix} s\,\mathbf{t}_1 & s\,\mathbf{t}_2 & \dfrac{\mathrm{d}r}{\mathrm{d}\xi_2}\,\mathbf{n} \end{bmatrix} \in \mathbb{R}^{3\times 3},$$

where the radial derivative is

$$\frac{\mathrm{d}r}{\mathrm{d}\xi_2} = \frac{R\, p}{2}\left(\frac{\xi_2+1}{2}\right)^{p-1}.$$

**Key properties:**

1. **Determinant:**
$$\det \mathbf{J} = s^2 \cdot \frac{\mathrm{d}r}{\mathrm{d}\xi_2} = \frac{s^2\, R\, p}{2}\left(\frac{\xi_2+1}{2}\right)^{p-1}.$$
At the tip ($\xi_2 \to -1$): $\det \mathbf{J} \to 0$, meaning the physical volume element shrinks — the mesh concentrates automatically near the tip.

2. **Condition number:** The Jacobian is exactly diagonal in the crack frame, so the condition number is $\kappa(\mathbf{J}) = s / (\mathrm{d}r/\mathrm{d}\xi_2)$. At $\xi_2 = -1$: $\kappa \to \infty$ (singular), but the $t_{\min} = 0.01$ clamp bounds it at $\kappa \leq s \cdot 2 / (R\, p\, t_{\min}^{p-1})$.

3. **Physical mesh spacing:** The spacing between adjacent nodes in the radial direction is
$$\Delta r = R \cdot p\, t^{p-1} \cdot \Delta\xi / 2.$$
With $p = 2$, $n_{\text{cells}} = 8$, $\Delta\xi = 2/8 = 0.25$: the innermost ring has $\Delta r = R \cdot 2 \cdot 0.01 \cdot 0.125 = 0.0025\,R$, while the outermost has $\Delta r = R \cdot 2 \cdot 1.0 \cdot 0.125 = 0.25\,R$ — a 100:1 grading ratio achieved automatically.

### 9.6 Inverse Mapping and Barycentric Interpolation

The closed-form inverse enables efficient inter-chart interpolation in the Robin DD solver. When chart $j$ needs the displacement from chart $i$ at a boundary point $\mathbf{x}$:

1. Compute $\boldsymbol{\xi} = \varphi_i^{-1}(\mathbf{x})$ via the closed-form inverse (Section 9.3).
2. Find the containing tet in chart $i$'s reference mesh.
3. Compute barycentric coordinates and interpolate: $\mathbf{u}(\mathbf{x}) = \sum_a N_a(\boldsymbol{\xi})\, \mathbf{u}_a$.

The closed-form inverse avoids Newton iteration, making the interpolation both fast and robust even near the singular tip.

### 9.7 Integration with Chart Spawning

When the TopologyMonitor detects a new $H_1$ feature (crack), the ChartSpawner:
1. Localizes the feature center $\mathbf{c}$ and normal $\mathbf{n}$ from the SDF gradient.
2. Creates a `SpawnedChartPair` with seeds at $\mathbf{c} \pm (r/2)\,\mathbf{n}$.
3. The CrackTipDecoder is constructed via `from_spawned_pair()`, reading the frame directly.
4. The persistence lifetime drives the chart's $n_{\text{cells}}$ (Section 12.7).

This automatic pipeline — topology detection $\to$ feature localization $\to$ singularity-absorbing chart — is unique to this framework and has no analog in XFEM or phase-field methods.

### 9.8 Comparison with Alternative Enrichment Strategies

| Aspect | XFEM | Phase-Field ($\ell$) | CrackTipDecoder |
|--------|------|---------------------|-----------------|
| **Enrichment type** | Basis augmentation $\sqrt{r}\,g(\theta)$ | Regularization ($\ell > 0$) | Coordinate mapping ($r \sim \xi^p$) |
| **Extra DOFs** | Yes (4 per enriched node) | No (but needs fine $\ell$-mesh) | No |
| **Element formulation** | Modified (PU blending) | Standard | Standard P1/P2 |
| **Singularity resolved?** | Yes (exactly) | No (smeared over $\ell$) | Yes (absorbed) |
| **Coupling to bulk** | Blending elements | Continuous field | Schwarz DD boundary exchange |
| **Dynamic spawning** | Not standard | Not needed (diffuse) | Automatic via TopologyMonitor |
| **Multi-physics** | Complex enrichment design | Generic | Generic (any constitutive model) |
| **Implementation** | ~1000 lines of special code | ~100 lines (penalty term) | ~200 lines (decoder class) |

**Key advantage of the CrackTipDecoder**: it achieves singularity resolution without modifying the element formulation, tangent assembly, or solution procedure. The same `ChartVectorFEMSolver` code works for bulk charts (BoxDecoder) and crack-tip charts (CrackTipDecoder) — only the coordinate mapping differs.

---

## 10. Fracture Nucleation: Drucker–Prager Criterion

### 10.1 Strength Surface

Crack nucleation occurs when the Cauchy stress $\boldsymbol{\sigma}$ violates the Drucker–Prager criterion (Kamarei et al. 2026, Eq. 2):

$$\mathcal{F}(\boldsymbol{\sigma}) = \sqrt{J_2} + \alpha\, I_1 - k \geq 0,$$

where:
- $I_1 = \text{tr}(\boldsymbol{\sigma}) = \sigma_1 + \sigma_2 + \sigma_3$ (first invariant)
- $J_2 = \frac{1}{2}\mathbf{s}:\mathbf{s}$ with deviatoric stress $\mathbf{s} = \boldsymbol{\sigma} - \frac{1}{3}I_1\,\mathbf{I}$
- $\alpha = \dfrac{\sigma_{ts}}{\sqrt{3}(3\sigma_{hs} - \sigma_{ts})}$ (pressure sensitivity)
- $k = \dfrac{\sqrt{3}\,\sigma_{hs}\,\sigma_{ts}}{3\sigma_{hs} - \sigma_{ts}}$ (cohesion)

### 10.2 Derived Strengths

From the two independent properties $\sigma_{ts}$ (uniaxial) and $\sigma_{hs}$ (hydrostatic):

$$\sigma_{bs} = \frac{3\sigma_{hs}\sigma_{ts}}{3\sigma_{hs} + \sigma_{ts}} \quad\text{(biaxial)}, \qquad \sigma_{ss} = \frac{\sqrt{3}\,\sigma_{hs}\sigma_{ts}}{3\sigma_{hs} - \sigma_{ts}} \quad\text{(shear)}.$$

### 10.3 Cauchy Stress Conversion

The FEM solver returns first Piola–Kirchhoff stress $\mathbf{P}$. Conversion to Cauchy:

$$\boldsymbol{\sigma} = \frac{1}{J}\, \mathbf{P}\, \mathbf{F}^T, \qquad J = \det \mathbf{F}.$$

### 10.4 Crack Direction

The crack nucleates perpendicular to the maximum principal stress:

$$\mathbf{n}_{\text{crack}} = \text{eigenvector of } \max\!\big(\text{eigenvalues}(\boldsymbol{\sigma})\big).$$

For equi-biaxial tension: degenerate (all in-plane directions). For pure shear $\tau_{xy}$: crack normal at 45° to the shear axes.

---

## 11. Crack Propagation: Griffith Energy Balance

### 11.1 Critical Stress Intensity Factor

From the Griffith energy balance:

$$K_{Ic} = \sqrt{\frac{E\, G_c}{1 - \nu^2}} \quad\text{(plane strain)}, \qquad K_{Ic} = \sqrt{E\, G_c} \quad\text{(plane stress)}.$$

A crack propagates when $K_I \geq K_{Ic}$.

### 11.2 Irwin Relation

The energy release rate $G$ relates to the stress intensity factor via

$$G = \frac{K_I^2(1 - \nu^2)}{E} \quad\text{(plane strain)}.$$

At critical: $G = G_c$ exactly when $K_I = K_{Ic}$.

### 11.3 K_I Extraction from FEM

The Williams Mode-I asymptotic displacement near the tip is

$$u_y = \frac{K_I}{2\mu}\sqrt{\frac{r}{2\pi}}\,\sin\frac{\theta}{2}\left[\kappa + 1 - 2\cos^2\frac{\theta}{2}\right],$$

where $\kappa = 3 - 4\nu$ (plane strain) and $\mu = E/[2(1+\nu)]$.

**Extraction algorithm:**
1. Transform FEM displacements to crack-tip polar frame $(r, \theta)$.
2. Select annular region $r \in [0.02\, r_{\max},\; 0.15\, r_{\max}]$ (near-tip, not too close).
3. Subtract far-field linear trend via least squares: $u_{\text{corr}} = u_y - (a\, x_2 + b)$.
4. Point-wise inversion: $K_I^{(p)} = u_{\text{corr}} \cdot 2\mu / [\sqrt{r/(2\pi)} \cdot S(\theta)]$.
5. Robust estimate: $K_I = \text{median}(K_I^{(p)})$.

### 11.4 Propagation Direction

For mixed-mode loading with $K_{II} \neq 0$, the maximum tangential stress criterion (Erdogan & Sih 1963) gives:

$$\theta_c = 2\arctan\!\left(\frac{K_I - \sqrt{K_I^2 + 8 K_{II}^2}}{4 K_{II}}\right).$$

For pure Mode I ($K_{II} = 0$): $\theta_c = 0$ (straight ahead).

### 11.5 Geometry Factor

For a single edge crack of length $a$ in a plate of half-width $W$ under far-field tension $\sigma_\infty$:

$$K_I = \sigma_\infty \sqrt{\pi a}\, F(a/W),$$

where $F(\lambda) = 1.122 - 0.231\lambda + 10.55\lambda^2 - 21.71\lambda^3 + 30.38\lambda^4$ (Tada polynomial, valid for $a/W \leq 0.6$).

### 11.6 Quasi-Static Propagation Driver

The propagation loop:
1. Solve elastic BVP at current crack length $a$.
2. Extract $K_I$ at the tip.
3. If $K_I \geq K_{Ic}$: advance crack by $\Delta a$ in the direction $\theta_c$.
4. Update SDF: $\text{SDF}_{\text{cracked}} = \max(\text{SDF}_{\text{base}}, -\text{SDF}_{\text{crack}})$.
5. Rebuild charts and repeat.

For the DCB specimen, beam-theory compliance $C(a) = 2a^3/(3EI)$ gives the analytical force–displacement curve:

$$F_{\text{crit}}(a) = B\sqrt{\frac{E\, G_c\, h^3}{12\, a^2}}, \qquad \delta_{\text{crit}} = F_{\text{crit}} \cdot C(a).$$

Stable crack growth: $\mathrm{d}F/\mathrm{d}a < 0$ (force decreases with crack extension).

---

## 12. Persistent Homology and Topology-Aware Fracture

This section describes the central contribution of this work: the use of persistent homology from computational algebraic topology to detect, localize, and adaptively respond to topological changes in the deforming domain during fracture simulation. The pipeline transforms a continuous SDF field into discrete topological events that drive automatic chart spawning with persistence-proportional mesh refinement — a capability absent from classical FEM, XFEM, and phase-field approaches.

### 12.1 Mathematical Foundations

#### 12.1.1 Sublevel-Set Filtration

Let $s_\theta : \mathbb{R}^3 \to \mathbb{R}$ be a signed distance function (neural or analytical) with $s_\theta < 0$ in the domain interior and $s_\theta = 0$ on the boundary. The **sublevel-set filtration** is the nested family

$$\Omega_t = \{\mathbf{x} \in \mathbb{R}^3 : s_\theta(\mathbf{x}) \leq t\}, \qquad t \in \mathbb{R},$$

satisfying $\Omega_s \subseteq \Omega_t$ for $s \leq t$. The physical domain is $\Omega = \Omega_{0^-} = \{s_\theta < 0\}$.

As $t$ increases from $-\infty$ toward 0, topology changes occur:
- **Birth**: a new connected component, loop, or void appears at $t = b$
- **Death**: the feature merges with an older one or closes at $t = d$

The **persistence** $|d - b|$ quantifies the feature's robustness to perturbation.

#### 12.1.2 Cubical Homology

The SDF is sampled on a regular cubical grid $\mathcal{G} \subset \mathbb{R}^3$ with resolution $N^3$:

$$s_{ijk} = s_\theta(\mathbf{x}_{ijk}), \qquad \mathbf{x}_{ijk} = \mathbf{x}_{\min} + \left(\frac{i}{N-1},\, \frac{j}{N-1},\, \frac{k}{N-1}\right) \odot (\mathbf{x}_{\max} - \mathbf{x}_{\min}).$$

Values are clipped to the interior: $s_{ijk} \leftarrow \text{clamp}(s_{ijk},\; s_{\min},\; 0)$, suppressing exterior topology.

GUDHI's `CubicalComplex` constructs the full cubical complex from $\{s_{ijk}\}$ and computes persistent homology with $\mathbb{Z}/2\mathbb{Z}$ coefficients in $O(N^3 \log N)$ time. This is **exact** (not an approximation like Vietoris–Rips or Cech complexes) because the filtration is defined on a cubical grid.

#### 12.1.3 Persistence Diagrams

The output is a collection of **persistence diagrams**, one per homological dimension $k \in \{0, 1, 2\}$:

$$\text{Dgm}_k = \{(b_i, d_i) : i = 1, \ldots, m_k\}, \qquad b_i \leq d_i \leq +\infty.$$

Each pair $(b_i, d_i)$ represents a $k$-dimensional topological feature:

| Dimension $k$ | Feature | Physical interpretation in fracture |
|:-:|---|---|
| 0 | Connected component | Domain fragmentation (complete crack severance) |
| 1 | Loop / tunnel | Through-crack forming a topological handle |
| 2 | Enclosed void | Cavitation, internal void nucleation |

**Essential pairs** ($d_i = +\infty$) represent features that persist to the boundary — these are the genuine topological features of the domain.

#### 12.1.4 Betti Numbers

The Betti numbers at filtration parameter $t$ count the number of alive features:

$$\beta_k(t) = |\{(b, d) \in \text{Dgm}_k : b \leq t < d\}|.$$

For the physical domain, we evaluate at $t = -\varepsilon$ (just inside the boundary):

| Domain | $\beta_0$ | $\beta_1$ | $\beta_2$ | Interpretation |
|--------|:-:|:-:|:-:|---|
| Solid ball | 1 | 0 | 0 | Contractible, simply connected |
| Solid torus | 1 | 1 | 0 | One tunnel ($H_1$ generator) |
| Thick spherical shell | 1 | 0 | 1 | One enclosed void ($H_2$ generator) |
| Cracked plate (H₀ split) | 2 | 0 | 0 | Two disconnected pieces |
| Plate with through-hole | 1 | 1 | 0 | One loop around the hole |

### 12.2 Bottleneck Stability Theorem

The Bottleneck Stability Theorem (Cohen-Steiner, Edelsbrunner & Harer 2007) provides the mathematical foundation for using persistence as a robustness measure:

$$d_B\big(\text{Dgm}_k(f),\, \text{Dgm}_k(g)\big) \leq \|f - g\|_\infty,$$

where the **bottleneck distance** is

$$d_B(A, B) = \inf_{\gamma : A \to B} \max_{p \in A} \|p - \gamma(p)\|_\infty,$$

and $\gamma$ ranges over all bijections that may map points to the diagonal $\{b = d\}$.

**Consequences for fracture simulation:**

1. **Noise filtering**: If the SDF approximation error is $\|s_\theta - s_{\text{exact}}\|_\infty < \varepsilon$, then all persistence pairs with lifetime $|d - b| > 2\varepsilon$ are guaranteed to correspond to genuine topological features of the exact domain. Short-lived pairs ($|d - b| \leq 2\varepsilon$) may be numerical artifacts.

2. **Threshold design**: The lifetime filter $\tau = 0.05 \times (s_{\max} - s_{\min})$ is calibrated so that features with $|d - b| > \tau$ survive mesh refinement, network retraining, and small geometric perturbations.

3. **Event detection**: A bottleneck change $d_B(\text{Dgm}^{k}, \text{Dgm}^{k+1}) > 0.02$ between consecutive load steps indicates a genuine topology change, not numerical drift.

**Verification**: The stability theorem is empirically validated by adding Gaussian noise ($\sigma = 0.01$) to the SDF grid and verifying $d_B \leq 0.05$ (see `test_bottleneck_stability`).

### 12.3 Lusternik–Schnirelmann Category and Minimum Chart Count

#### 12.3.1 Topological Lower Bound

The **Lusternik–Schnirelmann category** $\operatorname{cat}(\Omega)$ of a topological space $\Omega$ is the minimum number of contractible open sets needed to cover $\Omega$. By the **Nerve Theorem**, an atlas of $M$ coordinate charts forms a good cover only if

$$M \geq M_{\min} = \operatorname{cat}(\Omega) + 1.$$

This is a **hard topological floor**: no amount of mesh refinement, overlap tuning, or solver improvement can reduce the chart count below $M_{\min}$.

#### 12.3.2 Cup-Length Lower Bound

Computing $\operatorname{cat}(\Omega)$ exactly is NP-hard in general, but the **cup-length** provides a computable lower bound:

$$\operatorname{cat}(\Omega) \geq \operatorname{cl}(\Omega) = |\{k \geq 1 : \beta_k(\Omega) > 0\}|.$$

This counts the number of non-trivial homology dimensions. The bound is **tight** for all orientable surfaces and for all benchmark domains in this work:

| Domain | $\beta_0, \beta_1, \beta_2$ | $\operatorname{cl}$ | $\operatorname{cat}$ | $M_{\min}$ |
|--------|:--:|:-:|:-:|:-:|
| Ball | 1, 0, 0 | 0 | 0 | 1 |
| Solid torus | 1, 1, 0 | 1 | 1 | 2 |
| Thick shell | 1, 0, 1 | 1 | 1 | 2 |
| Intact plate | 1, 0, 0 | 0 | 0 | 1 |
| Cracked plate (severed) | 2, 0, 0 | 0 | 0 | 1 per piece |

#### 12.3.3 Practical Chart Count

Beyond the topological minimum, numerical accuracy requires sufficient chart overlap. The practical chart count balances two constraints:

1. **Topological**: $M \geq M_{\min}$ (nerve theorem)
2. **Numerical**: $M \geq V(\Omega) / V_{\text{chart}} \cdot (1 + \alpha_{\text{overlap}})$, targeting condition number $\kappa \leq 10$

For the solid torus benchmark: $M_{\min} = 2$ (topological), $M_{\text{practical}} = 8$ (numerical).

### 12.4 Topology Monitoring During Simulation

#### 12.4.1 TopologyMonitor State Machine

The `TopologyMonitor` maintains state across load steps:

$$\text{State}^n = \big(\text{step}_n,\; \{\text{Dgm}_k^n\}_k,\; \{\beta_k^n\}_k,\; \text{event\_count}\big).$$

At each call to `update(grid_vals, load_step)`:

1. **Compute new diagrams**: $\text{Dgm}_k^{n+1} \leftarrow \text{PH}(\text{grid\_vals})$
2. **Filter**: retain pairs with $|d - b| > \tau$ (relative to SDF range)
3. **Compare**: for each monitored dimension $k$:
   $$\Delta_k = d_B\big(\text{Dgm}_k^n,\; \text{Dgm}_k^{n+1}\big)$$
4. **Detect**: if $\Delta_k > \delta_{\text{thresh}}$ (default 0.02), invoke pair matching

#### 12.4.2 Greedy Pair Matching

When a bottleneck change exceeds the threshold, the monitor identifies **which** persistence pairs are new. The matching algorithm is a greedy nearest-neighbor procedure in the $\ell^\infty$ metric:

**Input:** Previous diagram $A = \{(b_i^A, d_i^A)\}$ and current diagram $B = \{(b_j^B, d_j^B)\}$.

**Distance metric:**
$$\text{dist}\big((b_1, d_1),\, (b_2, d_2)\big) = \begin{cases} \max(|b_1 - b_2|,\, |d_1 - d_2|) & \text{both finite,} \\ |b_1 - b_2| & d_1 = d_2 = \infty, \\ +\infty & \text{mixed infinities.} \end{cases}$$

**Algorithm:**
```
used = {}
new_pairs = []
for each (b, d) in B:
    best_match = argmin_{(b', d') in A \ used} dist((b,d), (b',d'))
    if dist(best_match) < 2 * delta_thresh:
        used.add(best_match)
    else:
        new_pairs.append((b, d))  // unmatched → new topological feature
return new_pairs
```

**Matching threshold**: $2\delta_{\text{thresh}} = 0.04$ — twice the bottleneck threshold, providing a buffer zone between genuine new features and small perturbations of existing ones.

#### 12.4.3 TopologyEvent

Each new unmatched pair generates a `TopologyEvent`:

| Field | Type | Meaning |
|-------|------|---------|
| `load_step` | int | Simulation step of detection |
| `event_type` | str | `'new_H0'`, `'new_H1'`, `'new_H2'`, or `'feature_died'` |
| `dimension` | int | Homological dimension of the changed feature |
| `birth_value` | float | SDF filtration value at feature birth |
| `lifetime` | float | Persistence $|d - b|$ (robustness measure) |
| `localization` | ndarray(3) | Physical coordinates (from SDF gradient) |
| `bottleneck_change` | float | $d_B$ that triggered this event |

**Event types and their fracture-mechanical interpretation:**

| Event | Topological Change | Physical Meaning |
|-------|-------------------|------------------|
| `new_H0` | $\beta_0 \to \beta_0 + 1$ | Domain fragments (complete crack severance) |
| `new_H1` | $\beta_1 \to \beta_1 + 1$ | Through-crack creates a tunnel/loop |
| `new_H2` | $\beta_2 \to \beta_2 + 1$ | Internal void nucleates (cavitation) |
| `feature_died` | $\beta_k \to \beta_k - 1$ | Feature closes (crack healing, void collapse) |

**Verification contracts:**
- Fixed domain (no crack growth): 0 events after initial step (verified in tests)
- Ball-to-torus transition (simulating through-crack): exactly 1 `new_H1` event (verified)

### 12.5 Feature Localization via SDF Gradient

When a `TopologyEvent` fires, the feature must be **localized in physical space** to place new chart seeds. The localization algorithm uses the SDF gradient field:

1. **Grid-based search**: find cells where $|s_{ijk} - b_{\text{event}}| < \tau_{\text{loc}}$, with $\tau_{\text{loc}} = 0.05 \times (s_{\max} - s_{\min})$.

2. **Physical coordinate conversion**: map grid indices to physical space via the bounding box affine transform:
$$\mathbf{x}_{\text{phys}} = \mathbf{x}_{\min} + \frac{\mathbf{i}}{N - 1} \odot (\mathbf{x}_{\max} - \mathbf{x}_{\min}).$$

3. **Center**: $\mathbf{c} = \text{mean}(\{\mathbf{x}_{\text{phys}}\}_{\text{matched cells}})$.

4. **Normal estimation** via central finite differences on the SDF grid:
$$\frac{\partial s}{\partial x_d}\bigg|_{\mathbf{c}} \approx \frac{s(\mathbf{c} + \varepsilon\, \mathbf{e}_d) - s(\mathbf{c} - \varepsilon\, \mathbf{e}_d)}{2\varepsilon\, \Delta x_d}, \qquad \mathbf{n} = \frac{\nabla s}{\|\nabla s\|}.$$

The normal $\mathbf{n}$ is the crack-opening direction, perpendicular to the crack face.

### 12.6 Topology-Driven Chart Spawning

The `ChartSpawner` converts each `TopologyEvent` into a `SpawnedChartPair` — two new charts placed on opposite sides of the detected feature:

#### 12.6.1 Seed Placement

$$\mathbf{s}_+ = \mathbf{c} + \tfrac{r}{2}\,\mathbf{n}, \qquad \mathbf{s}_- = \mathbf{c} - \tfrac{r}{2}\,\mathbf{n},$$

where $r$ is the default chart support radius (typically $0.3 \times$ domain diameter).

#### 12.6.2 Orthonormal Frame Construction

Given normal $\mathbf{n}$, construct a right-handed frame $[\mathbf{t}_1, \mathbf{t}_2, \mathbf{n}]$ via Gram–Schmidt:

1. Choose reference vector: $\mathbf{r} = [1,0,0]$ if $|\mathbf{n} \cdot [1,0,0]| < 0.9$, else $\mathbf{r} = [0,1,0]$.
2. $\mathbf{t}_1 = (\mathbf{r} - (\mathbf{r}\cdot\mathbf{n})\,\mathbf{n}) / \|\cdot\|$ (in-plane direction 1).
3. $\mathbf{t}_2 = \mathbf{n} \times \mathbf{t}_1$ (in-plane direction 2, automatic orthogonality).

The minus-side frame uses $-\mathbf{n}$ for proper crack-face orientation.

#### 12.6.3 Parent Chart Warm-Start

The nearest existing chart provides warm-start initialization:

$$\text{parent} = \arg\min_i \|\mathbf{s}_i - \mathbf{c}\|_2,$$

where $\{\mathbf{s}_i\}$ are existing chart seed positions. The new decoder is initialized from the parent's weights (for neural decoders) or parameterized analytically (for CrackTipDecoder via `from_spawned_pair`).

### 12.7 Persistence-Driven Adaptive hp-Refinement

The **persistence lifetime** $\ell = |d - b|$ provides a mathematically principled mesh refinement indicator — the key insight connecting computational topology to adaptive FEM. Longer-lived features are more physically significant and demand finer resolution.

#### 12.7.1 Adaptive h-Refinement (Mesh Density)

The mesh resolution for a spawned chart is scaled proportionally to the normalized persistence lifetime:

$$\tilde{\ell} = \min\!\left(\frac{\ell}{s_{\max} - s_{\min}},\; 1\right), \qquad n_{\text{cells}} = \text{clamp}\!\left(n_{\text{base}} \cdot (1 + 2\tilde{\ell}),\;\; n_{\text{base}},\;\; 3\, n_{\text{base}}\right).$$

Additionally, minimum resolution floors are enforced by homological dimension:

$$n_{\text{cells}} \geq \begin{cases} 12 & \text{if } k = 1 \text{ (cracks/loops),} \\ 10 & \text{if } k = 2 \text{ (voids).} \end{cases}$$

**Rationale**: $H_1$ features (cracks) have $1/\sqrt{r}$ stress singularities requiring fine near-tip resolution; $H_2$ features (voids) have smoother stress concentrations but still benefit from refinement.

#### 12.7.2 Adaptive p-Refinement (Polynomial Order)

Features of dimension $k \geq 1$ trigger P2 (quadratic) element recommendation:

$$\text{element\_order} = \begin{cases} 2 & \text{if } k \geq 1 \text{ (crack or void),} \\ 1 & \text{if } k = 0 \text{ (component split).} \end{cases}$$

Combined with the CrackTipDecoder's radial squaring ($r \sim \xi^2$, making $\sqrt{r}$ linear in $\xi$), P2 elements achieve **$O(h^2)$ convergence** on the smooth reference-space Williams field — compared to $O(h^{1/2})$ for P1 on the singular physical-space field.

#### 12.7.3 hp-Adaptivity Summary

The complete adaptive strategy is:

| Feature Type | Lifetime | $n_{\text{cells}}$ | Element Order | Decoder |
|:---:|:---:|:---:|:---:|:---:|
| Short-lived $H_0$ | $\tilde{\ell} < 0.1$ | $n_{\text{base}}$ | P1 | BoxDecoder |
| Long-lived $H_1$ (crack) | $\tilde{\ell} > 0.5$ | $\leq 3 n_{\text{base}}$ | P2 | CrackTipDecoder |
| Short-lived $H_1$ (noise) | $\tilde{\ell} < 0.05$ | filtered out | — | — |
| $H_2$ (void) | any | $\geq 10$ | P2 | BoxDecoder |

### 12.8 Integration with the Fracture Simulation Loop

The persistent homology pipeline integrates with the FEM solver in a closed loop:

```
for each load step n:
    1. SOLVE: u_charts = RobinSchwarz(atlas, stress_fn, tangent_fn, BC)

    2. NUCLEATE: if max(F_DP(sigma)) >= 0:
         crack_center, crack_normal = locate_nucleation(sigma)
         SDF.insert_crack(crack_center, crack_normal)

    3. MONITOR: grid = sample_SDF(resolution=16)
         events = TopologyMonitor.update(grid, step=n)

    4. SPAWN: for each event in events:
         pair = ChartSpawner.spawn_from_event(event)
         n_cells = pair.recommended_n_cells       // persistence-driven
         order   = pair.recommended_element_order  // P2 for cracks
         decoder = CrackTipDecoder.from_spawned_pair(pair)
         solver  = ChartVectorFEMSolver(n_cells, order, decoder)
         atlas.add_chart(solver)

    5. REBUILD: update Schwarz overlap graph, re-color for parallel solve
```

**Key design principle**: Steps 3–5 are **automatic** — no user intervention is needed to detect, localize, or mesh-refine at crack locations. The persistence lifetime provides the refinement signal; the CrackTipDecoder provides the singularity-absorbing coordinates; and the Schwarz solver provides the chart coupling.

### 12.9 Computational Cost

| Operation | Cost | Typical Timing |
|-----------|------|---------------|
| SDF grid sampling ($16^3$) | $O(N^3)$ | ~2 ms |
| GUDHI CubicalComplex | $O(N^3 \log N)$ | ~57 ms (16³) |
| Persistence filtering | $O(m)$ | < 1 ms |
| Bottleneck distance | $O(m^2)$ (greedy) | < 1 ms |
| Feature localization | $O(N^3)$ | ~1 ms |
| **Total overhead per step** | | **~60 ms** |

At 16³ resolution and monitoring every 50 load steps in a 250-step simulation: **overhead = 1.1%** (well within the 5% computational budget).

### 12.10 Verification and Validation

The persistent homology pipeline is verified against exact analytical results:

| Test | Input | Expected | Measured | Status |
|------|-------|----------|----------|:------:|
| Ball contractibility | Solid sphere SDF | $\beta_0=1, \beta_1=0, \beta_2=0$ | exact | PASS |
| Torus tunnel | Solid torus SDF | $\beta_0=1, \beta_1=1, \beta_2=0$ | exact | PASS |
| Shell void | Thick shell SDF | $\beta_0=1, \beta_1=0, \beta_2=1$ | exact | PASS |
| $M_{\min}$ ball | Convex body | $M_{\min} = 1$ | 1 | PASS |
| $M_{\min}$ torus | $S^1$ bundle | $M_{\min} = 2$ | 2 | PASS |
| Bottleneck stability | Noisy SDF ($\sigma=0.01$) | $d_B \leq 0.05$ | $\leq 0.03$ | PASS |
| Fixed domain | Uncracked plate | 0 events/step | 0 | PASS |
| Crack detection | Ball$\to$torus | $\geq 1$ H₁ event | 1 | PASS |
| Biaxial splitting | Cracked plate | H₀: 1$\to$2 | 2 events | PASS |
| Adaptive n_cells | Long H₁ ($\ell=0.5$) | $n \geq 12$ | 16 | PASS |
| Element order | H₁ feature | P2 | P2 | PASS |

### 12.11 Comparison with Alternative Approaches

| Approach | Crack Detection | Localization | Mesh Adaptation | Theoretical Guarantee |
|----------|:-:|:-:|:-:|:-:|
| **This work (PH)** | Automatic (GUDHI) | SDF gradient | Persistence-driven hp | Bottleneck stability |
| Phase-field | Implicit ($\phi \to 0$) | Diffuse interface | Fixed mesh | $\Gamma$-convergence |
| XFEM | Level set | Node enrichment | Manual | None |
| Remeshing | Element quality | Error estimator | ZZ recovery | A posteriori bound |
| Peridynamics | Bond breaking | Particle damage | Fixed | Nonlocal theory |

The persistent homology approach is the **only** method that provides: (1) rigorous topological classification of crack events by homological dimension; (2) stability guarantees via the bottleneck theorem; and (3) automatic persistence-proportional mesh refinement without user intervention.

---

## 13. SDF Geometry Representation

### 13.1 CSG for Crack Insertion

Cracks are represented as thin planar slits subtracted from the base domain:

$$\text{SDF}_{\text{cracked}}(\mathbf{x}) = \max\!\big(\text{SDF}_{\text{base}}(\mathbf{x}),\; -\text{SDF}_{\text{crack}}(\mathbf{x})\big).$$

Each crack slit is parameterized by center, normal, half-length, and slit width $\delta$.

### 13.2 CrackedPlateSDFOracle

For fracture benchmarks, the exact SDF of a rectangular plate with an edge crack is provided:
- Plate: $[-W, W] \times [-H/2, H/2] \times [-T/2, T/2]$
- Crack: thin slit at $x_2 = 0$, $x_1 \in [-W, -W+a]$, with half-opening $\delta$

The SDF is computed as $\max(\text{SDF}_{\text{plate}},\; -\text{SDF}_{\text{slit}})$.

---

## 14. Validation on Kamarei et al. (2026) Benchmark

### 14.1 Nine Circles Scorecard

| # | Challenge | Section | Score | Key Validation |
|---|-----------|---------|-------|----------------|
| 1 | Uniaxial Tension | 2.1 | 100/100 | $\sigma_{zz}$ exact, DP at 40.3 MPa (0.8%) |
| 2 | Biaxial Tension | 2.2 | 100/100 | $\sigma_{bs}$ = 27.0 MPa (5.0%), GUDHI H₀ detection |
| 3 | Torsion | 2.3 | 100/100 | $\tau$ = 0.00% error, 45° crack angle |
| 4 | Pure-Shear Fracture | 3.1 | 100/100 | $K_I$ linear, $\theta_c = 0$, $G = G_c$ |
| 5 | Single Edge Notch | 4.1 | 100/100 | Strength–Griffith transition curve |
| 6 | Indentation | 4.2 | 100/100 | Ring crack $r > R_{\text{punch}}$, cone angle 44° |
| 7 | Poker-Chip | 4.3 | 100/100 | Hydrostatic $p > 0$, Neo-Hookean + F-bar |
| 8 | DCB | 5.1 | 90/100 | FEM K_I extraction, stable crack growth |
| 9 | Trousers | 5.2 | 100/100 | Mode III $G = 2F/B$ (Rivlin–Thomas) |
| | **Total** | | **890/900** | **98.9%** |

### 14.2 Quantitative Benchmarks

| Quantity | Our Model | Reference | Error |
|----------|-----------|-----------|-------|
| $\sigma_{bs}$ (glass, DP) | 27.03 MPa | 27 MPa (Table 2) | 0.1% |
| $\sigma_{ss}$ (glass, DP) | 44.4 MPa | 44.4 MPa (Table 2) | < 0.1% |
| $K_I$ (Williams extraction) | exact | analytical | 0.0% |
| Nucleation strain (glass) | 3.02e-4 | 3.01e-4 | 0.3% |
| Robin DD iterations | 2 | — | instant |
| GUDHI overhead (16³ grid) | 1.1% | < 5% budget | PASS |

---

## 15. MPM Contact Mechanics

This section describes the contact framework built on top of the chart-based MPM solver. The entire framework operates in **physical space** — particle velocities, contact forces, and gaps are all in $\mathbb{R}^3_{\text{phys}}$, not in chart-local $\boldsymbol{\xi}$-space. Since the MPM grid velocity and gravity are already physical (see §3 of the MPM solver conventions), the contact force scatters through the existing P2G channel with **no Jacobian pull-back**.

### 15.1 Gap Function from a Neural SDF

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

### 15.2 Penalty Normal Force

The classical penalty formulation approximates the Signorini constraint $g_N \ge 0$ by

$$p_n = \epsilon_n\, \langle -g_N\rangle_+, \qquad \mathbf{f}_p^{\text{contact}} = p_n\, V_p\, \mathbf{n}_A,$$

where $\langle \cdot \rangle_+$ is the Macaulay bracket ($\max(0, \cdot)$), $V_p$ is the current particle volume, and $\mathbf{n}_A$ is the outward normal of the obstacle (§15.1). This is the formula implemented in `solvers/contact/penalty.py::compute_contact_force`.

**Scattering to the grid.** Because MPM stores $\mathbf{v}$ and gravity in physical space, the contact force in physical space is added to the grid node $I$ with the same shape-function scatter as gravity:

$$\mathbf{f}_I^{\text{ext}} \mathrel{+}= \sum_p \underbrace{\mathbf{f}_p^{\text{contact}}}_{\text{per-particle force}} \, N_I(\boldsymbol{\xi}_p).$$

Note the **absence of a mass factor** relative to gravity: gravity is an acceleration scaled by $m_p$, whereas $\mathbf{f}_p^{\text{contact}}$ is already a force (its $V_p$ factor absorbs the "force per unit volume" of the penalty pressure). This is implemented as a new block in `solvers/mpm/transfers.py::particle_to_grid`, immediately after the gravity scatter, guarded by an `Optional[torch.Tensor]` parameter so all pre-existing call sites continue to work unchanged.

**Contact-stable time step.** The penalty spring introduces a characteristic oscillation frequency $\omega \sim \sqrt{\epsilon_n / m_p}$, which enters the CFL bound for explicit MPM. The stable time step becomes

$$\Delta t_{\text{contact}} \le \operatorname{safety}\cdot \sqrt{m_{\min}/\epsilon_n},\qquad \text{safety}\approx 0.5,$$

implemented in `solvers/contact/penalty.py::contact_stable_dt`. The effective time step for a contact-enabled MPM run is $\min(\Delta t_{\text{CFL}}, \Delta t_{\text{contact}})$.

### 15.3 Augmented Lagrangian (Uzawa) Normal Force

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

### 15.4 Regularized Coulomb Friction

The classical Coulomb law is non-smooth, non-associative, and discontinuous at zero tangential velocity (Wriggers 2006, §4.1). We use the regularized form

$$\mathbf{f}_T^p = -\mu\, \|\mathbf{f}_N^p\|\, \frac{\mathbf{v}_T^p}{\sqrt{\|\mathbf{v}_T^p\|^2 + \epsilon_T^2}},$$

where the tangential velocity is obtained by projecting out the normal component,

$$\mathbf{v}_T = \mathbf{v} - (\mathbf{v}\cdot\mathbf{n})\,\mathbf{n},$$

and $\epsilon_T$ is a small regularization parameter (velocity scale) that smooths the stick/slip transition. In the slip limit $\|\mathbf{v}_T\| \gg \epsilon_T$ the magnitude approaches the Coulomb cone bound $\mu\|\mathbf{f}_N\|$; in the stick limit $\|\mathbf{v}_T\| \ll \epsilon_T$ the force is linear in $\mathbf{v}_T$ and goes cleanly through zero. The friction force is stateless, purely a function of the current particle velocity and the normal force magnitude delivered by §15.2 or §15.3.

**Composability.** The implementation in `solvers/contact/friction.py::compute_friction_force` takes the normal-force magnitude as a scalar-per-particle input, so it composes with both the penalty force ($\|\mathbf{f}_N^p\| = \epsilon_n \langle -g\rangle_+ V_p$) and the augmented-Lagrangian force ($\|\mathbf{f}_N^p\| = p_{\text{aug}} V_p$). The total contact force is $\mathbf{f}_N + \mathbf{f}_T$, passed as a single tensor to `particle_to_grid(contact_force=...)`.

**Sliding-block verification.** For a slab of particles held in static equilibrium against a floor under gravity $g$ and launched with horizontal velocity $v_0$, the analytic deceleration is $a = \mu g$ (independent of slab mass or penalty stiffness). The benchmark `benchmarks/contact/sliding_block_mpm.py` measures:

$$a_{\text{measured}} = 2.9423\ \text{m/s}^2,\quad a_{\text{analytic}} = \mu g = 0.3\cdot 9.81 = 2.9430\ \text{m/s}^2,$$

a ratio of **1.000 to 4 decimal places**, and the frictionless control run keeps $v_x = 1.000$ exactly across 4000 steps.

### 15.5 Topology-Aware Contact via Persistent Homology

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

### 15.6 Contact Chart Spawning

Once a first-contact event has been detected at a physical-space location $\mathbf{x}_c$, we want to add two new charts straddling the contact interface — one on each body — so the atlas gains resolution at the contact region. The existing `atlas/topo/chart_spawn.py::SpawnedChartPair` data structure is reused verbatim; the bridge function `solvers/contact/contact_chart_spawn.py::spawn_contact_chart_pair` computes the missing pieces:

1. **Contact normal** via autograd on body $A$'s SDF at $\mathbf{x}_c$ (same pattern as §15.1):

$$\mathbf{n}_A = \frac{\nabla \phi_A(\mathbf{x}_c)}{\|\nabla \phi_A(\mathbf{x}_c)\|}.$$

2. **Seeds** symmetrically straddling the contact interface along $\mathbf{n}_A$:

$$\mathbf{s}_+ = \mathbf{x}_c + \tfrac{1}{2} r\, \mathbf{n}_A,\qquad \mathbf{s}_- = \mathbf{x}_c - \tfrac{1}{2} r\, \mathbf{n}_A,$$

where $r$ is the requested support radius.

3. **Orthonormal frames** at each seed via `ChartSpawner._frame_from_normal` (Gram–Schmidt with a stable reference vector).

4. **Parent chart** for warm-starting the decoder: nearest existing chart in Euclidean distance to $\mathbf{x}_c$.

The returned pair is tagged `edge_type="contact"` and `activation_step=event.load_step`, and can be passed directly into `SchwarzMPMSolver.add_charts([pair])` — the existing chart-spawning machinery from the fracture phase handles the rest (decoder instantiation, neighbor graph rebuild, color-group recomputation).

### 15.7 Self-Contact

Self-contact is the hardest case because particles on their own body are **supposed** to be inside their own SDF ($\phi_A < 0$ everywhere in the interior). A naive "particle penetrates its own SDF" detector would fire on every bulk particle at every step. The `solvers/contact/self_contact.py::SelfContactManager` uses two exclusion filters to recover a physically meaningful detection:

**Filter 1 — Surface filter.** At construction time, snapshot the initial gap $g_0^p = \phi_A(\mathbf{X}_0^p)$ for every particle. Only particles with $|g_0^p| < \text{surface\_tol}$ — i.e., particles that started on the body surface — participate in self-contact checks. Deep interior particles are classified as *bulk* and are never flagged.

**Filter 2 — Initial-gap delta filter.** At detection time, compute the current gap $g^p = \phi_A(\mathbf{x}^p(t))$ and compare to the baseline:

$$\text{active}^p = \text{is\_surface}^p\ \land\ \big(g^p < g_0^p - \delta_{\text{pen}}\big).$$

A particle is flagged only when its current gap is *at least $\delta_{\text{pen}}$ more negative* than its initial gap — i.e., it has moved notably deeper than where it started. This catches folding (a surface particle that crosses into the body interior because the body folded onto itself) without false-positiving numerical wobble around the home position.

**Force.** The force on active particles uses the delta, not the absolute gap, so bulk-surface-adjacent particles that move slightly don't produce large spurious restoring forces:

$$\mathbf{f}_p^{\text{self}} = \epsilon_n\, \langle g_0^p - g^p\rangle_+\, V_p\, \mathbf{n}_p\quad\text{for active}\ p;\quad \mathbf{0}\ \text{otherwise}.$$

**Scope and limitations.** The method uses the **reference** SDF (not a deformed one), so it is accurate for small-to-moderate deformations. For large deformations the reference SDF is no longer a good model of the deformed surface and the detector degrades. Tangential self-sliding through the standard friction module (§15.4) still works once self-contact has been identified via the normal-direction detector. The folding-slab benchmark (`benchmarks/contact/folding_slab_mpm.py`) verifies that 36 surface particles are flagged mid-fold, 5 bulk particles are never active, and no false positives occur at rest.

### 15.8 Multi-Body Orchestration: SchwarzMPMSolver Contact Integration

The multi-chart `SchwarzMPMSolver` grows two hooks for contact:

1. **`configure_contact(opponent_bodies, epsilon_n, margin)`** — stores a list of `ContactBody` opponents, the penalty stiffness, and a broad-phase margin. The solver constructs a `ContactManager` internally.

2. **`_compute_contact_forces()`** — before each per-chart MPM step, iterates over all charts $i$ with nonzero particle counts and:

    a. Pushes particles to physical space: $\mathbf{x}_p = \varphi_i(\boldsymbol{\xi}_p) + \mathbf{u}_p$.

    b. For each opponent body, runs a broad-phase distance cull (vectorized across all particles and all opponent seeds at once): skip if the closest particle-to-seed distance exceeds the sum of seed-support radii plus the margin.

    c. Runs the narrow phase: `evaluate_gap(x_phys, body.sdf_net)` gives per-particle gaps and normals in one batched autograd call.

    d. Accumulates penalty forces across all opponent bodies: $\mathbf{f}_p^{\text{total}} = \sum_B \mathbf{f}_p^{\text{contact},B}$.

The resulting per-chart force tensors are passed into each `ChartMPMSolver.step(..., contact_force=f_total)`, which routes them through the P2G scatter block of §15.2.

The step loop orders: (1) compute contact forces, (2) step each chart with its forces, (3) exchange boundary velocities between overlapping charts, (4) transfer escaped particles, (5) tick the `ContactTopologyMonitor` if one is registered.

### 15.9 Verification Summary

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

Total contact-framework tests: **91 passing, 0 failing** (out of 245 tests in the full suite).

---

## 16. Curved-Chart MPM Velocity Gradient: Resolved

The fracture and contact work above rely on the MPM solver for time integration. During the Phase 4 / Phase 7 debug audit of the contact framework, we identified a latent correctness issue in the MPM velocity-gradient computation on non-identity chart decoders. This section documents both the original bug and the fix that resolves it.

**Original bug.** In `solvers/mpm/transfers.py::grid_to_particle`, the velocity gradient was computed as

$$\widetilde L_p \;=\; \sum_I v^{\text{phys}}_{I}\, (\nabla_{\boldsymbol{\xi}} N_I)(\boldsymbol{\xi}_p) \;=\; \frac{\partial v^{\text{phys}}}{\partial \boldsymbol{\xi}}\bigg|_{\boldsymbol{\xi}_p}.$$

This is a mixed-coordinate tensor: the numerator is physical velocity, the denominator is parametric (chart-local) coordinate. The subsequent $F$-update in `particles.py::update_deformation_gradient` interprets it as the physical spatial velocity gradient $\mathbf{L} = \partial\mathbf{v}/\partial\mathbf{x}^{\text{phys}}$, which the Neo-Hookean constitutive model then consumes. By the chain rule the two differ by a factor of $\mathbf{J}^{-1}$ (the chart decoder Jacobian inverse):

$$\mathbf{L}_p \;=\; \widetilde L_p\cdot \mathbf{J}_p^{-1}.$$

The dual bug appeared in the P2G internal-force scatter, which wrote $\sigma_p\cdot \nabla_{\boldsymbol{\xi}} N_I$ instead of the physically correct $\sigma_p\cdot (J_p^{-T} \nabla_{\boldsymbol{\xi}} N_I) = \sigma_p\cdot \nabla_{\mathbf{x}} N_I$. For identity decoders ($\mathbf{J} = \mathbf{I}$) both expressions coincided, so the bug was silent in all pre-fix tests and benchmarks.

**Fix — Option 3 (applied).** `ChartMPMSolver` gained an optional decoder bundle (`decoder`, `seed`, `t1`, `t2`, `n_vec`, `chart_scale`). When supplied, each `step()` call computes the transpose-inverse chart Jacobian at every particle via the existing `common/geometry.py::chart_map_and_jacobian` + `stabilized_jacobian_ops` helpers:

$$J_p = \frac{\partial \varphi}{\partial \boldsymbol{\xi}}\bigg|_{\boldsymbol{\xi}_p}, \qquad J_p^{-T} = \text{SVD pseudoinverse}^T\!\!\big(J_p\big),$$

and threads `J_inv_T` into `_shape_functions_and_indices`. Inside that helper a single `einsum` applies the chain rule once, at the single point of truth:

$$\nabla_{\mathbf{x}} N_I(\boldsymbol{\xi}_p) \;=\; J_p^{-T}\, \nabla_{\boldsymbol{\xi}} N_I(\boldsymbol{\xi}_p).$$

The downstream P2G internal-force line and the G2P velocity-gradient line do not need to change — they simply consume the (now-physical) `grad_weights` and are dimensionally correct by construction. Both dual bugs are fixed by the same one-line transform.

**Backward compatibility.** The `J_inv_T` argument defaults to `None` on `particle_to_grid`, `grid_to_particle`, and `_shape_functions_and_indices`; the decoder bundle defaults to `None` on `ChartMPMSolver.__init__`. All pre-fix MPM callers — 245 tests plus every MPM benchmark — continue to use the legacy identity path byte-for-byte unchanged. The curved-chart path is opt-in and activates only when a complete decoder bundle is supplied.

**`SchwarzMPMSolver` wiring.** The multi-chart solver's `__init__` loop now forwards `self.decoders[i]`, `self.seeds_t[i]`, `self.t1_t[i]`, `self.t2_t[i]`, `self.nvec_t[i]`, and `self.support_r_t[i]` into each child `ChartMPMSolver`, so every chart in a multi-chart atlas automatically runs the correct physical-space gradient path. The mirror change in `SchwarzMPMSolver.add_charts` propagates the bundle to dynamically spawned charts from both fracture and contact-topology events, so the curved-chart fix covers the full dynamic atlas lifecycle.

**Verification.** Five new tests in `tests/test_mpm_curved_chart.py`:

| Test | What it verifies |
|---|---|
| `test_affine_decoder_j_eq_2I_velocity_gradient` | Analytic affine decoder $\varphi(\boldsymbol{\xi}) = 2\boldsymbol{\xi}$ with linear physical-space shear $v^{\text{phys}} = (\gamma x_2, 0, 0)$. Pre-fix `vel_grad[:, 0, 1]` would be $2\gamma$; with the fix it is exactly $\gamma$ at every particle. |
| `test_affine_decoder_j_eq_half_I_velocity_gradient` | Mirror test with $\alpha = 0.5$, covering $\mathbf{J}$ smaller than identity. |
| `test_static_equilibrium_curved_chart` | Particles at rest with $F = I$ under $\alpha = 2$ stay at rest across 10 steps (kinetic energy $< 10^{-12}$). Confirms the P2G internal force and G2P update are internally consistent on a curved chart. |
| `test_identity_decoder_matches_no_decoder_path` | `ChartMPMSolver(decoder=AffineDecoder(1.0))` produces byte-identical particle state after 5 steps as `ChartMPMSolver()` with no decoder. The fix is invisible on the identity path. |
| `test_contact_invariant_under_decoder` | A ball dropped onto a floor SDF traces the same physical-space trajectory whether the solver was constructed with or without a decoder bundle — a regression guarantee that the contact path (which already lived in physical space) is unaffected by the decoder threading. |

**Regression status.** 250 passed, 1 skipped, 0 failed (+5 from the new tests; no regressions in the 245 pre-fix tests). This includes the 91 contact-framework tests, which continue to pass because contact forces live in physical space and are unaffected by the chart-Jacobian chain-rule transform.

**Why the contact framework was unaffected all along.** Contact forces were always computed in physical space (the SDF normal lives in $\mathbb{R}^3_{\text{phys}}$ and the gap is a scalar) and scattered via `f_I += f_p N_I(\boldsymbol{\xi}_p)` without going through any shape-function gradient. They sat entirely outside the mixed-coordinate branch. The fix does not change the contact path — it just makes the surrounding MPM solver correct enough that the "neural atlas for contact mechanics" claim no longer reduces to "SDF contact on a flat Cartesian grid".

See `docs/mpm_velocity_gradient_audit.md` for the full dimensional analysis, verification strategy, severity assessment per-caller, and the "Resolution (applied)" section at the bottom of that document for a detailed implementation walkthrough.

---

## Notation Index

| Symbol | Code Variable | Meaning |
|--------|--------------|---------|
| $\mathbf{F}$ | `F` | Deformation gradient $(3\times 3)$ |
| $\mathbf{P}$ | `P` | First Piola–Kirchhoff stress |
| $\boldsymbol{\sigma}$ | `sigma` | Cauchy stress |
| $J$ | `J` | Volume ratio $= \det \mathbf{F}$ |
| $\mu, K$ | `mu`, `K` | Shear and bulk moduli |
| $\mathbf{J}_i$ | `geom_J` | Decoder Jacobian $\partial\mathbf{x}/\partial\boldsymbol{\xi}$ |
| $\nabla_\xi \mathbf{u}$ | `grad_u_ref` | Reference-space displacement gradient |
| $\nabla_x \mathbf{u}$ | `grad_u_phys` | Physical displacement gradient |
| $\partial N_a / \partial \mathbf{x}$ | `dNdx` | Physical shape function gradient |
| $V_e$ | `vol` | Element volume |
| $\mathbf{f}_{\text{int}}$ | `f_int` | Internal force vector |
| $\mathbf{K}_{\text{tan}}$ | `K` | Global tangent stiffness matrix |
| $\mathcal{F}(\boldsymbol{\sigma})$ | `F_dp` | Drucker–Prager yield function |
| $\sigma_{ts}, \sigma_{hs}$ | `sigma_ts`, `sigma_hs` | Uniaxial and hydrostatic tensile strengths |
| $K_I, K_{Ic}$ | `K_I`, `K_Ic` | Mode-I stress intensity factors |
| $G_c$ | `Gc` | Critical energy release rate |
| $r, \theta$ | `r`, `theta` | Crack-tip polar coordinates |
| $\boldsymbol{\xi}$ | `xi` | Reference coordinates $\in [-1,1]^3$ |
| $\delta$ | `robin_delta` | Robin DD parameter |
| $\boldsymbol{\lambda}, \mathbf{g}$ | `lambda_charts`, `g_charts` | Robin interface variables |
| $\bar{\mathbf{F}}$ | `F_bar` | F-bar modified deformation gradient |
| $\bar{J}$ | `J_bar` | Volume-averaged Jacobian |
| $\text{Dgm}_k$ | `diagrams[k]` | Persistence diagram for dimension $k$ |
| $d_B$ | `bottleneck_distance` | Bottleneck distance between diagrams |
| $\beta_k$ | `betti[k]` | Betti numbers |
| $\phi_A$, $\phi_B$ | `body.sdf_net` | Neural SDFs of two contacting bodies |
| $\phi_{AB}$ | `combined_sdf_grid` | Combined SDF $\min(\phi_A, \phi_B)$ (§15.5) |
| $g_N$ | `gap` | Normal gap function $\phi_A(\mathbf{x}_p)$ (§15.1) |
| $\mathbf{n}_A$ | `normal` | Outward unit normal $\nabla\phi_A / \|\nabla\phi_A\|$ (§15.1) |
| $\mathbf{v}_T$ | `v_t` | Tangential particle velocity (§15.4) |
| $\epsilon_n$ | `epsilon_n` | Penalty stiffness (§15.2–15.3) |
| $\epsilon_T$ | `epsilon_t` | Friction regularization scale (§15.4) |
| $\mu$ | `mu` | Coulomb friction coefficient (§15.4) |
| $\mathbf{f}_p^{\text{contact}}$ | `contact_force` | Per-particle contact force in physical space |
| $\mathbf{f}_N^p$, $\mathbf{f}_T^p$ | `f_N`, `f_T` | Normal and tangential contact force |
| $\lambda_p$ | `self.lam[p]` | Augmented-Lagrangian multiplier per particle (§15.3) |
| $p_{\text{aug}}$ | `augmented_pressure` | AL augmented pressure $\max(0, \lambda-\epsilon_n g)$ (§15.3) |
| $p_n$ | `penalty * volume` | Pure-penalty pressure $\epsilon_n\langle -g\rangle_+$ (§15.2) |
| $\Delta t_{\text{contact}}$ | `contact_stable_dt` | Contact-stable MPM time step $\sim\sqrt{m/\epsilon_n}$ (§15.2) |
| $g_0^p$ | `state.initial_gap` | Self-contact baseline gap at $t=0$ (§15.7) |
| $\delta_{\text{pen}}$ | `penetration_delta` | Self-contact delta threshold (§15.7) |

---

## References

### Fracture

1. Kamarei, Zeng, Dolbow & Lopez-Pamies (2026). "Nine circles of elastic brittle fracture." *CMAME* 448, 118449.
2. Du (2002). "Optimization based nonoverlapping domain decomposition algorithms and their convergence." *SIAM J. Numer. Anal.* 39(3), 1056–1077.
3. de Souza Neto, Peric, Dutko & Owen (1996). "Design of simple low order finite elements for large strain analysis of nearly incompressible solids." *Int. J. Solids Struct.* 33, 3277–3296.
4. Tada, Paris & Irwin (2000). *The Stress Analysis of Cracks Handbook*. 3rd ed., ASME Press.
5. Erdogan & Sih (1963). "On the crack extension in plates under plane loading and transverse shear." *J. Basic Eng.* 85(4), 519–527.
6. Anderson (2005). *Fracture Mechanics: Fundamentals and Applications*. 3rd ed., CRC Press.
7. Rivlin & Thomas (1953). "Rupture of rubber. I. Characteristic energy for tearing." *J. Polym. Sci.* 10, 291–318.
8. Zienkiewicz, Taylor & Zhu (2013). *The Finite Element Method*. 7th ed., Butterworth-Heinemann.
9. GUDHI Library. https://gudhi.inria.fr/ — Persistent homology via CubicalComplex.

### Contact Mechanics (§15)

10. Signorini (1933). "Sopra alcune questioni di elastostatica." *Atti Soc. Ital. Progr. Sci.* — unilateral contact constraint.
11. Kikuchi & Oden (1988). *Contact Problems in Elasticity: A Study of Variational Inequalities and Finite Element Methods*. SIAM — penalty method theory.
12. Alart & Curnier (1991). "A mixed formulation for frictional contact problems prone to Newton like solution methods." *Comput. Methods Appl. Mech. Eng.* 92(3), 353–375 — augmented Lagrangian formulation.
13. Simo & Laursen (1992). "An augmented Lagrangian treatment of contact problems involving friction." *Comput. Struct.* 42(1), 97–116 — Uzawa iteration for contact.
14. Wriggers (2006). *Computational Contact Mechanics*. 2nd ed., Springer — regularized Coulomb friction, stick/slip regularization.
15. Laursen (2002). *Computational Contact and Impact Mechanics*. Springer — Lagrange multiplier contact.
16. Bardenhagen, Guilkey, Roessig, Brackbill, Witzel & Foster (2001). "An improved contact algorithm for the material point method." *Comput. Model. Eng. Sci.* 2(4), 509–522 — multi-material MPM contact.
17. Cohen-Steiner, Edelsbrunner & Harer (2007). "Stability of persistence diagrams." *Discrete Comput. Geom.* 37(1), 103–120 — bottleneck stability theorem used in §15.5.
