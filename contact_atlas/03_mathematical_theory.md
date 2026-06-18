# Mathematical Theory: Contact Mechanics on Neural Atlas Domains

> **Historical note.** Design-history theory doc. Some FEM-assembly passages reference
> `ChartVectorFEMSolver` and related modules now under `archive/solvers/fem/`; the implemented
> framework is MPM-based (see `docs/contact_theory_manual.md`). Kept for the variational/well-posedness
> rationale, not as an active spec.

## 1. Problem Setting

### 1.1 Neural Atlas Domain Representation

Let Omega subset R^3 be a solid body. In the neural atlas framework, Omega is represented by:

1. **Neural SDF.** A learned function phi: R^3 -> R satisfying the Eikonal equation |nabla phi| = 1 a.e., with Omega = {x : phi(x) < 0} and partial Omega = {x : phi(x) = 0}.

2. **Atlas.** A collection of M charts {(U_i, varphi_i)}_{i=1}^M where:
   - U_i subset R^3 is an open parametric domain (reference cube [-r_i, r_i]^3)
   - varphi_i: U_i -> R^3 is a learned chart decoder (MLP with linear base + nonlinear residual)
   - The images V_i = varphi_i(U_i) cover Omega: Omega subset union_i V_i

3. **Partition of Unity.** Smooth functions {psi_i}_{i=1}^M with psi_i: R^3 -> [0,1] (MaskNet outputs composed with softmax) satisfying sum_i psi_i(x) = 1 for x in Omega.

4. **Chart Jacobian.** J_i(xi) = D varphi_i / D xi in R^{3x3}, with det(J_i) > 0 (ensured by training).

### 1.2 Two-Body Contact Configuration

Consider two bodies Omega^A and Omega^B with:
- Neural SDFs: phi^A, phi^B
- Atlases: {(U_i^A, varphi_i^A)}_{i=1}^{M_A} and {(U_j^B, varphi_j^B)}_{j=1}^{M_B}
- Deformation maps: x^A = X^A + u^A(X^A) and x^B = X^B + u^B(X^B)
- Potential contact surface: Gamma_c subset partial Omega^A intersect partial Omega^B (in deformed configuration)

---

## 2. Kinematics of Contact on Mapped Domains

### 2.1 Gap Function via Neural SDF

For a material point X^B on body B's reference boundary (partial Omega^B_0), the deformed position is:

```
x^B = varphi_j^B(xi^B) + u^B(varphi_j^B(xi^B))
```

where xi^B in U_j^B is the parametric coordinate in chart j. The **normal gap function** is:

```
g_N(xi^B) = phi^A(x^B(xi^B))
```

This is the signed distance from the deformed point on B to the (reference) surface of A. For the deformed surface of A:

```
g_N^{def}(xi^B) = phi^A( (varphi_i^A)^{-1}(x^B - u^A(x^B)) )
```

which requires the inverse decoder map (varphi_i^A)^{-1}, obtained via Newton iteration.

**Contact condition (Signorini):**

```
g_N >= 0,    t_N >= 0,    t_N * g_N = 0    on Gamma_c
```

where t_N is the contact pressure (normal traction).

### 2.2 Contact Normal from SDF Gradient

The outward unit normal to body A at contact point x is:

```
n^A(x) = nabla phi^A(x) / |nabla phi^A(x)|
```

By the Eikonal property, |nabla phi^A| approx 1, so n^A approx nabla phi^A. The tangent plane at x is:

```
T_x(partial Omega^A) = {v in R^3 : v . n^A(x) = 0}
```

An orthonormal basis {tau_1, tau_2} for T_x is constructed via Gram-Schmidt from n^A (as in `_frame_from_normal` in the chart spawner).

### 2.3 Gap Rate and Velocity Decomposition

The gap rate (time derivative of gap):

```
g_dot_N = n^A . (v^B - v^A)
```

where v^A, v^B are material velocities at the contact point. The **tangential slip velocity**:

```
v_T = (v^B - v^A) - g_dot_N * n^A = (I - n^A (x) n^A) . (v^B - v^A)
```

### 2.4 Surface Jacobian and Area Transform

Under the chart map varphi_i, the reference boundary partial Omega_0 in parametric space maps to physical space. The surface area element transforms as:

```
dA = |J_i^{-T} N_ref| * |det(J_i)| * dA_ref
```

where N_ref is the outward normal in parametric space. Under additional deformation u:

```
dA_{def} = |cof(F) . N_0| * dA_0 = |det(F)| * |F^{-T} N_0| * dA_0
```

(Nanson's formula), where F = I + nabla u is the deformation gradient and N_0 is the reference normal.

---

## 3. Variational Formulation

### 3.1 Weak Form with Contact

Find u^A in V^A, u^B in V^B, and lambda in Lambda (contact multiplier space) such that:

**Equilibrium (body A):**
```
a^A(u^A, delta_u^A) + integral_{Gamma_c} lambda * delta_g_N^A dA = L^A(delta_u^A)
    for all delta_u^A in V^A
```

**Equilibrium (body B):**
```
a^B(u^B, delta_u^B) - integral_{Gamma_c} lambda * delta_g_N^B dA = L^B(delta_u^B)
    for all delta_u^B in V^B
```

**Contact constraint:**
```
integral_{Gamma_c} delta_lambda * g_N dA >= 0,    lambda >= 0
    for all delta_lambda in Lambda, delta_lambda >= 0
```

Here:
- a^alpha(u, v) = integral_{Omega^alpha} P(F) : nabla v dV is the internal virtual work (P = first Piola-Kirchhoff stress)
- L^alpha(v) = integral_{Omega^alpha} b . v dV + integral_{Gamma_N^alpha} t_bar . v dA is the external virtual work
- delta_g_N^A = -n^A . delta_u^A and delta_g_N^B = n^A . delta_u^B are the gap variations

### 3.2 Chart-Local Assembly of Contact Virtual Work

The contact integral on Gamma_c must be assembled from chart-local contributions. For a contact point at parametric coordinate xi^B in chart j of body B:

```
delta W_c = integral_{Gamma_c^j} lambda(xi^B) * [n^A . delta_u^B(varphi_j^B(xi^B))] * |cof(F^B) N_0^B| dA_0^B(xi^B)
```

In the discretized setting with P1 elements on the Freudenthal grid, the boundary face integral becomes a sum over boundary triangles with 1-point quadrature (centroid).

### 3.3 Mapped Bilinear Form on Charts

On chart i of body alpha, the bilinear form is:

```
a_i^alpha(u, v) = integral_{U_i} P(F(xi)) : [nabla_xi v * J_i^{-1}(xi)] * |det(J_i(xi))| dxi
```

where F(xi) = I + nabla_xi u * J_i^{-1} is the deformation gradient in physical space, computed from the chart-local displacement gradient and the decoder Jacobian.

The (archived) `ChartVectorFEMSolver` assembled this via:
```
K_i[a,b] = integral_{U_i} (nabla_xi N_a . J_i^{-1})^T : C : (nabla_xi N_b . J_i^{-1}) |det(J_i)| dxi
```

Contact forces are added to the right-hand side of each chart's system.

---

## 4. Contact Formulations on Neural Atlas

### 4.1 Penalty Regularization

Replace the exact constraint with a penalized version. The contact virtual work becomes:

```
delta W_c^{pen} = integral_{Gamma_c} epsilon_n * <g_N>_- * delta_g_N dA
```

where <x>_- = max(0, -x) is the Macaulay bracket (negative part). This is equivalent to adding a normal spring with stiffness epsilon_n at each contact point.

**Linearization for Newton's method:**

```
D(delta W_c^{pen}) = integral_{Gamma_c} [epsilon_n * H(-g_N) * delta_g_N * Delta_g_N
                                         + epsilon_n * <g_N>_- * D(delta_g_N)] dA
```

where H is the Heaviside function and:

```
D(delta_g_N) = -(nabla^2 phi^A)(delta_u, Delta_u) + kappa * (delta_u_T . Delta_u_T)
```

involves the SDF Hessian nabla^2 phi^A (curvature of the contact surface). The curvature term kappa = principal curvatures of the SDF level set, computable via:

```
kappa_ij = (delta_ij - n_i n_j) * (partial^2 phi^A / partial x_i partial x_j) / |nabla phi^A|
```

This is the **shape operator** (Weingarten map) of the contact surface, directly available from the neural SDF via second-order autograd.

### 4.2 Augmented Lagrangian (Alart-Curnier)

Introduce multiplier lambda^k (updated in outer loop) and solve the augmented problem:

```
delta W_c^{AL} = integral_{Gamma_c} max(0, lambda^k - epsilon_n * g_N) * delta_g_N dA
```

The multiplier update (Uzawa):

```
lambda^{k+1} = max(0, lambda^k - epsilon_n * g_N(u^{k+1}))
```

**Convergence:** For epsilon_n > 0, the Uzawa iteration converges at rate:

```
||lambda^{k+1} - lambda^*|| <= rho * ||lambda^k - lambda^*||
```

where rho = 1 - c * epsilon_n / (epsilon_n + K_max) < 1, with K_max the maximum eigenvalue of the stiffness operator restricted to the contact surface. Choosing epsilon_n ~ O(E * h^{-1}) (E = Young's modulus, h = mesh size) gives rho ~ 1/2, i.e., convergence in ~10 iterations.

### 4.3 Nitsche's Method on Neural Atlas

The Nitsche contact formulation adds consistency and penalty terms:

```
delta W_c^{Nit} = -integral_{Gamma_c} {sigma . n} * [delta_u] dA         (consistency)
                  - integral_{Gamma_c} {delta_sigma . n} * [g_N]_+ dA     (adjoint consistency)
                  + integral_{Gamma_c} (gamma/h) * [g_N]_+ * delta_g_N dA  (penalty)
```

where {.} denotes the weighted average across the interface, [.] denotes the jump, and gamma > 0 is the stabilization parameter.

**Connection to Robin Schwarz:** The Robin transmission condition on chart interface Gamma_ij is:

```
sigma_i . n + delta * u_i = sigma_j . n + delta * u_j    on Gamma_ij
```

This is exactly the Nitsche interface condition with gamma = delta * h. For contact, replace the equality with:

```
sigma_i . n + delta * u_i >= 0,    lambda >= 0,    lambda * (sigma_i . n + delta * u_i) = 0
```

This is the **Robin contact complementarity** problem, solved by projecting onto the non-negative cone.

### 4.4 Mortar Contact on Non-Matching Chart Grids

The mortar contact constraint is:

```
integral_{Gamma_c^m} delta_lambda * (u^B - u^A - g_0) dA = 0
    for all delta_lambda in Lambda_h
```

where Gamma_c^m is the mortar (master) surface and g_0 is the initial gap.

**PoU-weighted mortar integral:** Using the mask functions psi_i, the mortar integral is decomposed into chart contributions:

```
integral_{Gamma_c^m} = sum_{i in A, j in B} integral_{Gamma_c^m intersect V_i intersect V_j} psi_i * psi_j * (...) dA
```

The MaskNet partition-of-unity weights provide the natural mortar weighting. This avoids the standard difficulty of choosing master/slave surfaces: the PoU symmetrizes the formulation.

---

## 5. Schwarz Domain Decomposition with Contact

### 5.1 Multiplicative Schwarz with Contact Subproblems

Partition the charts into color groups C_1, ..., C_K (no two overlapping charts share a color). The multiplicative Schwarz iteration with contact:

**Algorithm (Schwarz-Contact):**
```
Given u^0 (initial guess), lambda^0 = 0

For k = 1, 2, ..., K_max:
    For each color group C_l, l = 1, ..., K:
        For each chart i in C_l (parallel within color):
            1. Compute Schwarz BCs from neighbors:
               u_i^{art} = sum_j psi_j(x) * u_j^{k-1}(varphi_j^{-1}(x))  (PoU blend)

            2. Compute contact forces:
               For each contact pair involving chart i:
                   g_N = phi^{opponent}(x_i^{bnd} + u_i^{bnd})
                   f_c = epsilon_n * max(0, -g_N) * n + lambda^{k-1}

            3. Solve chart subproblem:
               a_i(u_i, v) = L_i(v) - integral_{Gamma_c^i} f_c . v dA
               subject to: u_i = u_i^{art} on artificial boundary
                           u_i = u_i^{phys} on physical Dirichlet boundary

    Check convergence: max_i ||u_i^k - u_i^{k-1}|| / ||u_i^k|| < tol
```

### 5.2 Parallel Robin Schwarz with Contact

The Robin variant exchanges both displacement and traction, enabling parallel solution:

**Algorithm (Robin-Contact):**
```
Given u^0, lambda_R^0 = 0, g_R^0 = 0

For k = 1, 2, ...:
    # Update interface data (includes contact projection)
    For each chart i:
        u_avg = (1/|N_i|) sum_{j in N_i} u_j^{k-1}(bnd_nodes_i)

        # Standard Robin update for Schwarz overlaps:
        lambda_R_i^k = omega * (u_i^{k-1} + u_avg)/2 + (1-omega) * lambda_R_i^{k-1}
        g_R_i^k = g_R_i^{k-1} + delta * (u_avg - u_i^{k-1})/2

        # Contact Robin update (projected):
        For contact interfaces:
            lambda_c_i^k = max(0, lambda_c_i^{k-1} + epsilon * gap_i^{k-1})
            g_c_i^k = g_c_i^{k-1} + delta * (u_opponent - u_i^{k-1})/2

    # Solve all charts in parallel with Robin + contact BCs:
    For each chart i (all parallel):
        Solve: a_i(u_i, v) + delta * integral_{partial_art} u_i . v dA
             = L_i(v) + integral_{partial_art} (delta * lambda_R + g_R) . v dA
                       + integral_{Gamma_c} (delta * lambda_c + g_c) . v dA

    Check convergence
```

### 5.3 Convergence Theory

**Theorem (Schwarz-Contact Convergence).** Under the following assumptions:
1. Each body satisfies standard ellipticity: a^alpha(v, v) >= alpha_0 ||v||_{H^1}^2
2. The contact penalty epsilon_n < C * h^{-2} (bounded from above)
3. Chart overlaps satisfy a generous overlap condition: delta_overlap >= c * H (H = chart diameter)
4. The PoU partition {psi_i} is uniformly bounded: C_1 <= sum_i |nabla psi_i|^2 <= C_2

Then the multiplicative Schwarz-contact iteration converges with rate:

```
||u^{k+1} - u^*||_a <= rho * ||u^k - u^*||_a
```

where rho depends on the overlap ratio, the number of colors, and the contact stiffness epsilon_n. The rate degrades as epsilon_n -> infinity (penalty becomes stiff) but is bounded below by the overlap-dependent Schwarz rate.

**Proof sketch.** The contact penalty adds a non-negative term to the bilinear form: a_c(u,v) = epsilon_n * integral max(0,-g) * delta_g dA >= 0. This preserves coercivity. The subproblem contraction follows from the standard Schwarz lemma (Lions 1988) applied to the augmented bilinear form a + a_c. The contact active set introduces a nonlinearity, but for a fixed active set, the problem is linear and Schwarz convergence applies. Active set changes are bounded (monotone after sufficient iterations by the theory of Hintermuller et al. 2002).

**Robin variant:** Convergence rate for Robin-contact follows from the analysis of Du (2002, SINUM 39). The Robin parameter delta must satisfy delta >= c * E/h for stability, where E is the Young's modulus and h is the mesh size. The contact projection (max(0, ...)) preserves the contraction property because it is a non-expansive operator on L^2.

---

## 6. Friction Theory

### 6.1 Coulomb Friction Law

The tangential traction t_T and tangential velocity v_T satisfy:

```
|t_T| <= mu * t_N                          (friction cone)
|t_T| < mu * t_N  =>  v_T = 0              (stick)
|t_T| = mu * t_N  =>  exists gamma >= 0:   (slip)
                       v_T = -gamma * t_T / |t_T|
```

This is a **non-smooth, non-associative** constitutive law (no potential for friction + normal contact jointly).

### 6.2 Regularized Friction

Replace the non-smooth slip rule with:

```
t_T = -mu * t_N * v_T / sqrt(|v_T|^2 + epsilon_T^2)
```

This smooths the stick-slip transition over a velocity scale epsilon_T. The regularized law derives from a pseudo-potential:

```
Psi_T(v_T) = mu * t_N * sqrt(|v_T|^2 + epsilon_T^2)
t_T = -partial Psi_T / partial v_T
```

making the regularized problem variational.

### 6.3 Friction in Chart-Local Coordinates

In chart i, the tangential velocity is decomposed using the SDF-derived frame:

```
v_T = (I - n (x) n) . v = v - (v . n) * n
```

In parametric coordinates:

```
v_T^{xi} = J_i^{-1} . v_T
```

The frictional virtual work contribution in chart i is:

```
delta W_f^i = integral_{Gamma_c^i} t_T . delta_u_T * |det(J_i)| * |J_i^{-T} N_{ref}| dA_{ref}
```

### 6.4 Augmented Lagrangian for Friction (Alart-Curnier)

Introduce a tangential multiplier lambda_T (friction traction). The augmented formulation:

**Normal:**
```
t_N^{aug} = max(0, lambda_N + epsilon_n * g_N)
```

**Tangential:**
```
t_T^{trial} = lambda_T - epsilon_t * v_T * dt
If |t_T^{trial}| <= mu * t_N^{aug}:    (stick)
    t_T^{aug} = t_T^{trial}
Else:                                    (slip)
    t_T^{aug} = mu * t_N^{aug} * t_T^{trial} / |t_T^{trial}|
```

**Multiplier updates:**
```
lambda_N^{k+1} = t_N^{aug}
lambda_T^{k+1} = t_T^{aug}
```

This is the Alart-Curnier formulation (1991), which reduces the frictional contact problem to a sequence of augmented penalty subproblems.

---

## 7. Large Deformation Contact Theory

### 7.1 Deformed-Configuration Gap via Composed SDF

For finite deformations, the gap must be evaluated on the deformed surface. Given the deformation map chi^A: Omega_0^A -> R^3, the deformed SDF of body A is:

```
phi^A_{def}(x) = phi^A_0((chi^A)^{-1}(x))
```

In the neural atlas framework, (chi^A)^{-1}(x) requires:
1. Finding which chart i covers point x: use PoU weights psi_i(x) > threshold
2. Inverting the composed map: find xi such that varphi_i(xi) + u(varphi_i(xi)) = x
3. This is a Newton iteration: xi_{n+1} = xi_n - [J_i + (nabla u) J_i]^{-1} [varphi_i(xi_n) + u(varphi_i(xi_n)) - x]

The Jacobian of the composed map is:

```
D(varphi_i + u circ varphi_i) / D xi = (I + nabla_x u) . J_i = F . J_i
```

where F = I + nabla u is the deformation gradient. Invertibility requires det(F) > 0 and det(J_i) > 0, both maintained by the framework.

### 7.2 Linearization of Deformed Gap

The gap linearization at a deformed contact point x^B = X^B + u^B:

```
D g_N = nabla phi^A(x^B) . (delta_u^B - delta_u^A)
      + (nabla^2 phi^A(x^B) . Delta_u) . (delta_u^B - delta_u^A)
      + ...
```

The first term gives the **geometric stiffness** (first-order gap change). The second term involves the SDF Hessian and contributes to the **curvature stiffness**. For flat contact surfaces (Hessian ~ 0), only the geometric stiffness is needed.

### 7.3 Objectivity and Frame Invariance

The neural SDF phi^A is defined in a fixed spatial frame. Under a superimposed rigid body motion Q . x + c (Q orthogonal), the SDF transforms as:

```
phi^A(Q . x + c) = phi^A(x)    (if the body rotates with the motion)
```

This is automatically satisfied if the SDF is re-evaluated at the material point's position relative to body A's reference frame. The chart decoder provides this: varphi_i maps from a body-fixed parametric space, so the composed gap function is objective.

For the contact normal:

```
n^A(Q . x + c) = Q . n^A(x)
```

which transforms correctly under rotation, preserving objectivity of the contact formulation.

---

## 8. Topology of Contact

### 8.1 Persistent Homology of the Contact Configuration

Consider the combined SDF field:

```
phi_{AB}(x) = min(phi^A(x), phi^B(x))
```

This represents the union Omega^A union Omega^B. The sublevel sets {phi_{AB} < t} evolve as t increases from -infinity to 0:

- At t << 0: Two separate interior regions (beta_0 = 2)
- At contact (t = t_c): The two regions merge (beta_0 decreases by 1)
- At t = 0: Full combined domain

A **death event in H_0** at filtration value t_c in the persistence diagram of phi_{AB} indicates contact between the two bodies. The birth-death pair (t_birth, t_c) represents the second connected component that merges at t_c.

### 8.2 Lusternik-Schnirelmann Category for Contact

The LS category of the combined domain Omega^A union Omega^B depends on the topology of the contact region:

- **Point contact:** cat(Omega^A union Omega^B) = max(cat(Omega^A), cat(Omega^B))
- **Surface contact (genus-changing):** cat may increase, requiring additional charts

The minimum chart count for the combined domain is:

```
M_min^{AB} = cat(Omega^A union Omega^B) + 1
```

If M_min^{AB} > M_A + M_B (the sum of individual chart counts), additional contact charts are needed. This motivates the topology-driven chart spawning at contact interfaces.

### 8.3 Contact-Induced Topological Events

| Event | Homological Signature | SDF Indicator |
|---|---|---|
| First contact | H_0 death (2 components -> 1) | min(phi^A, phi^B) < 0 at some point |
| Contact area growth | H_0 unchanged, contact surface area increases | Level set {phi^A = 0} intersect {phi^B < epsilon} grows |
| Enclosure (body B inside A) | H_2 birth (new void) | phi^A(x^B) < 0 for all x^B in partial Omega^B |
| Separation | H_0 birth (1 component -> 2) | min(phi^A, phi^B) > 0 everywhere |
| Self-contact | H_1 birth (new loop) | phi^A(x^A) < 0 for non-adjacent surface points |

---

## 9. Well-Posedness and Stability

### 9.1 Existence of Solutions

**Theorem.** Under the following conditions:
1. Each body's elastic energy W(F) is polyconvex (e.g., Neo-Hookean)
2. The contact constraint set K = {(u^A, u^B) : g_N >= 0 on Gamma_c} is non-empty
3. Applied loads are in L^2(Omega) and Dirichlet data is in H^{1/2}(Gamma_D)

There exists a minimizer (u^A, u^B) in K of the total energy:

```
E(u^A, u^B) = sum_{alpha=A,B} integral_{Omega^alpha} W(I + nabla u^alpha) dV
             - sum_{alpha=A,B} L^alpha(u^alpha)
```

**Proof sketch.** Polyconvexity implies weak lower semicontinuity of E. The constraint set K is weakly closed in H^1 x H^1 (trace inequality + continuity of the gap operator). Coercivity from the growth conditions on W gives a minimizing sequence bounded in H^1. Weak compactness in H^1 + weak closedness of K + weak lower semicontinuity of E yields existence (Ball 1977, Ciarlet 1988).

### 9.2 Stability of Penalty Regularization

The penalty solution u_epsilon satisfies:

```
||u_epsilon - u^*||_{H^1} <= C / sqrt(epsilon_n)
```

where u^* is the exact (Lagrange multiplier) solution. The contact pressure lambda_epsilon = epsilon_n * max(0, -g_N(u_epsilon)) satisfies:

```
||lambda_epsilon - lambda^*||_{H^{-1/2}(Gamma_c)} <= C * sqrt(epsilon_n)
```

So the displacement converges at rate O(epsilon_n^{-1/2}) and the pressure at O(epsilon_n^{1/2}). The augmented Lagrangian achieves exact enforcement (lambda^k -> lambda^*) without requiring epsilon_n -> infinity.

### 9.3 Inf-Sup Condition for Mortar Contact

The mortar contact formulation requires the inf-sup (LBB) condition:

```
sup_{v in V_h} (integral_{Gamma_c} lambda_h * v . n dA) / ||v||_{H^1}
>= beta * ||lambda_h||_{H^{-1/2}(Gamma_c)}
for all lambda_h in Lambda_h
```

In the neural atlas context, the PoU-weighted mortar integral inherits the inf-sup stability from the underlying P1 discretization, provided the mortar space Lambda_h is chosen appropriately (e.g., piecewise constant on the coarser side). The MaskNet weights psi_i are bounded below by a positive constant on their support, ensuring the weighted integral is non-degenerate.

---

## 10. Error Analysis

### 10.1 Sources of Error in Neural Atlas Contact

| Error Source | Magnitude | Dependence |
|---|---|---|
| SDF training (Eikonal) | O(epsilon_SDF) ~ 1e-3 | Network capacity, training epochs |
| Chart decoder approximation | O(epsilon_dec) | MLP width/depth, training |
| FEM discretization | O(h^p) | Mesh size h, polynomial order p=1 |
| Penalty regularization | O(epsilon_n^{-1/2}) | Penalty parameter |
| Schwarz iteration | O(rho^k) | Overlap ratio, number of iterations k |
| Newton linearization | O(||delta_u||^2) | Increment size (quadratic) |

### 10.2 Total Error Bound

The total error in the contact pressure is bounded by:

```
||lambda_h - lambda^*|| <= C_1 * h + C_2 / sqrt(epsilon_n) + C_3 * epsilon_SDF + C_4 * rho^k
```

For practical accuracy:
- h ~ 1e-2 (mesh size in chart parametric space)
- epsilon_n ~ E/h ~ 1e4 (Young's modulus / mesh size)
- epsilon_SDF ~ 1e-3 (SDF training accuracy)
- k ~ 10-20 Schwarz iterations (rho ~ 0.5)

The SDF error epsilon_SDF ~ 1e-3 sets the practical accuracy floor for the gap function. This means contact pressures are accurate to within O(epsilon_n * epsilon_SDF) ~ O(E * 1e-3/h) in absolute terms.

### 10.3 A Priori Error Estimate (Penalty FEM on Charts)

**Theorem.** Let u_h^{eps} be the P1 FEM solution with penalty contact on the neural atlas. Under the regularity assumption u^* in H^2(Omega):

```
||u_h^{eps} - u^*||_{H^1(Omega)} <= C * (h + epsilon_n^{-1/2} + epsilon_SDF + epsilon_dec)
```

where:
- h: mesh size (Freudenthal grid spacing in parametric space)
- epsilon_n: penalty parameter
- epsilon_SDF: SDF approximation error (L^infinity norm)
- epsilon_dec: decoder approximation error (L^infinity norm)

The proof follows from: (a) standard P1 FEM interpolation (Cea's lemma), (b) penalty error bound (Kikuchi-Oden), (c) perturbation of the gap function by epsilon_SDF (Lipschitz dependence of the variational inequality on data), (d) perturbation of the domain by epsilon_dec (Strang-type lemma for geometry error).

---

## 11. MPM Contact Theory

### 11.1 Weak Form for MPM Contact

In MPM, the weak form is discretized by particles rather than elements. The contact virtual work contribution for particle p on body B:

```
delta W_c^p = epsilon_n * max(0, -g_N(x_p)) * (n . delta_v_p) * V_p * dt
```

where V_p is the particle volume and dt is the time step. This force is scattered to the grid during P2G:

```
f_I^{contact} = sum_p epsilon_n * max(0, -g_N(x_p)) * n * N_I(xi_p) * V_p
```

where N_I is the shape function for grid node I evaluated at the particle's parametric position xi_p.

### 11.2 Stability Condition

The explicit MPM time integrator with penalty contact requires:

```
dt <= min(dt_CFL, dt_contact)
```

where:
- dt_CFL = alpha * h / c_s (CFL condition, c_s = wave speed, h = grid spacing)
- dt_contact = beta * sqrt(m_min / epsilon_n) (contact stability, m_min = minimum particle mass)

For stiff contact (epsilon_n >> E), the contact stability dominates: dt_contact << dt_CFL. Mitigations:
1. Use moderate epsilon_n and accept O(1/epsilon_n) penetration
2. Implicit or semi-implicit contact treatment (requires matrix solve)
3. Barrier methods with logarithmic penalty: phi_barrier = -kappa * log(g_N) for g_N > 0 (IPC, Li et al. 2020)

### 11.3 Chart-Local MPM Contact Force

In chart j of body B, a contact particle at xi_p^B has:
- Physical position: x_p = varphi_j^B(xi_p^B) + u_p (tracked by particle)
- Gap: g_N = phi^A(x_p)
- Contact force (physical): f_phys = epsilon_n * max(0, -g_N) * n^A
- Contact force (chart-local): f_xi = J_j^{-T} f_phys

The force pull-back f_xi = J_j^{-T} f_phys ensures consistency with the virtual work transformation:

```
f_phys . delta_x = f_phys . (J_j delta_xi) = (J_j^T f_phys) . delta_xi = f_xi . delta_xi
```

Wait — this gives f_xi = J_j^T f_phys (transpose, not inverse-transpose). The correct transformation depends on whether we work with covariant or contravariant components:

- **Force as covector (work-conjugate to displacement):** f_xi = J_j^T f_phys
- **Force as vector (Cartesian):** f_xi = J_j^{-1} f_phys

In the MPM implementation where grid forces are assembled via f_I = sum_p f_p * N_I(xi_p), the force f_p should be in the same space as the grid velocity. Since grid velocity is in physical space (the MPM solver stores `particles.v` in physical space), the contact force should also be in physical space — no pull-back is needed.

**Correction:** Re-examining `solvers/mpm/transfers.py`, forces are assembled in physical space (gravity `g` is in physical space and is added directly to grid forces). Therefore, **contact forces in physical space are added directly**, with no Jacobian transformation required for the current MPM implementation.

---

## 12. Summary of Key Mathematical Results

| Result | Statement | Implication |
|---|---|---|
| SDF gap differentiability | phi^A in C^inf(R^3) (tanh MLP) | Newton linearization well-defined |
| Eikonal property | \|nabla phi\| ~ 1 | Gap = distance, normals = unit vectors |
| Penalty convergence | \|u_eps - u*\|_{H^1} = O(eps^{-1/2}) | Need eps_n ~ 1/h^2 for O(h) accuracy |
| Uzawa convergence | \|lambda^{k+1} - lambda*\| <= rho \|lambda^k - lambda*\| | 3-10 outer iterations suffice |
| Schwarz + contact | Converges if overlap > c*H | Standard Schwarz theory extends |
| Objectivity | phi(Qx + c) = phi(x) in body frame | Contact formulation is frame-invariant |
| LS category bound | M_min = cat(Omega^A union Omega^B) + 1 | May need contact charts |
| MPM stability | dt <= sqrt(m_min / eps_n) | Stiff contact limits time step |
| Total error | O(h + eps_n^{-1/2} + eps_SDF + eps_dec) | SDF accuracy is practical floor |
