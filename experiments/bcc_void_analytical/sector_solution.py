"""
Sector solution for cylindrical void in BCC {110}<111> single crystal.

Uses Rice's (1973) anisotropic slip-line theory to derive the
piecewise-analytical stress field around a cylindrical void under
equibiaxial far-field compression.

Framework (from Rice 1973, Kysar 2005):
  1. The yield surface in 2D Mohr stress plane is a polygon.
  2. Around the void, the stress state traverses this polygon.
  3. In each angular sector, the stress lies on one face of the polygon.
  4. The generalized Hencky equations determine stress variation along
     the slip-line characteristics within each sector.
  5. Sector boundaries are determined by stress continuity conditions.

Key equations (Rice 1973):
  Along α-lines: σ_m + s = const    (s = arc length on yield contour)
  Along β-lines: σ_m - s = const
  where σ_m = (σ_11 + σ_22)/2 is the mean in-plane stress.

For rigid-ideally plastic material, within each sector where a single
effective slip system is active (stress on one face of yield polygon):
  - The stress state is determined by σ_m (which varies with position)
  - The deviatoric part is fixed (lies on the yield surface face)

The equilibrium equations in polar coords (r, θ):
  ∂σ_rr/∂r + (1/r)∂σ_rθ/∂θ + (σ_rr - σ_θθ)/r = 0
  ∂σ_rθ/∂r + (1/r)∂σ_θθ/∂θ + 2σ_rθ/r = 0

Within a sector where the deviatoric stress is constant:
  σ_rr = σ_m + A cos(2θ) + B sin(2θ)    (A, B fixed by active face)
  σ_θθ = σ_m - A cos(2θ) - B sin(2θ)
  σ_rθ = -A sin(2θ) + B cos(2θ)

  where A = (σ_11 - σ_22)/2 and B = σ_12 on the active yield face.

  Substituting into equilibrium: ∂σ_m/∂r = 0 (σ_m = σ_m(θ) only within a sector)
  and the θ-equilibrium gives dσ_m/dθ as a function of A, B.

Actually, for the Kysar approach, the stress components in polar coords
are expressed as:
  σ_rr(r, θ) = σ_m(θ) - τ_k cos(2θ - 2φ_k)
  σ_θθ(r, θ) = σ_m(θ) + τ_k cos(2θ - 2φ_k)
  σ_rθ(r, θ) = τ_k sin(2θ - 2φ_k)

where τ_k and φ_k characterize the active yield surface face k,
and σ_m(θ) satisfies the equilibrium ODE:
  dσ_m/dθ = 2τ_k sin(2θ - 2φ_k)   [wrong sign? check]

Actually, let me re-derive carefully from first principles.
"""

import numpy as np
import sympy as sp
from sympy import symbols, cos, sin, Function, Eq, simplify, solve, pi, sqrt, atan2, Rational
from sympy import Piecewise

# ============================================================
# Step 1: Define the BCC yield surface
# ============================================================
# From derive_bcc_slip_systems.py, we found 5 effective systems
# but the yield polygon has 6 vertices (hexagonal).
#
# Vertices (in units of τ_CRSS, in (Δσ, σ₁₂) space):
#   V1 = (+1.83712, 0)        V4 = (-1.83712, 0)
#   V2 = (+1.48356, +0.50)    V5 = (-1.48356, -0.50)
#   V3 = (-1.48356, +0.50)    V6 = (+1.48356, -0.50)
#
# Let's use exact SymPy values.

tau = symbols('tau', positive=True)  # τ_CRSS

# From the derivation:
# The vertices come from intersections of yield lines.
# Let me compute them exactly using SymPy.

# The 5 effective systems have Schmid tensor components:
# System A: d_11 = -2/3, d_12 = -√6/12 ≈ -0.2041
# System B: d_11 = 0,    d_12 = -1/√2 ≈ -0.7071
# System C: d_11 = +2/3, d_12 = -√6/12
# System D: d_11 = +2/3, d_12 = +√6/12
# System E: d_11 = -2/3, d_12 = +√6/12

# Let me verify these exact values.
# From the output: d_norm for system A was [[-0.66667, -0.23570], ...]
# -2/3 = -0.66667 ✓
# The off-diagonal: from the raw Schmid tensor, P_12.
# System B: d_norm = [[0, -0.70711], ...] → d_12 = -1/√2 ✓

# Yield condition for each system:
# |d_11 * (σ₁₁-σ₂₂) + 2*d_12 * σ₁₂| = τ_CRSS
# But this needs normalization. The magnitude |P_eff| determines the
# effective Schmid factor.
#
# From the output, |d_eff| (in_plane_norm) for systems A,C,D,E = 1.22474
# and for system B = 0.81650
# But these are the UNNORMALIZED norms. After normalization to unit:
# |d_norm| for A,C,D,E: sqrt((-2/3)^2 + 2*(√6/12)^2) = sqrt(4/9 + 2*6/144)
#   = sqrt(4/9 + 1/12) = sqrt(16/36 + 3/36) = sqrt(19/36)
# Hmm that doesn't look clean. Let me recalculate.

# Actually the d_norm values from the code:
# System A: d_11 = -0.66667, d_12 = -0.23570
# Check: sqrt(d_11^2 + 2*d_12^2) = sqrt(0.44444 + 0.11111) = sqrt(0.55556) = 0.74536
# But incompressibility means d_11 + d_22 = 0, so d_22 = +0.66667
# The "norm" of the traceless 2x2 tensor is:
# ||d|| = sqrt(d_11^2 + d_22^2 + 2*d_12^2) = sqrt(2*d_11^2 + 2*d_12^2)
#       = sqrt(2) * sqrt(d_11^2 + d_12^2)
# = sqrt(2) * sqrt(0.44444 + 0.05556) = sqrt(2) * sqrt(0.5) = 1.0
# So the normalized Schmid tensor has unit Frobenius norm! Good.

# The yield condition in Mohr plane ((σ₁₁-σ₂₂)/2, σ₁₂):
# Let X = (σ₁₁-σ₂₂)/2 and Y = σ₁₂
# Then τ = d_11*(σ₁₁-σ₂₂) + 2*d_12*σ₁₂ = 2*d_11*X + 2*d_12*Y
# Yield: |2*d_11*X + 2*d_12*Y| ≤ τ_CRSS

# For system A: |2*(-2/3)*X + 2*(-√6/12)*Y| ≤ τ  =>  |(-4/3)X + (-√6/6)Y| ≤ τ
# Hmm, this gives different "distances" for different systems.

# Actually, the yield condition should be:
# The resolved shear stress on effective system α:
#   τ_α = P_α_ij σ_ij = P_α_11(σ₁₁-σ₂₂) + 2*P_α_12*σ₁₂
#        (using tracelessness: P_11 + P_22 = 0)
#
# For yield: |τ_α| = τ_CRSS for the active system, ≤ for all others.
#
# But the P_α values in the output are the NORMALIZED Schmid tensors
# (unit Frobenius norm). The actual yield condition uses the
# PHYSICAL Schmid tensor, which has a specific magnitude.
#
# Actually, I think the issue is: the τ_CRSS that appears in the yield
# condition is the CRSS on the individual crystallographic slip system,
# not on the effective system.
#
# For BCC {110}<111>, if we normalize slip plane normals and directions:
# |n| = sqrt(2), |s| = sqrt(3) for {110}<111>
# Normalized: |n̂| = |ŝ| = 1
# Schmid tensor: P_ij = (ŝ_i n̂_j + ŝ_j n̂_i)/2
# Resolved shear stress: τ = P_ij σ_ij
# Yield: |τ| = τ_CRSS
#
# For effective systems (pairs): the resolved shear stress is
# τ_eff = P_eff_ij σ_ij where P_eff = P_α + ratio * P_β
# The yield condition |τ_α| = τ_CRSS on the FIRST system of the pair
# translates to: |P_α_ij σ_ij| = τ_CRSS
#
# So the constraint from each INDIVIDUAL crystallographic system is:
# |P_α_ij σ_ij| ≤ τ_CRSS
#
# The yield surface is the intersection of ALL 12 conditions:
# |P_α_ij σ_ij| ≤ τ_CRSS for α = 1, ..., 12
#
# In the (X, Y) = ((σ₁₁-σ₂₂)/2, σ₁₂) Mohr plane:
# τ_α = P_α_11*(σ₁₁-σ₂₂) + 2*P_α_12*σ₁₂ = 2*P_α_11*X + 2*P_α_12*Y
# So each system gives: |2*P_α_11*X + 2*P_α_12*Y| ≤ τ_CRSS

# Let me recompute the yield surface from the INDIVIDUAL systems
# (not the effective pairs) since that gives the true yield polygon.

print("=" * 70)
print("BCC Yield Surface from Individual Slip Systems")
print("(in Mohr plane: X = (σ₁₁-σ₂₂)/2, Y = σ₁₂)")
print("=" * 70)

# Recompute Schmid tensors in primed coords for all 12 systems
e1p = np.array([0, 0, 1])
e2p = np.array([-1, 1, 0]) / np.sqrt(2)
e3p = np.array([1, 1, 0]) / np.sqrt(2)
R = np.array([e1p, e2p, e3p])

slip_data = [
    ((1, 1, 0), (-1, 1, 1)),
    ((1, 1, 0), (1, -1, 1)),
    ((1, -1, 0), (1, 1, 1)),
    ((1, -1, 0), (-1, -1, 1)),
    ((1, 0, 1), (-1, 1, 1)),
    ((1, 0, 1), (1, 1, -1)),
    ((1, 0, -1), (1, 1, 1)),
    ((1, 0, -1), (-1, 1, -1)),
    ((0, 1, 1), (1, -1, 1)),
    ((0, 1, 1), (1, 1, -1)),
    ((0, 1, -1), (1, 1, 1)),
    ((0, 1, -1), (1, -1, -1)),
]

schmid_2d = []  # (a_k, b_k) such that τ_k = a_k * X + b_k * Y
for idx, (n, s) in enumerate(slip_data):
    n_vec = np.array(n, dtype=float)
    s_vec = np.array(s, dtype=float)
    n_hat = n_vec / np.linalg.norm(n_vec)
    s_hat = s_vec / np.linalg.norm(s_vec)

    n_p = R @ n_hat
    s_p = R @ s_hat

    # In-plane Schmid tensor components
    P_11 = (s_p[0]*n_p[0] + s_p[0]*n_p[0]) / 2  # = s_1'*n_1'
    P_22 = s_p[1]*n_p[1]
    P_12 = (s_p[0]*n_p[1] + s_p[1]*n_p[0]) / 2

    # Resolved shear stress: τ = P_11*(σ₁₁-σ₂₂) + 2*P_12*σ₁₂ + P_11*σ₂₂ + P_22*σ₂₂ ...
    # Actually: τ = P_ij σ_ij = P_11*σ_11 + P_22*σ_22 + 2*P_12*σ_12
    # = P_11*(σ_11-σ_22) + (P_11+P_22)*σ_22 + 2*P_12*σ_12
    # But the yield condition in the Mohr plane uses (X, Y) where σ_m doesn't matter
    # (since yield depends only on deviatoric stress for incompressible material).
    # τ = P_11*(σ_11-σ_22) + 2*P_12*σ_12 (since P_11+P_22 ≈ 0 for in-plane incompressibility)
    # = 2*P_11*X + 2*P_12*Y

    a_k = 2 * P_11  # coefficient of X
    b_k = 2 * P_12  # coefficient of Y

    schmid_2d.append((a_k, b_k))
    print(f"  System {idx+1:2d}: τ = {a_k:+.5f}*X + {b_k:+.5f}*Y")

# The yield polygon is: max over k of |a_k*X + b_k*Y| ≤ τ_CRSS
# Vertices are at intersections where exactly 2 yield conditions are active.

print("\n" + "=" * 70)
print("Yield surface vertices (all 12 system constraints)")
print("=" * 70)

all_vertices = []
for i in range(12):
    for si in [+1, -1]:
        for j in range(i+1, 12):
            for sj in [+1, -1]:
                A = np.array([[schmid_2d[i][0], schmid_2d[i][1]],
                              [schmid_2d[j][0], schmid_2d[j][1]]])
                b = np.array([si, sj])
                det = np.linalg.det(A)
                if abs(det) < 1e-10:
                    continue
                sol = np.linalg.solve(A, b)  # (X, Y) in units of τ_CRSS

                # Check if on the yield surface (satisfies ALL constraints)
                ok = True
                for k in range(12):
                    tau_k = abs(schmid_2d[k][0]*sol[0] + schmid_2d[k][1]*sol[1])
                    if tau_k > 1.0 + 1e-8:
                        ok = False
                        break
                if ok:
                    # Check for duplicates
                    is_dup = False
                    for v in all_vertices:
                        if abs(v[0] - sol[0]) < 1e-8 and abs(v[1] - sol[1]) < 1e-8:
                            is_dup = True
                            break
                    if not is_dup:
                        all_vertices.append(sol)

# Sort by angle
angles_v = [np.arctan2(v[1], v[0]) for v in all_vertices]
order = np.argsort(angles_v)
all_vertices = [all_vertices[i] for i in order]

print(f"\nUnique vertices: {len(all_vertices)}")
for i, v in enumerate(all_vertices):
    angle = np.degrees(np.arctan2(v[1], v[0]))
    r = np.sqrt(v[0]**2 + v[1]**2)
    print(f"  V{i+1}: ({v[0]:+.5f}, {v[1]:+.5f}) τ_CRSS   angle={angle:+7.2f}°  r={r:.5f}")

# Identify which yield face is active between adjacent vertices
print("\n" + "=" * 70)
print("Active yield faces between vertices")
print("=" * 70)

n_vert = len(all_vertices)
faces = []
for i in range(n_vert):
    v1 = all_vertices[i]
    v2 = all_vertices[(i+1) % n_vert]
    mid = (np.array(v1) + np.array(v2)) / 2

    # Find which system(s) are active at this midpoint
    active = []
    for k in range(12):
        tau_k = schmid_2d[k][0]*mid[0] + schmid_2d[k][1]*mid[1]
        if abs(abs(tau_k) - 1.0) < 1e-6:
            active.append((k+1, np.sign(tau_k)))

    face_angle = np.degrees(np.arctan2(v2[1]-v1[1], v2[0]-v1[0]))
    faces.append({
        'v1': i, 'v2': (i+1) % n_vert,
        'active': active,
        'face_angle': face_angle,
    })
    active_str = ', '.join([f"{'+'if s>0 else '-'}sys{k}" for k, s in active])
    print(f"  Face V{i+1}→V{(i+1)%n_vert+1}: active = [{active_str}]  face_angle = {face_angle:.1f}°")

# ============================================================
# Step 2: Sector solution in polar coordinates
# ============================================================
print("\n" + "=" * 70)
print("Sector Solution: Stress field around cylindrical void")
print("=" * 70)

# For a cylindrical void of radius a under equibiaxial far-field stress
# σ_rr = σ_θθ = -p (compressive), σ_rθ = 0 at r → ∞
#
# At the void surface (r = a): σ_rr = 0, σ_rθ = 0 (traction-free)
#
# The stress field is decomposed into angular sectors. Within each sector,
# one yield face is active and the stress is expressed as:
#
#   σ_rr = σ_m(θ) + τ_face * cos(2θ - 2ψ)
#   σ_θθ = σ_m(θ) - τ_face * cos(2θ - 2ψ)
#   σ_rθ = -τ_face * sin(2θ - 2ψ)
#
# where τ_face and ψ characterize the active yield face, and
# σ_m(θ) is the mean stress that satisfies equilibrium:
#   dσ_m/dθ = 2τ_face * sin(2θ - 2ψ)
#
# Wait, I need to be more careful. Let me derive from scratch.

# In Mohr plane, the stress state on face k has:
#   (σ₁₁-σ₂₂)/2 = X_face(s) — parameterized along the face
#   σ₁₂ = Y_face(s) — parameterized along the face
#
# But for a SINGLE face of a polygon, the deviatoric stress is:
#   a_k * X + b_k * Y = ±τ_CRSS  (the face equation)
#
# This is a LINE in (X, Y) space. The stress state moves along this line
# as θ varies within the sector.
#
# In polar coordinates:
#   X = (σ₁₁-σ₂₂)/2 → transform to (r,θ)
#   Y = σ₁₂ → transform to (r,θ)
#
# The transformation between Cartesian and polar stress:
#   σ_rr = σ_11*cos²θ + σ_22*sin²θ + 2σ_12*sinθ*cosθ
#        = σ_m + X*cos(2θ) + Y*sin(2θ)
#   σ_θθ = σ_m - X*cos(2θ) - Y*sin(2θ)
#   σ_rθ = (σ_22-σ_11)*sinθ*cosθ + σ_12*(cos²θ-sin²θ)
#        = -X*sin(2θ) + Y*cos(2θ)
#
# where σ_m = (σ_11+σ_22)/2 = (σ_rr+σ_θθ)/2

# On face k: a_k * X + b_k * Y = sign_k * τ_CRSS
# This constrains (X, Y) to lie on a line. We parameterize:
# X = X₀ + t * (-b_k)  (direction along the face)
# Y = Y₀ + t * a_k
# where (X₀, Y₀) is a point on the line (e.g., the nearest point to origin)

# The equilibrium equations in polar coords:
#   ∂σ_rr/∂r + (σ_rr - σ_θθ)/r + (1/r)∂σ_rθ/∂θ = 0
#   ∂σ_rθ/∂r + 2σ_rθ/r + (1/r)∂σ_θθ/∂θ = 0
#
# For a rigid-plastic material, the stress field is independent of r
# within each sector (this is the key simplification for the void problem
# under power-law or rigid-plastic response).
#
# Wait, that's not right. For the void problem, σ_m varies with BOTH r and θ.
# But for rigid-ideally-plastic material, within a fully plastic sector,
# the Hencky equations constrain σ_m.
#
# Actually, let me follow Kysar's approach more carefully.

# Kysar (2005) divides the problem into:
# 1. "Constant stress sectors" where σ_m is constant (no dependence on r or θ)
#    These occur in angular regions where the stress state is at a vertex
#    of the yield polygon (double slip).
# 2. "Fan sectors" where σ_m varies with θ only
#    These occur where the stress state traverses along a face of the
#    yield polygon.
#
# The alternation of constant-stress and fan sectors creates the full
# angular structure.

# For the void problem with equibiaxial loading, due to symmetry,
# we only need to solve in 0 ≤ θ ≤ π/2 (quadrant), then extend.

# Let me implement the yield surface traversal.

# The yield polygon vertices (sorted by angle from origin in Mohr plane):
print("\nYield polygon vertices (sorted):")
for i, v in enumerate(all_vertices):
    angle = np.degrees(np.arctan2(v[1], v[0]))
    print(f"  V{i+1}: X = {v[0]:+.5f}, Y = {v[1]:+.5f}  (angle = {angle:+.2f}°)")

# ============================================================
# Step 3: Boundary conditions → determine the stress path
# ============================================================
print("\n" + "=" * 70)
print("Boundary Conditions")
print("=" * 70)

# At the void surface (r = a):
#   σ_rr = 0, σ_rθ = 0
#   This means: σ_m + X*cos(2θ) + Y*sin(2θ) = 0
#               -X*sin(2θ) + Y*cos(2θ) = 0
#
# From the second equation: Y/X = tan(2θ), i.e., Y = X*tan(2θ)
# or equivalently: the Mohr stress point (X, Y) lies on the ray
# from origin at angle 2θ in the Mohr plane.
#
# From the first: σ_m = -X*cos(2θ) - Y*sin(2θ) = -X/cos(2θ)
#                 (using Y = X*tan(2θ))

# At far field (r → ∞):
#   σ_rr = σ_θθ = -p (equibiaxial), σ_rθ = 0
#   This means: X_∞ = 0, Y_∞ = 0, σ_m = -p
#   The stress state is at the ORIGIN of the Mohr plane.

# The stress path:
# As we move from r = a (void surface) to r → ∞ (far field),
# the deviatoric stress (X, Y) moves from a point ON the yield surface
# to the ORIGIN.
#
# As θ changes, the point on the yield surface where the void surface
# condition is satisfied changes.
#
# At θ = 0: void BC requires Y = 0 (from σ_rθ = 0 with sin(0) = 0)
# and σ_rr = σ_m + X = 0 → σ_m = -X
# So X must be on the yield surface with Y = 0.
# The yield surface intersects Y = 0 at X = ±X_max

# Looking at the vertices: on Y = 0 axis, the yield polygon extends to
# X = ±1.83712 (these are vertices). But the face between these vertices
# may be at smaller |X|.

# Let me find where the yield surface intersects Y = 0:
# The active face at Y = 0 must have a_k * X_max = ±τ_CRSS with b_k * 0 = 0
# Since different faces have different a_k values, the tightest constraint
# at Y = 0 is from the face with the LARGEST |a_k|.

# Check which faces pass through Y = 0:
print("\nYield surface at Y = 0:")
x_max_positive = float('inf')
x_max_negative = float('-inf')
for k in range(12):
    a_k, b_k = schmid_2d[k]
    if abs(a_k) > 1e-10:
        # a_k * X = ±1 at Y = 0
        x_pos = 1.0 / a_k
        x_neg = -1.0 / a_k
        if x_pos > 0:
            x_max_positive = min(x_max_positive, x_pos)
        if x_neg > 0:
            x_max_positive = min(x_max_positive, x_neg)
    # Also check: is (X, 0) inside yield surface?

# More directly: find the maximum X on the yield surface at Y = 0
X_test = np.linspace(-3, 3, 10000)
for X in X_test:
    ok = True
    for k in range(12):
        if abs(schmid_2d[k][0] * X) > 1.0 + 1e-8:
            ok = False
            break
    if ok and X > 0:
        x_yield_max = X

for X in reversed(X_test):
    ok = True
    for k in range(12):
        if abs(schmid_2d[k][0] * X) > 1.0 + 1e-8:
            ok = False
            break
    if ok and X < 0:
        x_yield_min = X

print(f"  Maximum X at Y=0: {x_yield_max:.5f} τ_CRSS")
print(f"  Minimum X at Y=0: {x_yield_min:.5f} τ_CRSS")

# ============================================================
# Step 4: Determine sector structure
# ============================================================
print("\n" + "=" * 70)
print("Sector Structure Determination")
print("=" * 70)

# For equibiaxial loading, the problem has 4-fold symmetry:
# σ(θ) = σ(π-θ) = σ(-θ) = σ(π+θ)
#
# Wait, for the BCC crystal, the symmetry depends on the crystal
# orientation. With [110] void axis and the in-plane directions
# e1' = [001] and e2' = [-110]/√2, the symmetry group is determined
# by the crystal symmetry of the (110) plane.
#
# The (110) plane in BCC has:
# - Mirror symmetry about the [001] axis (e1'): θ → -θ
# - Mirror symmetry about the [-110] axis (e2'): θ → π-θ
# - Combined: 4 equivalent sectors in [0, π/2]
#
# Actually, this depends on whether the crystal symmetry in the (110)
# plane is 2mm (which it is for BCC). So we have:
# σ(θ) = σ(-θ)    (mirror about e1')
# σ(θ) = σ(π-θ)   (mirror about e2')
#
# This means we only need to solve for 0 ≤ θ ≤ π/2.

# At θ = 0 (void surface, along e1' = [001]):
# σ_rθ = 0 (by symmetry), σ_rr = 0 (traction free)
# → X = (σ₁₁-σ₂₂)/2 at this point
# From σ_rr = σ_m + X = 0 → σ_m = -X (at void surface)
# The stress must be on the yield surface.

# At θ = π/2 (void surface, along e2' = [-110]/√2):
# Same conditions by symmetry.

# The stress traversal around the void surface must go from one
# yield vertex to another as θ increases from 0 to π/2.

# For the void problem in isotropic plasticity:
# At θ = 0: σ_rθ = 0 → (X, Y) is on the X-axis of Mohr plane
# The stress is at the maximum |X| point of the yield surface.
# As θ increases, the stress moves along the yield surface.
# At θ = π/4: the σ_rθ = 0 condition requires Y = X*tan(π/2) = ∞,
# which doesn't make sense → the void surface stress reaches a
# different vertex or face.

# Actually, I think I need to be more systematic. Let me trace
# the stress state around the void by using the equilibrium + yield
# conditions.

# For a rigid-ideally-plastic material, everywhere in the plastic zone
# the stress lies on or inside the yield surface. At the void surface,
# σ_rr = σ_rθ = 0 constrains the stress.

# Let me compute the void surface stress state as a function of θ.
# At angle θ on the void surface:
# σ_rr = σ_m + X cos(2θ) + Y sin(2θ) = 0   ...(i)
# σ_rθ = -X sin(2θ) + Y cos(2θ) = 0          ...(ii)
# From (ii): Y = X tan(2θ)  (for 2θ ≠ π/2, 3π/2)
# Sub into (i): σ_m = -X cos(2θ) - X tan(2θ) sin(2θ) = -X/cos(2θ)
# So the void surface stress in Mohr plane is on the ray at angle 2θ.

# Now, this stress must also be ON the yield surface:
# |a_k * X + b_k * Y| = 1 for the active face k

# For each θ, find the (X, Y) on the yield surface intersected by the
# ray at angle 2θ from origin:
# X = R cos(2θ), Y = R sin(2θ) for some R > 0
# Active face k: a_k * R cos(2θ) + b_k * R sin(2θ) = ±1
# R = ±1 / (a_k cos(2θ) + b_k sin(2θ))

# The void surface stress traces out the yield polygon as θ goes from 0 to π.
# At each θ, we pick the INNERMOST yield face (smallest positive R).

theta_vals = np.linspace(0, np.pi, 361)
void_X = np.zeros_like(theta_vals)
void_Y = np.zeros_like(theta_vals)
void_sigma_m = np.zeros_like(theta_vals)
void_active_face = np.zeros(len(theta_vals), dtype=int)

print("\nTracing yield surface along void boundary:")
for ti, theta in enumerate(theta_vals):
    c2 = np.cos(2*theta)
    s2 = np.sin(2*theta)

    # Find R for each face
    R_min = float('inf')
    best_k = -1
    for k in range(12):
        a_k, b_k = schmid_2d[k]
        denom = a_k * c2 + b_k * s2
        if abs(denom) < 1e-10:
            continue
        R_pos = 1.0 / denom
        R_neg = -1.0 / denom
        for R in [R_pos, R_neg]:
            if R > 1e-10 and R < R_min:
                # Verify this R satisfies ALL yield conditions
                X_cand = R * c2
                Y_cand = R * s2
                feasible = True
                for m in range(12):
                    tau_m = abs(schmid_2d[m][0]*X_cand + schmid_2d[m][1]*Y_cand)
                    if tau_m > 1.0 + 1e-6:
                        feasible = False
                        break
                if feasible:
                    R_min = R
                    best_k = k

    if R_min < float('inf'):
        void_X[ti] = R_min * c2
        void_Y[ti] = R_min * s2
        void_sigma_m[ti] = -R_min / max(abs(c2), 1e-10) if abs(c2) > 1e-3 else -R_min * c2 / (c2**2 + s2**2)
        void_active_face[ti] = best_k + 1

# Print key angles
print(f"\n  θ =   0°: X = {void_X[0]:+.5f}, Y = {void_Y[0]:+.5f}, active = sys {void_active_face[0]}")
print(f"  θ =  30°: X = {void_X[30]:+.5f}, Y = {void_Y[30]:+.5f}, active = sys {void_active_face[30]}")
print(f"  θ =  45°: X = {void_X[45]:+.5f}, Y = {void_Y[45]:+.5f}, active = sys {void_active_face[45]}")
print(f"  θ =  60°: X = {void_X[60]:+.5f}, Y = {void_Y[60]:+.5f}, active = sys {void_active_face[60]}")
print(f"  θ =  90°: X = {void_X[90]:+.5f}, Y = {void_Y[90]:+.5f}, active = sys {void_active_face[90]}")
print(f"  θ = 120°: X = {void_X[120]:+.5f}, Y = {void_Y[120]:+.5f}, active = sys {void_active_face[120]}")
print(f"  θ = 135°: X = {void_X[135]:+.5f}, Y = {void_Y[135]:+.5f}, active = sys {void_active_face[135]}")
print(f"  θ = 150°: X = {void_X[150]:+.5f}, Y = {void_Y[150]:+.5f}, active = sys {void_active_face[150]}")
print(f"  θ = 180°: X = {void_X[180]:+.5f}, Y = {void_Y[180]:+.5f}, active = sys {void_active_face[180]}")

# Identify sector boundaries (where active face changes)
print("\nSector boundaries (active face transitions):")
boundaries = []
for ti in range(1, len(theta_vals)):
    if void_active_face[ti] != void_active_face[ti-1]:
        theta_deg = np.degrees(theta_vals[ti])
        boundaries.append(theta_deg)
        print(f"  θ ≈ {theta_deg:6.1f}°: sys {void_active_face[ti-1]} → sys {void_active_face[ti]}")

print(f"\nTotal sectors in [0°, 180°]: {len(boundaries) + 1}")

# ============================================================
# Step 5: Plot the sector structure
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Yield surface with void surface stress path
    ax1 = axes[0]
    # Plot yield polygon
    vx = [v[0] for v in all_vertices] + [all_vertices[0][0]]
    vy = [v[1] for v in all_vertices] + [all_vertices[0][1]]
    ax1.plot(vx, vy, 'b-', linewidth=2)
    ax1.fill(vx, vy, alpha=0.1, color='blue')
    # Plot void surface stress path
    ax1.plot(void_X, void_Y, 'r-', linewidth=1.5, label='Void surface stress')
    # Mark key angles
    for theta_deg in [0, 45, 90, 135, 180]:
        ti = theta_deg
        ax1.plot(void_X[ti], void_Y[ti], 'ko', markersize=5)
        ax1.annotate(f'{theta_deg}°', (void_X[ti], void_Y[ti]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax1.set_xlabel(r'$X = (\sigma_{11}-\sigma_{22})/2$', fontsize=11)
    ax1.set_ylabel(r'$Y = \sigma_{12}$', fontsize=11)
    ax1.set_title('(a) Void surface stress on yield surface', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # (b) Active system vs θ
    ax2 = axes[1]
    theta_deg_arr = np.degrees(theta_vals)
    ax2.plot(theta_deg_arr, void_active_face, 'b-', linewidth=1.5)
    ax2.set_xlabel(r'$\theta$ (degrees)', fontsize=11)
    ax2.set_ylabel('Active slip system', fontsize=11)
    ax2.set_title('(b) Active system around void', fontsize=12)
    ax2.grid(True, alpha=0.3)
    for b in boundaries:
        ax2.axvline(x=b, color='r', linestyle='--', alpha=0.5)

    # (c) Stress components at void surface
    ax3 = axes[2]
    sigma_m_void = np.zeros_like(theta_vals)
    sigma_rr_void = np.zeros_like(theta_vals)
    sigma_tt_void = np.zeros_like(theta_vals)
    sigma_rt_void = np.zeros_like(theta_vals)

    for ti, theta in enumerate(theta_vals):
        c2 = np.cos(2*theta)
        s2 = np.sin(2*theta)
        X, Y = void_X[ti], void_Y[ti]
        # σ_m from void BC: σ_rr = σ_m + X*c2 + Y*s2 = 0
        sm = -(X*c2 + Y*s2)
        sigma_m_void[ti] = sm
        sigma_rr_void[ti] = sm + X*c2 + Y*s2  # should be 0
        sigma_tt_void[ti] = sm - X*c2 - Y*s2  # = -2*(X*c2 + Y*s2)
        sigma_rt_void[ti] = -X*s2 + Y*c2       # should be 0

    ax3.plot(theta_deg_arr, sigma_m_void, 'b-', linewidth=2, label=r'$\sigma_m$')
    ax3.plot(theta_deg_arr, sigma_tt_void, 'r-', linewidth=2, label=r'$\sigma_{\theta\theta}$')
    ax3.plot(theta_deg_arr, sigma_rr_void, 'g--', linewidth=1, label=r'$\sigma_{rr}$ (check=0)')
    ax3.plot(theta_deg_arr, sigma_rt_void, 'm--', linewidth=1, label=r'$\sigma_{r\theta}$ (check=0)')
    ax3.set_xlabel(r'$\theta$ (degrees)', fontsize=11)
    ax3.set_ylabel(r'Stress / $\tau_{CRSS}$', fontsize=11)
    ax3.set_title('(c) Stress at void surface', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    for b in boundaries:
        ax3.axvline(x=b, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig_path = 'experiments/bcc_void_analytical/sector_structure.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")

except ImportError:
    print("\nMatplotlib not available; skipping plot.")

# ============================================================
# Step 6: Activation pressure for void growth
# ============================================================
print("\n" + "=" * 70)
print("Activation Pressure for Void Growth")
print("=" * 70)

# The far-field equibiaxial stress needed to maintain plastic flow
# around the entire void is determined by the mean stress σ_m.
# The activation pressure p* is the minimum far-field pressure
# such that the entire void surface is in a plastic state.
#
# From the stress field: at far field, σ_m → -p (compressive)
# At the void surface: σ_m varies with θ.
# The maximum |σ_m| at the void surface gives the activation pressure.

sm_max = np.max(np.abs(sigma_m_void))
print(f"\nMaximum |σ_m| at void surface: {sm_max:.5f} τ_CRSS")
print(f"This is proportional to the activation pressure for void growth.")

# For comparison, isotropic von Mises:
# p_iso = τ_y * (1 + ln(2)) ≈ 1.693 * τ_y (approximate for thick-walled tube)
# Actually for a void: p_iso = σ_y * (2/3) * ln(b/a) → for infinite medium, ∞
# The ACTIVATION of first yield at the void surface under equibiaxial stress:
# For isotropic: σ_rr = 0 at void, σ_θθ = 2p at void →
#   max shear = p → p_yield = τ_y (= σ_y/2 for Tresca, = σ_y/√3 for Mises)

# For the anisotropic case: the activation pressure depends on the yield
# surface geometry. The stress at the void must lie on the yield surface
# with σ_rr = σ_rθ = 0. The minimum far-field pressure to achieve this
# is related to the "reach" of the yield surface in the direction
# perpendicular to the void surface normal.

print(f"\nFor reference:")
print(f"  Isotropic Tresca: p*/τ_y = 1.0")
print(f"  Isotropic Mises:  p*/τ_y = 2/√3 ≈ {2/np.sqrt(3):.5f}")
print(f"  BCC {'{'}110{'}'}<111>:   p*/τ_CRSS ≈ {sm_max:.5f}")

print("\n" + "=" * 70)
print("SECTOR SOLUTION COMPLETE")
print("=" * 70)
