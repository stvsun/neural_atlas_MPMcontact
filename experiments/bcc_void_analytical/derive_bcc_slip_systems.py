"""
Derive effective in-plane slip systems for BCC {110}<111> crystal
with cylindrical void axis parallel to [110].

Following Kysar et al. (2005) methodology for FCC, adapted to BCC.

BCC has 12 slip systems in the {110}<111> family.
For plane strain in the (110) plane (void axis || [110]):
  We need pairs of 3D slip systems that combine to give
  d_33 = d_13 = d_23 = 0 (plane strain conditions).

Reference: Rice (1987), Kysar et al. (2005), Hartley & Kysar (2020)
"""

import sympy as sp
from sympy import sqrt, Rational, Matrix, pprint, simplify, symbols, cos, sin, atan2, pi
import numpy as np

# ============================================================
# Step 1: Define all 12 BCC {110}<111> slip systems
# ============================================================

slip_data = [
    # Plane (110)
    ((1, 1, 0), (-1, 1, 1)),    # 1
    ((1, 1, 0), (1, -1, 1)),    # 2
    # Plane (1,-1,0)
    ((1, -1, 0), (1, 1, 1)),    # 3
    ((1, -1, 0), (-1, -1, 1)),  # 4
    # Plane (101)
    ((1, 0, 1), (-1, 1, 1)),    # 5
    ((1, 0, 1), (1, 1, -1)),    # 6
    # Plane (10,-1)
    ((1, 0, -1), (1, 1, 1)),    # 7
    ((1, 0, -1), (-1, 1, -1)),  # 8
    # Plane (011)
    ((0, 1, 1), (1, -1, 1)),    # 9
    ((0, 1, 1), (1, 1, -1)),    # 10
    # Plane (01,-1)
    ((0, 1, -1), (1, 1, 1)),    # 11
    ((0, 1, -1), (1, -1, -1)),  # 12
]

print("=" * 70)
print("BCC {110}<111> Slip Systems (12 total)")
print("=" * 70)

for i, (n, s) in enumerate(slip_data):
    n_vec = Matrix(n)
    s_vec = Matrix(s)
    dot = n_vec.dot(s_vec)
    assert dot == 0, f"System {i+1}: n·s = {dot} != 0"
    print(f"System {i+1:2d}: n = {str(n):12s}  s = {str(s):14s}  n·s = {dot}")

# ============================================================
# Step 2: Coordinate transformation
# ============================================================
# Void axis e3' = [110]/sqrt(2)
# In-plane: e1' = [001], e2' = [-110]/sqrt(2)

e1p = Matrix([0, 0, 1])
e2p = Matrix([-1, 1, 0]) / sqrt(2)
e3p = Matrix([1, 1, 0]) / sqrt(2)

R = Matrix([
    [e1p[0], e1p[1], e1p[2]],
    [e2p[0], e2p[1], e2p[2]],
    [e3p[0], e3p[1], e3p[2]],
])

print(f"\nCoordinate system: e1'=[001], e2'=[-110]/√2, e3'=[110]/√2")

# ============================================================
# Step 3: Compute full strain rate tensor for each system
# ============================================================
print("\n" + "=" * 70)
print("Strain rate tensor d_ij = (s_i n_j + s_j n_i)/2 for each system")
print("(in primed coordinates, normalized by |n||s|)")
print("=" * 70)

systems = []
for idx, (n, s) in enumerate(slip_data):
    n_vec = Matrix(n) / Matrix(n).norm()
    s_vec = Matrix(s) / Matrix(s).norm()

    # Transform to primed coords
    n_p = simplify(R * n_vec)
    s_p = simplify(R * s_vec)

    # Strain rate tensor (3x3 symmetric)
    d = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            d[i, j] = simplify((s_p[i] * n_p[j] + s_p[j] * n_p[i]) / 2)

    systems.append({
        'idx': idx + 1,
        'n_cryst': n,
        's_cryst': s,
        'n_p': n_p,
        's_p': s_p,
        'd': d,
    })

    print(f"\nSystem {idx+1}: n={n}, s={s}")
    print(f"  n' = [{float(n_p[0]):8.5f}, {float(n_p[1]):8.5f}, {float(n_p[2]):8.5f}]")
    print(f"  s' = [{float(s_p[0]):8.5f}, {float(s_p[1]):8.5f}, {float(s_p[2]):8.5f}]")
    print(f"  d_11={float(d[0,0]):8.5f}  d_22={float(d[1,1]):8.5f}  d_33={float(d[2,2]):8.5f}")
    print(f"  d_12={float(d[0,1]):8.5f}  d_13={float(d[0,2]):8.5f}  d_23={float(d[1,2]):8.5f}")

# ============================================================
# Step 4: Find ALL pairs achieving plane strain
# ============================================================
print("\n" + "=" * 70)
print("Comprehensive search: pairs (α, β) with d_33=d_13=d_23=0")
print("(allowing unequal slip rates γ̇_α, γ̇_β)")
print("=" * 70)

valid_pairs = []
for i in range(12):
    for j in range(i+1, 12):
        si = systems[i]
        sj = systems[j]
        di = si['d']
        dj = sj['d']

        # For plane strain: γ̇_α * d^α_{33} + γ̇_β * d^β_{33} = 0
        #                    γ̇_α * d^α_{13} + γ̇_β * d^β_{13} = 0
        #                    γ̇_α * d^α_{23} + γ̇_β * d^β_{23} = 0
        #
        # This is a system A * [γ̇_α; γ̇_β] = 0 where A is 3×2.
        # Non-trivial solution exists iff rank(A) ≤ 1.

        A = sp.Matrix([
            [di[2, 2], dj[2, 2]],
            [di[0, 2], dj[0, 2]],
            [di[1, 2], dj[1, 2]],
        ])
        A_float = np.array(A.tolist(), dtype=float)

        # Check rank
        rank = np.linalg.matrix_rank(A_float, tol=1e-8)

        if rank <= 1:
            # Find the ratio γ̇_β / γ̇_α
            if abs(A_float[0, 0]) > 1e-10:
                ratio = -A_float[0, 0] / A_float[0, 1] if abs(A_float[0, 1]) > 1e-10 else None
            elif abs(A_float[1, 0]) > 1e-10:
                ratio = -A_float[1, 0] / A_float[1, 1] if abs(A_float[1, 1]) > 1e-10 else None
            elif abs(A_float[2, 0]) > 1e-10:
                ratio = -A_float[2, 0] / A_float[2, 1] if abs(A_float[2, 1]) > 1e-10 else None
            else:
                ratio = 1.0  # both columns zero → any ratio works

            if ratio is None:
                continue  # degenerate

            # Compute effective in-plane strain tensor
            d_eff = di + ratio * dj
            d_eff_float = np.array(d_eff.tolist(), dtype=float)

            # Check it actually has in-plane content
            in_plane_norm = np.sqrt(d_eff_float[0, 0]**2 + d_eff_float[1, 1]**2 + 2*d_eff_float[0, 1]**2)
            if in_plane_norm < 1e-10:
                continue  # no in-plane strain (pure anti-plane pair)

            # Verify plane strain conditions
            ps_err = abs(d_eff_float[2, 2]) + abs(d_eff_float[0, 2]) + abs(d_eff_float[1, 2])
            assert ps_err < 1e-8, f"Plane strain check failed: err={ps_err}"

            valid_pairs.append({
                'pair': (i+1, j+1),
                'ratio': ratio,
                'd_eff': d_eff_float[:2, :2],  # 2×2 in-plane
                'in_plane_norm': in_plane_norm,
                'n_i': si['n_cryst'],
                's_i': si['s_cryst'],
                'n_j': sj['n_cryst'],
                's_j': sj['s_cryst'],
            })

            print(f"\n  Pair ({i+1:2d}, {j+1:2d}): γ̇_β/γ̇_α = {ratio:+.4f}")
            print(f"    Sys {i+1}: n={si['n_cryst']}, s={si['s_cryst']}")
            print(f"    Sys {j+1}: n={sj['n_cryst']}, s={sj['s_cryst']}")
            print(f"    d_eff (in-plane): [[{d_eff_float[0,0]:+.5f}, {d_eff_float[0,1]:+.5f}],")
            print(f"                       [{d_eff_float[1,0]:+.5f}, {d_eff_float[1,1]:+.5f}]]")
            print(f"    |d_eff| = {in_plane_norm:.5f}")

print(f"\nTotal valid pairs found: {len(valid_pairs)}")

# ============================================================
# Step 5: Identify DISTINCT effective in-plane systems
# ============================================================
print("\n" + "=" * 70)
print("Distinct effective in-plane slip systems")
print("=" * 70)

# Two effective systems are the "same" if their d_eff tensors
# are proportional (same slip direction in 2D)

# Characterize each by the angle of the slip line in the (e1', e2') plane
# d_eff_11 = -sin(2φ)/2 * |d|,  d_eff_12 = cos(2φ)/2 * |d|  (for unit CRSS)
# Actually for a 2D Schmid tensor: d_11 = s1*n1, d_12 = (s1*n2+s2*n1)/2
# The "angle" is determined by atan2(-d_11, d_12) / 2

distinct = []
for vp in valid_pairs:
    d = vp['d_eff']
    # Normalize
    norm = vp['in_plane_norm']
    d_norm = d / norm

    # Angle: atan2(d_11, d_12) gives 2*phi in Mohr plane
    # For the resolved shear stress: τ = d_11*(σ_11-σ_22) + 2*d_12*σ_12
    # In Mohr plane (Δσ, σ_12): normal direction is (d_11, 2*d_12)/|.|
    angle_2phi = np.arctan2(2*d_norm[0, 1], d_norm[0, 0])  # angle in Mohr plane
    phi = angle_2phi / 2  # slip line angle in physical plane

    vp['phi_deg'] = np.degrees(phi)
    vp['angle_2phi_deg'] = np.degrees(angle_2phi)

    # Check if this is a new distinct system
    is_new = True
    for ds in distinct:
        angle_diff = abs(vp['phi_deg'] - ds['phi_deg'])
        if angle_diff < 1.0 or abs(angle_diff - 180) < 1.0:
            is_new = False
            ds['pairs'].append(vp['pair'])
            break
    if is_new:
        distinct.append({
            'phi_deg': vp['phi_deg'],
            'angle_2phi_deg': vp['angle_2phi_deg'],
            'd_norm': d_norm,
            'pairs': [vp['pair']],
            'representative': vp,
        })

print(f"\nNumber of distinct effective in-plane systems: {len(distinct)}")
for k, ds in enumerate(sorted(distinct, key=lambda x: x['phi_deg'])):
    print(f"\n  System {chr(65+k)} (φ = {ds['phi_deg']:+7.2f}°):")
    print(f"    Mohr plane angle 2φ = {ds['angle_2phi_deg']:+7.2f}°")
    print(f"    d_norm = [[{ds['d_norm'][0,0]:+.5f}, {ds['d_norm'][0,1]:+.5f}],")
    print(f"              [{ds['d_norm'][1,0]:+.5f}, {ds['d_norm'][1,1]:+.5f}]]")
    print(f"    Contributing pairs: {ds['pairs']}")
    vp = ds['representative']
    print(f"    Example: sys {vp['pair'][0]} (n={vp['n_i']}, s={vp['s_i']}) + "
          f"sys {vp['pair'][1]} (n={vp['n_j']}, s={vp['s_j']})")

# ============================================================
# Step 6: Construct the yield surface
# ============================================================
print("\n" + "=" * 70)
print("2D Yield Surface: Hexagonal yield locus in Mohr stress plane")
print("(Δσ, σ₁₂) = ((σ₁₁-σ₂₂)/2, σ₁₂)")
print("=" * 70)

# Each effective system defines a pair of parallel lines:
#   |d_11*(σ₁₁-σ₂₂) + 2*d_12*σ₁₂| = τ_CRSS
# These lines are at distance τ_CRSS/|d_eff| from the origin,
# with normal direction (d_11, 2*d_12)/|(d_11, 2*d_12)|

# The intersection of all |τ| ≤ τ_CRSS conditions forms the yield polygon.

print("\nYield conditions:")
for k, ds in enumerate(sorted(distinct, key=lambda x: x['phi_deg'])):
    d = ds['d_norm']
    coeff_dsig = d[0, 0]   # coefficient of (σ₁₁-σ₂₂)
    coeff_s12 = 2 * d[0, 1]  # coefficient of σ₁₂
    norm = np.sqrt(coeff_dsig**2 + coeff_s12**2)
    print(f"  System {chr(65+k)}: {coeff_dsig/norm:+.5f}·Δσ + {coeff_s12/norm:+.5f}·σ₁₂ = ±τ_CRSS/({norm:.5f}·|d|)")
    print(f"    Line normal angle in Mohr plane: {np.degrees(np.arctan2(coeff_s12, coeff_dsig)):.2f}°")

# ============================================================
# Step 7: Compare with FCC
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON: BCC vs FCC effective in-plane systems")
print("=" * 70)

# FCC {111}<110> with [110] void axis:
# The 12 FCC slip systems combine pairwise into 3 effective systems.
# Following Rice (1987) and Kysar (2005):
#   System A: φ_FCC = 0°    (slip line along [001] = e1')
#   System B: φ_FCC = +54.74° (= arctan(√2) ≈ 54.74°)
#   System C: φ_FCC = -54.74°
# The FCC hexagonal yield surface has vertices separated by 60° in Mohr plane.

print("\nFCC (Kysar 2005, Rice 1987):")
print("  3 effective systems at φ = 0°, ±54.74°")
print("  Hexagonal yield surface: vertices at 0°, 60°, 120°, 180°, 240°, 300° in Mohr plane")

bcc_angles = sorted([ds['phi_deg'] for ds in distinct])
print(f"\nBCC (this work):")
print(f"  {len(distinct)} effective systems at φ = {', '.join(f'{a:+.2f}°' for a in bcc_angles)}")

if len(distinct) == 3:
    seps = [bcc_angles[1] - bcc_angles[0], bcc_angles[2] - bcc_angles[1]]
    print(f"  Angular separations: {seps[0]:.2f}°, {seps[1]:.2f}°")

    # The key physical result:
    print("\n" + "=" * 70)
    print("KEY RESULT: BCC vs FCC hexagonal yield surface rotation")
    print("=" * 70)
    print(f"\nFCC hexagon is oriented at 0° (one face normal along e1' = [001])")
    print(f"BCC hexagon is rotated by {bcc_angles[0]:.2f}° relative to FCC")
    print(f"\nThis rotation encodes the fundamental difference in how BCC and FCC")
    print(f"crystals resist void growth: the orientations of active slip systems")
    print(f"around the void differ, leading to different sector boundaries and")
    print(f"different activation pressures for plastic flow.")

# ============================================================
# Step 8: Compute yield surface vertices
# ============================================================
print("\n" + "=" * 70)
print("Yield surface vertices (double-slip states)")
print("=" * 70)

# Each vertex is the intersection of two adjacent yield lines.
# For a hexagonal yield surface with N=3 systems, there are 6 vertices
# (2 per system pair).

sorted_distinct = sorted(distinct, key=lambda x: x['phi_deg'])
vertices = []

for k in range(len(sorted_distinct)):
    for sign_k in [+1, -1]:
        for l in range(k+1, len(sorted_distinct)):
            for sign_l in [+1, -1]:
                dk = sorted_distinct[k]['d_norm']
                dl = sorted_distinct[l]['d_norm']

                # Line k: dk[0,0]*Δσ + 2*dk[0,1]*σ₁₂ = sign_k * τ_CRSS / norm_k
                # Line l: dl[0,0]*Δσ + 2*dl[0,1]*σ₁₂ = sign_l * τ_CRSS / norm_l
                # Solve 2×2 system for (Δσ, σ₁₂)

                norm_k = np.sqrt(dk[0,0]**2 + (2*dk[0,1])**2)
                norm_l = np.sqrt(dl[0,0]**2 + (2*dl[0,1])**2)

                A = np.array([
                    [dk[0,0], 2*dk[0,1]],
                    [dl[0,0], 2*dl[0,1]],
                ])
                b = np.array([sign_k / norm_k, sign_l / norm_l])

                det = np.linalg.det(A)
                if abs(det) < 1e-10:
                    continue

                sol = np.linalg.solve(A, b)  # (Δσ, σ₁₂) in units of τ_CRSS
                vertices.append({
                    'dsig': sol[0],
                    's12': sol[1],
                    'systems': (chr(65+k), chr(65+l)),
                    'signs': (sign_k, sign_l),
                })

# Filter to vertices on the yield surface (must satisfy all other yield conditions)
on_surface = []
for v in vertices:
    inside = True
    for ds in sorted_distinct:
        d = ds['d_norm']
        norm = np.sqrt(d[0,0]**2 + (2*d[0,1])**2)
        tau = abs(d[0,0]*v['dsig'] + 2*d[0,1]*v['s12'])
        if tau > 1.0/norm + 1e-8:
            inside = False
            break
    if inside:
        on_surface.append(v)

print(f"\nVertices on yield surface: {len(on_surface)} (expect 6 for hexagon)")
for v in on_surface:
    angle = np.degrees(np.arctan2(v['s12'], v['dsig']))
    print(f"  ({v['dsig']:+.5f}, {v['s12']:+.5f}) τ_CRSS  "
          f"angle={angle:+7.2f}°  "
          f"systems={v['systems']}, signs={v['signs']}")

# ============================================================
# Step 9: Plot the yield surface
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- BCC yield surface ---
    if on_surface:
        # Sort vertices by angle for polygon
        vs = [(v['dsig'], v['s12']) for v in on_surface]
        angles = [np.arctan2(v[1], v[0]) for v in vs]
        order = np.argsort(angles)
        xs = [vs[i][0] for i in order] + [vs[order[0]][0]]
        ys = [vs[i][1] for i in order] + [vs[order[0]][1]]

        ax1.plot(xs, ys, 'b-o', linewidth=2, markersize=8, label='BCC {110}<111>')
        ax1.fill(xs, ys, alpha=0.15, color='blue')

    # Draw yield lines
    for k, ds in enumerate(sorted_distinct):
        d = ds['d_norm']
        norm_d = np.sqrt(d[0,0]**2 + (2*d[0,1])**2)
        # Normal to the line: (d[0,0], 2*d[0,1]) / norm
        nx, ny = d[0,0]/norm_d, 2*d[0,1]/norm_d
        # Line direction: (-ny, nx)
        for sign in [+1, -1]:
            # Point on line: sign/norm_d * (nx, ny)
            cx, cy = sign/norm_d * nx, sign/norm_d * ny
            t = np.linspace(-3, 3, 100)
            ax1.plot(cx - ny*t, cy + nx*t, '--', color='gray', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel(r'$(\sigma_{11}-\sigma_{22})/2\tau_{CRSS}$', fontsize=12)
    ax1.set_ylabel(r'$\sigma_{12}/\tau_{CRSS}$', fontsize=12)
    ax1.set_title('BCC {110}<111> Yield Surface\n(void axis || [110])', fontsize=13)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.legend()

    # --- FCC yield surface for comparison ---
    # FCC vertices from Kysar (2005): hexagonal yield surface
    # The 3 FCC effective systems at φ = 0°, ±54.74° (arctan(√2))
    # Yield surface vertices in (Δσ/2, σ₁₂) space:
    fcc_phi = [0, np.arctan(np.sqrt(2)), -np.arctan(np.sqrt(2))]
    fcc_lines = []
    for phi in fcc_phi:
        # Schmid tensor: d_11 = -sin(2φ)/2, d_12 = cos(2φ)/2
        d11 = -np.sin(2*phi)/2
        d12 = np.cos(2*phi)/2
        fcc_lines.append((d11, 2*d12))

    fcc_vertices = []
    for k in range(3):
        for sign_k in [+1, -1]:
            for l in range(k+1, 3):
                for sign_l in [+1, -1]:
                    A = np.array([list(fcc_lines[k]), list(fcc_lines[l])])
                    norm_k = np.linalg.norm(fcc_lines[k])
                    norm_l = np.linalg.norm(fcc_lines[l])
                    b = np.array([sign_k/norm_k, sign_l/norm_l])
                    det = np.linalg.det(A)
                    if abs(det) < 1e-10:
                        continue
                    sol = np.linalg.solve(A, b)
                    # Check if on surface
                    ok = True
                    for m in range(3):
                        tau = abs(fcc_lines[m][0]*sol[0] + fcc_lines[m][1]*sol[1])
                        if tau > 1.0/np.linalg.norm(fcc_lines[m]) + 1e-8:
                            ok = False
                            break
                    if ok:
                        fcc_vertices.append(sol)

    if fcc_vertices:
        fcc_angles = [np.arctan2(v[1], v[0]) for v in fcc_vertices]
        fcc_order = np.argsort(fcc_angles)
        fcc_xs = [fcc_vertices[i][0] for i in fcc_order] + [fcc_vertices[fcc_order[0]][0]]
        fcc_ys = [fcc_vertices[i][1] for i in fcc_order] + [fcc_vertices[fcc_order[0]][1]]

    # Plot both on ax2
    if on_surface:
        ax2.plot(xs, ys, 'b-o', linewidth=2, markersize=8, label='BCC {110}<111>')
        ax2.fill(xs, ys, alpha=0.1, color='blue')
    if fcc_vertices:
        ax2.plot(fcc_xs, fcc_ys, 'r-s', linewidth=2, markersize=8, label='FCC {111}<110>')
        ax2.fill(fcc_xs, fcc_ys, alpha=0.1, color='red')

    ax2.set_xlabel(r'$(\sigma_{11}-\sigma_{22})/2\tau_{CRSS}$', fontsize=12)
    ax2.set_ylabel(r'$\sigma_{12}/\tau_{CRSS}$', fontsize=12)
    ax2.set_title('BCC vs FCC Yield Surface Comparison\n(void axis || [110])', fontsize=13)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    fig_path = 'experiments/bcc_void_analytical/bcc_vs_fcc_yield_surface.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")

except ImportError:
    print("\nMatplotlib not available; skipping plot.")

print("\n" + "=" * 70)
print("DERIVATION COMPLETE")
print("=" * 70)
