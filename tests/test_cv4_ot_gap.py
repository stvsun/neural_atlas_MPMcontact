"""CV-4 nine-disc unit cell — OT-gap field acceptance test.

Verifies the measure-coupling (optimal-transport) driver ``cv4_ot_gap`` on a coarse disc mesh:
a single elastic disc squeezed equibiaxially by four RIGID FLAT WALLS, with the per-wall confining
force N EMERGING from the OT-gap penalty contact (not prescribed).  The recovered centre stress must
match the equibiaxial closed form sigma_xx = sigma_yy = -2 N / (pi R t), the contact must be
D4-balanced (four equal wall forces), and Newton must converge.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.contact.cv_numerical.cv4_ot_gap import run


def test_cv4_ot_gap_equibiaxial():
    # coarse mesh -> ~0.6 s; tolerances still tight because the physics is geometry-exact.
    m, _ = run(n_rings=28, verbose=False)

    assert m["converged"], f"Newton did not converge ({m['iters']} iters)"
    assert m["N_emergent"] > 0.0, "emergent confining force must be positive"

    # equibiaxial centre stress vs closed form -2N/(pi R t) at the emergent N
    assert m["sxx_relerr"] < 0.03, f"centre sigma_xx err {m['sxx_relerr']:.4f}"
    assert m["syy_relerr"] < 0.03, f"centre sigma_yy err {m['syy_relerr']:.4f}"

    # isotropy + zero shear (equibiaxial state)
    assert m["equibiaxial_anisotropy"] < 0.03, f"anisotropy {m['equibiaxial_anisotropy']:.4f}"
    assert m["shear_rel"] < 0.03, f"shear {m['shear_rel']:.4f}"

    # D4 symmetry: the four emergent wall forces are equal
    assert m["force_imbalance"] < 0.05, f"wall-force imbalance {m['force_imbalance']:.4f}"
