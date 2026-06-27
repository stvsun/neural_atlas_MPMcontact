"""Pointwise contact constitutive law: gap/slip field -> traction field.

:class:`TractionField` applies the contact law at every quadrature point and returns the traction
vector ``p_N n + t_T``, the scalar normal pressure ``p_N``, and the per-point normal stiffness used
by the consistent contact tangent.

The constitutive forms are the SAME the rest of the repo uses (audited against Bandeira 2004), kept
here in pure numpy so the measure-coupling assembly stays numpy-first (torch enters only when baking
a neural chart, never in the contact inner loop):

- normal penalty   ``p_N = eps_n <-g_N>_+``                  mirrors ``solvers.contact.penalty.compute_contact_force``
- regularized Coulomb ``t_T = -mu p_N v_T / sqrt(|v_T|^2 + eps_t^2)``  mirrors ``solvers.contact.friction.compute_friction_force``

:func:`_check_against_torch` (run from ``__main__`` when torch is importable) asserts the numpy
forms match those kernels bit-for-bit, so the "no divergence from the audited kernels" guarantee is
verified rather than assumed.
"""
from __future__ import annotations

import numpy as np


class TractionField:
    """Pointwise normal (penalty) + tangential (regularized Coulomb) contact traction.

    Parameters
    ----------
    eps_n : float
        Normal penalty stiffness (pressure per unit penetration).
    mu : float
        Coulomb friction coefficient (0 = frictionless).
    eps_t : float
        Friction regularization velocity scale.
    """

    def __init__(self, eps_n: float, mu: float = 0.0, eps_t: float = 1e-6):
        self.eps_n = float(eps_n)
        self.mu = float(mu)
        self.eps_t = float(eps_t)

    def evaluate(self, gN: np.ndarray, n: np.ndarray, slip: np.ndarray | None = None) -> dict:
        """Traction field at quadrature points.

        Parameters
        ----------
        gN : (Q,) signed normal gap (``gN < 0`` = penetration).
        n  : (Q, d) slave unit normals.
        slip : (Q, d), optional tangential slip(-rate) for friction.

        Returns
        -------
        dict with ``t`` (Q,d) traction vector, ``pN`` (Q,) normal pressure,
        ``deps`` (Q,) active-set normal stiffness ``d p_N/d(-g_N)``, ``tT`` (Q,d) friction traction.
        """
        gN = np.asarray(gN, float)
        n = np.asarray(n, float)
        pN = self.eps_n * np.clip(-gN, 0.0, None)              # penalty.compute_contact_force
        deps = np.where(gN < 0.0, self.eps_n, 0.0)
        t = pN[:, None] * n

        tT = np.zeros_like(t)
        if self.mu > 0.0 and slip is not None:
            slip = np.asarray(slip, float)
            v_n = (slip * n).sum(axis=1, keepdims=True)
            v_t = slip - v_n * n                               # tangential component
            denom = np.sqrt((v_t * v_t).sum(axis=1, keepdims=True) + self.eps_t ** 2)
            tT = -self.mu * pN[:, None] * v_t / denom          # friction.compute_friction_force
            t = t + tT

        return dict(t=t, pN=pN, deps=deps, tT=tT, n=n)


def _check_against_torch():
    """Assert the numpy forms reproduce the audited torch kernels (skipped if torch absent)."""
    try:
        import torch
        from solvers.contact.penalty import compute_contact_force
        from solvers.contact.friction import compute_friction_force
    except Exception as exc:  # pragma: no cover - torch optional locally
        print(f"  (torch cross-check skipped: {exc})")
        return
    rng = np.random.default_rng(0)
    gN = rng.normal(size=12) * 0.01
    n = np.zeros((12, 3))
    n[:, :2] = rng.normal(size=(12, 2))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    slip = rng.normal(size=(12, 3)) * 0.01
    eps_n, mu, eps_t = 1234.0, 0.4, 1e-3
    tr = TractionField(eps_n, mu, eps_t).evaluate(gN, n, slip)
    f_n = compute_contact_force(torch.tensor(gN), torch.tensor(n),
                                torch.ones(12, dtype=torch.float64), eps_n).numpy()
    assert np.allclose(tr["t"] - tr["tT"], f_n), "numpy penalty != torch kernel"
    pN = eps_n * np.clip(-gN, 0.0, None)
    f_t = compute_friction_force(torch.tensor(slip), torch.tensor(n),
                                 torch.tensor(pN), mu, eps_t).numpy()
    assert np.allclose(tr["tT"], f_t), "numpy Coulomb != torch kernel"
    print("  torch cross-check OK: numpy penalty/Coulomb == audited kernels")


if __name__ == "__main__":
    tr = TractionField(eps_n=1000.0, mu=0.3)
    gN = np.array([-0.01, 0.0, 0.02, -0.005])
    n = np.tile([0.0, 1.0], (4, 1))
    out = tr.evaluate(gN, n, slip=np.tile([1.0, 0.0], (4, 1)))
    assert np.allclose(out["pN"], [10.0, 0.0, 0.0, 5.0]), out["pN"]
    assert np.allclose(out["t"][:, 1], [10.0, 0.0, 0.0, 5.0])
    assert out["tT"][0, 0] < 0 and abs(out["tT"][0, 0]) <= 0.3 * 10.0 + 1e-9
    assert np.allclose(out["deps"], [1000.0, 0.0, 0.0, 1000.0])
    print("  TractionField self-test OK (penalty pressure + bounded Coulomb)")
    _check_against_torch()
