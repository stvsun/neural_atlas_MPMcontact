"""V&V for the ACTIVE chart-based elastostatic FEM (ported from the frozen archive).

This is the numerical-CV foundation (plan M0): a P1/P2 tetrahedral small-strain
linear-elastic FEM that solves on a ChartDecoder coordinate chart. The port lifts only the
elasticity core (no crack/DENNS/J-integral baggage) into active `solvers/fem/`, with the
chart-Jacobian inverse routed through `common.geometry.stabilized_jacobian_ops` (SVD-clamped)
so an ill-conditioned NEURAL chart cannot blow up the assembly.

Covers: patch test (machine-zero interior forces), uniaxial small-strain stress (exact),
MMS O(h^2) convergence, a near-identity ChartDecoder solve, and two-chart Schwarz coupling.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.fem.chart_vector_fem import ChartVectorFEMSolver          # noqa: E402
from solvers.fem.linear_elastic import make_linear_elastic_small_strain  # noqa: E402

DT = torch.float64


def _lame(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def test_patch_test_linear_displacement():
    """A linear displacement field must produce ZERO net internal force at interior nodes
    (constant stress => div(sigma)=0). The classic FEM patch test."""
    solver = ChartVectorFEMSolver(n_cells=5, support_r=1.0, chart_decoder=None, dtype=DT)
    stress_fn, _ = make_linear_elastic_small_strain(100.0, 0.3)
    # u = A x + b  (linear) -> constant grad u -> constant strain/stress
    A = torch.tensor([[0.01, 0.004, -0.002],
                      [0.003, -0.006, 0.005],
                      [-0.001, 0.002, 0.008]], dtype=DT)
    u = solver.nodes @ A.T
    f_int = solver.internal_forces(u, stress_fn)
    interior = ~solver.boundary_mask
    assert interior.any()
    assert f_int[interior].abs().max().item() < 1e-10


def test_uniaxial_small_strain_exact():
    """Prescribed uniaxial strain -> sigma_xx = (lambda + 2 mu) eps_xx, exact for the
    constant-tangent small-strain model (Newton converges in one step)."""
    E, nu, r = 100.0, 0.3, 1.0
    solver = ChartVectorFEMSolver(n_cells=6, support_r=r, chart_decoder=None, dtype=DT)
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)
    delta = 0.01
    u_bc = torch.zeros(solver.n_nodes, 3, dtype=DT)
    u_bc[:, 0] = (solver.nodes[:, 0] + r) / (2 * r) * delta
    f_ext = torch.zeros(solver.n_nodes, 3, dtype=DT)
    u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc,
                               solver.boundary_mask, max_iter=10, tol=1e-12)
    P = stress_fn(solver.compute_F(u))
    lam, mu = _lame(E, nu)
    expected = (lam + 2 * mu) * (delta / (2 * r))
    got = P[:, 0, 0].mean().item()
    assert abs(got - expected) / expected < 1e-6


def _mms_small_strain_body_force(x, E, nu, eps_amp):
    """f = -div(sigma(eps(u_mms))) for u_mms = eps_amp*sin(pi x)sin(pi y)sin(pi z)*[1,1,1],
    small-strain linear elasticity, via autograd (exact for the manufactured field)."""
    lam, mu = _lame(E, nu)
    pi = math.pi
    xad = x.clone().requires_grad_(True)
    s = eps_amp * torch.sin(pi * xad[:, 0]) * torch.sin(pi * xad[:, 1]) * torch.sin(pi * xad[:, 2])
    u = torch.stack([s, s, s], dim=1)
    grads = [torch.autograd.grad(u[:, i], xad, torch.ones(xad.shape[0], dtype=x.dtype),
                                 create_graph=True, retain_graph=True)[0] for i in range(3)]
    grad_u = torch.stack(grads, dim=1)                       # (N,3,3)
    eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
    tr = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]
    I3 = torch.eye(3, dtype=x.dtype)
    sigma = lam * tr.view(-1, 1, 1) * I3 + 2 * mu * eps
    div = torch.zeros_like(u)
    for i in range(3):
        for J in range(3):
            div[:, i] = div[:, i] + torch.autograd.grad(
                sigma[:, i, J], xad, torch.ones(xad.shape[0], dtype=x.dtype),
                create_graph=False, retain_graph=True)[0][:, J]
    return -div.detach()


def _mms_u(x, eps_amp):
    pi = math.pi
    s = eps_amp * torch.sin(pi * x[:, 0]) * torch.sin(pi * x[:, 1]) * torch.sin(pi * x[:, 2])
    return torch.stack([s, s, s], dim=1)


def test_mms_convergence_O_h2():
    """Manufactured-solution convergence: L2 displacement error should fall ~O(h^2)."""
    E, nu, eps_amp = 100.0, 0.3, 0.005
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)
    res, errs = [4, 8, 16], []
    for nc in res:
        solver = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=None, dtype=DT)
        u_bc = _mms_u(solver.nodes, eps_amp)
        # consistent nodal load from the centroid body force (lumped P1)
        centroids = solver.nodes[solver.elements[:, :4]].mean(dim=1)
        f_cent = _mms_small_strain_body_force(centroids, E, nu, eps_amp)
        f_ext = torch.zeros(solver.n_nodes, 3, dtype=DT)
        f_elem = (solver.vol[:, None, None] / 4.0) * f_cent.unsqueeze(1).expand(-1, 4, -1)
        idx = solver.elements[:, :4].reshape(-1).unsqueeze(-1).expand(-1, 3)
        f_ext.scatter_add_(0, idx, f_elem.reshape(-1, 3))
        u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc,
                                   solver.boundary_mask, max_iter=20, tol=1e-12)
        u_ref = _mms_u(solver.nodes, eps_amp)
        errs.append((torch.norm(u - u_ref) / torch.norm(u_ref)).item())
    rates = [math.log(errs[i - 1] / errs[i]) / math.log(res[i] / res[i - 1])
             for i in range(1, len(errs))]
    assert errs[-1] < errs[0]                                # refinement helps
    assert max(rates) > 1.7                                  # at least one clean O(h^2) step
    assert min(rates) > 1.2                                  # no stagnation


def test_chartdecoder_solve_stable():
    """A near-identity ChartDecoder map solves stably (finite displacement & stress); the
    stabilized-Jacobian pushforward keeps det J well-conditioned."""
    from common.models import ChartDecoder
    dec = ChartDecoder(width=16, depth=2).double()
    dec.raw_scale = torch.nn.Parameter(torch.tensor(-5.0, dtype=DT))   # tiny residual
    dk = dict(seed=torch.zeros(3, dtype=DT),
              t1=torch.tensor([1.0, 0, 0], dtype=DT), t2=torch.tensor([0, 1.0, 0], dtype=DT),
              n=torch.tensor([0, 0, 1.0], dtype=DT), chart_scale=torch.tensor(1.0, dtype=DT))
    solver = ChartVectorFEMSolver(n_cells=5, support_r=1.0, chart_decoder=dec,
                                  decoder_kwargs=dk, dtype=DT)
    assert solver.geom_valid.all()                           # all elements well-posed
    assert solver.geom_detJ.min().item() > 0                 # no foldover
    stress_fn, tangent_fn = make_linear_elastic_small_strain(100.0, 0.3)
    delta = 0.005
    u_bc = torch.zeros(solver.n_nodes, 3, dtype=DT)
    u_bc[:, 0] = (solver.nodes[:, 0] + 1.0) / 2.0 * delta
    f_ext = torch.zeros(solver.n_nodes, 3, dtype=DT)
    u = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc,
                               solver.boundary_mask, max_iter=20, tol=1e-9)
    assert torch.isfinite(u).all()
    assert 0 < u.abs().max().item() < 5 * delta


def test_two_chart_schwarz_uniaxial():
    """Two overlapping charts (a split bar) under uniaxial strain: multiplicative Schwarz
    completes and each chart reports a finite, consistent uniaxial stress."""
    from solvers.fem.schwarz_vector_fem import SchwarzVectorFEMSolver
    E, nu, delta, nc = 100.0, 0.3, 0.01, 5
    s1 = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=None, dtype=DT)
    s2 = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, chart_decoder=None, dtype=DT)
    for s, shift in ((s1, -1.0), (s2, +1.0)):
        s.nodes = s.nodes.clone(); s.nodes[:, 0] += shift
        s.nodes_phys = s.nodes.clone()
        th = s.h * 0.1
        s.boundary_mask = (
            (s.nodes[:, 0] < shift - 1.0 + th) | (s.nodes[:, 0] > shift + 1.0 - th) |
            (s.nodes[:, 1] < -1.0 + th) | (s.nodes[:, 1] > 1.0 - th) |
            (s.nodes[:, 2] < -1.0 + th) | (s.nodes[:, 2] > 1.0 - th))
    seeds = torch.tensor([[-1.0, 0, 0], [1.0, 0, 0]], dtype=DT)
    stress_fn, tangent_fn = make_linear_elastic_small_strain(E, nu)

    def phys_bc_fn(nodes_phys):
        n = len(nodes_phys); u_bc = np.zeros((n, 3)); mask = np.zeros(n, dtype=bool); tol = 0.15
        sel = ((nodes_phys[:, 0] < -2.0 + tol) | (nodes_phys[:, 0] > 2.0 - tol) |
               (np.abs(nodes_phys[:, 1]) > 1.0 - tol) | (np.abs(nodes_phys[:, 2]) > 1.0 - tol))
        u_bc[sel, 0] = delta * (nodes_phys[sel, 0] + 2.0) / 4.0
        mask[sel] = True
        return u_bc, mask

    schwarz = SchwarzVectorFEMSolver(chart_solvers=[s1, s2], seeds=seeds, neighbors=[[1], [0]])
    u_charts = schwarz.solve(stress_fn, tangent_fn, phys_bc_fn,
                             max_schwarz_iters=10, tol=1e-4, newton_max_iter=10, newton_tol=1e-8)
    assert u_charts[0] is not None and u_charts[1] is not None
    lam, mu = _lame(E, nu)
    expected = (lam + 2 * mu) * (delta / 4.0)
    for P in schwarz.get_stress(stress_fn):
        if len(P) > 0:
            assert np.isfinite(P[:, 0, 0].mean().item())
            assert abs(P[:, 0, 0].mean().item() - expected) / expected < 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
