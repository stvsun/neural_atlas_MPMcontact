#!/usr/bin/env python3
"""Unit and integration tests for the differentiable elastoplasticity pipeline.

Run with:  python -m pytest experiments/torus_elastoplastic/test_elastoplastic.py -v
Or:        python experiments/torus_elastoplastic/test_elastoplastic.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure repo root is on path
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.torus_elastoplastic.return_mapping import (
    ReturnMappingState,
    smooth_return_map,
    sym_logm,
    sym_expm,
    dev,
    frobenius_norm,
)
from experiments.torus_elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from experiments.torus_elastoplastic.incremental_solver import (
    IncrementalSolver,
    cosine_anneal,
)


# ---------------------------------------------------------------------------
# Material constants (shared across tests)
# ---------------------------------------------------------------------------
E_VAL, NU_VAL = 200.0, 0.3
MU_VAL = E_VAL / (2.0 * (1.0 + NU_VAL))
K_VAL = E_VAL / (3.0 * (1.0 - 2.0 * NU_VAL))
DEVICE = "cpu"
DTYPE = torch.float64


def _uniaxial_F(eps: float) -> torch.Tensor:
    """Incremental deformation gradient for uniaxial tension (incompressible lateral)."""
    lam = 1.0 + eps
    lat = 1.0 / math.sqrt(lam)
    return torch.diag(torch.tensor([lam, lat, lat], device=DEVICE, dtype=DTYPE))


# ===================================================================
# Return mapping tests
# ===================================================================

class TestReturnMapping:
    """Tests for smooth_return_map and helper functions."""

    def test_sym_logm_identity(self):
        """logm(I) = 0."""
        I = torch.eye(3, device=DEVICE, dtype=DTYPE)
        result = sym_logm(I.unsqueeze(0))
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-14)

    def test_sym_expm_zero(self):
        """expm(0) = I."""
        Z = torch.zeros(1, 3, 3, device=DEVICE, dtype=DTYPE)
        result = sym_expm(Z)
        I = torch.eye(3, device=DEVICE, dtype=DTYPE)
        assert torch.allclose(result[0], I, atol=1e-14)

    def test_logm_expm_roundtrip(self):
        """expm(logm(A)) = A for SPD matrix."""
        A = torch.tensor(
            [[2.0, 0.5, 0.1],
             [0.5, 3.0, 0.3],
             [0.1, 0.3, 1.5]],
            device=DEVICE, dtype=DTYPE,
        ).unsqueeze(0)
        result = sym_expm(sym_logm(A))
        assert torch.allclose(result, A, atol=1e-12)

    def test_logm_grad_degenerate_eigenvalues(self):
        """Gradient of logm(A) at A=I (degenerate eigenvalues) should be finite."""
        I = torch.eye(3, device=DEVICE, dtype=DTYPE).unsqueeze(0).requires_grad_(True)
        result = sym_logm(I)
        loss = result.sum()
        loss.backward()
        assert I.grad is not None
        assert torch.isfinite(I.grad).all()

    def test_elastic_regime(self):
        """Below yield: tau ≈ 2μ·log(1+eps) for uniaxial strain."""
        mu = torch.tensor(MU_VAL, device=DEVICE)
        K_val = torch.tensor(K_VAL, device=DEVICE)
        tau_y = torch.tensor(10.0, device=DEVICE)  # high yield
        H_kin = torch.tensor(0.0, device=DEVICE)

        state = ReturnMappingState.zeros((), device=DEVICE, dtype=DTYPE)
        eps_small = 0.0005
        F_inc = _uniaxial_F(eps_small)
        tau, _ = smooth_return_map(F_inc, state, mu, K_val, tau_y, H_kin, epsilon=1e-4)

        expected = 2.0 * MU_VAL * math.log(1.0 + eps_small)
        assert abs(tau[0, 0].item() - expected) / abs(expected) < 0.01

    def test_stress_saturation_perfect_plasticity(self):
        """With H_kin=0, deviatoric stress saturates."""
        mu = torch.tensor(MU_VAL, device=DEVICE)
        K_val = torch.tensor(K_VAL, device=DEVICE)
        tau_y = torch.tensor(0.4, device=DEVICE)
        H_kin = torch.tensor(0.0, device=DEVICE)

        state = ReturnMappingState.zeros((), device=DEVICE, dtype=DTYPE)
        dev_stresses = []
        for _ in range(80):
            F_inc = _uniaxial_F(0.002)
            tau, state = smooth_return_map(F_inc, state, mu, K_val, tau_y, H_kin, epsilon=1e-3)
            tr = tau.diagonal(dim1=-2, dim2=-1).sum().item()
            dev_stresses.append(tau[0, 0].item() - tr / 3.0)

        # Last 20 values should be nearly constant
        late = dev_stresses[-20:]
        dev_range = max(late) - min(late)
        assert dev_range < 0.05 * abs(late[-1])

    def test_autograd_through_tau_y(self):
        """Gradient d(tau)/d(tau_y) is finite."""
        mu = torch.tensor(MU_VAL, device=DEVICE)
        K_val = torch.tensor(K_VAL, device=DEVICE)
        tau_y = torch.tensor(0.4, device=DEVICE, requires_grad=True)
        H_kin = torch.tensor(10.0, device=DEVICE)

        state = ReturnMappingState.zeros((), device=DEVICE, dtype=DTYPE)
        for _ in range(10):
            F_inc = _uniaxial_F(0.005)
            tau, state = smooth_return_map(F_inc, state, mu, K_val, tau_y, H_kin, epsilon=1e-3)

        tau[0, 0].backward()
        assert tau_y.grad is not None
        assert torch.isfinite(tau_y.grad)

    def test_bauschinger_effect(self):
        """Kinematic hardening produces Bauschinger effect on reverse loading."""
        mu = torch.tensor(MU_VAL, device=DEVICE)
        K_val = torch.tensor(K_VAL, device=DEVICE)
        tau_y = torch.tensor(0.4, device=DEVICE)
        H_kin = torch.tensor(30.0, device=DEVICE)

        state = ReturnMappingState.zeros((), device=DEVICE, dtype=DTYPE)

        # Load in tension
        for _ in range(40):
            F_inc = _uniaxial_F(0.003)
            tau, state = smooth_return_map(F_inc, state, mu, K_val, tau_y, H_kin, epsilon=1e-3)

        tr = tau.diagonal(dim1=-2, dim2=-1).sum().item()
        peak_dev = tau[0, 0].item() - tr / 3.0

        # Reverse load
        comp_dev = None
        for _ in range(80):
            F_inc = _uniaxial_F(-0.003)
            tau, state = smooth_return_map(F_inc, state, mu, K_val, tau_y, H_kin, epsilon=1e-3)
            tr = tau.diagonal(dim1=-2, dim2=-1).sum().item()
            d = tau[0, 0].item() - tr / 3.0
            if comp_dev is None and d < 0 and state.ep_bar.item() > 0.01:
                comp_dev = d

        assert comp_dev is not None, "Did not detect compression yield"
        assert abs(comp_dev) < abs(peak_dev), "Bauschinger: |comp yield| should be < |tension peak|"


# ===================================================================
# Vector FEM tests
# ===================================================================

class TestChartVectorFEM:
    """Tests for ChartVectorFEMSolver."""

    def test_patch_test(self):
        """Linear displacement => zero internal force at interior nodes."""
        solver = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device=DEVICE, dtype=DTYPE)
        A = torch.tensor(
            [[0.01, 0.02, -0.01],
             [0.005, -0.01, 0.015],
             [-0.02, 0.01, 0.005]],
            device=DEVICE, dtype=DTYPE,
        )
        b = torch.tensor([0.1, -0.05, 0.03], device=DEVICE, dtype=DTYPE)
        u = solver.nodes @ A.T + b

        stress_fn, _ = solver.make_neo_hookean(1.0, 10.0)
        f_int = solver.internal_forces(u, stress_fn)

        interior = ~solver.boundary_mask
        max_f = torch.max(torch.abs(f_int[interior])).item()
        assert max_f < 1e-10

    def test_mms_convergence(self):
        """O(h^2) convergence for manufactured solution with body force."""
        eps_mms = 0.005
        mu_mms, K_mms = 1.0, 10.0

        def u_mfg(x):
            s = eps_mms * torch.sin(math.pi * x[:, 0]) * torch.sin(math.pi * x[:, 1]) * torch.sin(math.pi * x[:, 2])
            return torch.stack([s, s, s], dim=1)

        def body_force(x):
            """Body force f = -div(P) via autograd."""
            x_ad = x.clone().requires_grad_(True)
            u = u_mfg(x_ad)
            grads = []
            for i in range(3):
                g = torch.autograd.grad(u[:, i], x_ad, grad_outputs=torch.ones(x_ad.shape[0], device=DEVICE, dtype=DTYPE), create_graph=True, retain_graph=True)[0]
                grads.append(g)
            grad_u = torch.stack(grads, dim=1)
            F_val = torch.eye(3, device=DEVICE, dtype=DTYPE).unsqueeze(0) + grad_u
            P = ChartVectorFEMSolver.neo_hookean_stress(F_val, mu_mms, K_mms)
            div_P = torch.zeros_like(u)
            for i in range(3):
                for J in range(3):
                    dPdx = torch.autograd.grad(P[:, i, J], x_ad, grad_outputs=torch.ones(x_ad.shape[0], device=DEVICE, dtype=DTYPE), create_graph=False, retain_graph=True)[0][:, J]
                    div_P[:, i] = div_P[:, i] + dPdx
            return -div_P.detach()

        errors = []
        for nc in [4, 8]:
            solver = ChartVectorFEMSolver(n_cells=nc, support_r=1.0, device=DEVICE, dtype=DTYPE)
            stress_fn, tangent_fn = solver.make_neo_hookean(mu_mms, K_mms)

            # Assemble body force
            centroids = solver.nodes[solver.elements].mean(dim=1)
            f_cent = body_force(centroids)
            f_ext = torch.zeros(solver.n_nodes, 3, device=DEVICE, dtype=DTYPE)
            f_ext_elem = (solver.vol[:, None, None] / 4.0) * f_cent.unsqueeze(1).expand(-1, 4, -1)
            idx = solver.elements.reshape(-1).unsqueeze(-1).expand(-1, 3)
            f_ext.scatter_add_(0, idx, f_ext_elem.reshape(-1, 3))

            u_bc = u_mfg(solver.nodes)
            u_sol = solver.solve_nonlinear(stress_fn, tangent_fn, f_ext, u_bc, solver.boundary_mask, max_iter=30, tol=1e-12)
            err = torch.norm(u_sol - u_mfg(solver.nodes)) / max(torch.norm(u_mfg(solver.nodes)).item(), 1e-30)
            errors.append(err.item())

        if errors[0] > 1e-15 and errors[1] > 1e-15:
            rate = math.log(errors[0] / errors[1]) / math.log(2)
            assert rate > 1.5, f"MMS rate {rate:.2f} < 1.5"

    def test_autograd_material_params(self):
        """Gradient flows through mu and K."""
        solver = ChartVectorFEMSolver(n_cells=4, support_r=1.0, device=DEVICE, dtype=DTYPE)
        mu = torch.tensor(1.0, device=DEVICE, dtype=DTYPE, requires_grad=True)
        K_val = torch.tensor(10.0, device=DEVICE, dtype=DTYPE, requires_grad=True)

        stress_fn = lambda F: solver.neo_hookean_stress(F, mu, K_val)
        tangent_fn = lambda F: solver.neo_hookean_tangent(F, mu, K_val)

        # One Newton step from u=0
        u_zero = torch.zeros(solver.n_nodes, 3, device=DEVICE, dtype=DTYPE)
        K_mat = solver.tangent_stiffness(u_zero, tangent_fn)
        f_ext = torch.zeros(solver.n_nodes, 3, device=DEVICE, dtype=DTYPE)
        f_ext[:, 2] = -0.01

        bc_dof = solver.boundary_mask.unsqueeze(1).expand(-1, 3).reshape(-1)
        rhs = f_ext.reshape(-1).clone()
        K_mat[bc_dof, :] = 0.0
        K_mat[:, bc_dof] = 0.0
        K_mat[bc_dof, bc_dof] = 1.0
        rhs[bc_dof] = 0.0

        u_sol = torch.linalg.solve(K_mat, rhs)
        interior = torch.where(~solver.boundary_mask)[0]
        obj = u_sol[3 * interior[len(interior)//2] + 2]
        obj.backward()

        assert mu.grad is not None and torch.isfinite(mu.grad)
        assert K_val.grad is not None and torch.isfinite(K_val.grad)


# ===================================================================
# Incremental solver tests
# ===================================================================

class TestIncrementalSolver:
    """Tests for IncrementalSolver."""

    def _make_solver(self):
        return ChartVectorFEMSolver(n_cells=4, support_r=0.5, device=DEVICE, dtype=DTYPE)

    def test_elastic_only(self):
        """High yield stress => zero plastic strain."""
        fem = self._make_solver()
        mu = torch.tensor(MU_VAL, device=DEVICE)
        K_val = torch.tensor(K_VAL, device=DEVICE)
        tau_y = torch.tensor(100.0, device=DEVICE)
        H_kin = torch.tensor(0.0, device=DEVICE)

        solver = IncrementalSolver(fem, mu, K_val, tau_y, H_kin, epsilon=1e-3)

        bc_schedule = []
        for step in range(3):
            lam = (step + 1) / 3
            u_bc = torch.zeros_like(fem.nodes)
            u_bc[:, 0] = 0.005 * lam * fem.nodes[:, 0]
            bc_schedule.append(u_bc)

        _, state_hist = solver.solve_history(
            bc_schedule, fem.boundary_mask, verbose=False, max_newton_iter=15, tol=1e-10,
        )
        assert state_hist[-1].ep_bar.max().item() < 1e-6

    def test_plastic_accumulation(self):
        """Low yield stress => positive plastic strain."""
        fem = self._make_solver()
        mu = torch.tensor(MU_VAL, device=DEVICE)
        K_val = torch.tensor(K_VAL, device=DEVICE)
        tau_y = torch.tensor(0.5, device=DEVICE)
        H_kin = torch.tensor(10.0, device=DEVICE)

        solver = IncrementalSolver(fem, mu, K_val, tau_y, H_kin, epsilon=1e-3)

        bc_schedule = []
        for step in range(5):
            lam = (step + 1) / 5
            u_bc = torch.zeros_like(fem.nodes)
            u_bc[:, 0] = 0.03 * lam * fem.nodes[:, 0]
            bc_schedule.append(u_bc)

        _, state_hist = solver.solve_history(
            bc_schedule, fem.boundary_mask, verbose=False, max_newton_iter=25, tol=1e-8,
        )
        assert state_hist[-1].ep_bar.max().item() > 0.001

    def test_cosine_anneal(self):
        """Cosine annealing schedule is monotonically decreasing."""
        vals = [cosine_anneal(i, 100, 0.1, 0.001) for i in range(100)]
        assert abs(vals[0] - 0.1) < 1e-10
        assert abs(vals[-1] - 0.001) < 1e-10
        assert all(vals[i] >= vals[i+1] - 1e-15 for i in range(len(vals)-1))


# ===================================================================
# Run tests
# ===================================================================

if __name__ == "__main__":
    # Simple test runner (also works with pytest)
    test_classes = [TestReturnMapping, TestChartVectorFEM, TestIncrementalSolver]
    n_pass = 0
    n_fail = 0
    n_total = 0

    for cls in test_classes:
        obj = cls()
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")
        for name in dir(obj):
            if name.startswith("test_"):
                n_total += 1
                try:
                    getattr(obj, name)()
                    print(f"  {name}: PASS")
                    n_pass += 1
                except Exception as e:
                    print(f"  {name}: FAIL — {e}")
                    n_fail += 1

    print(f"\n{'='*60}")
    print(f"  {n_pass}/{n_total} passed, {n_fail} failed")
    print(f"{'='*60}")
