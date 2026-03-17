#!/usr/bin/env python3
"""
Unit tests for ChartFEMSolver.

Test 1: Identity-map Poisson on a cube.
  Solve -Δu = f on [-r, r]³ with u = sin(πx)sin(πy)sin(πz) and identity decoder.
  Verify O(h²) convergence for P1 elements.

Test 2: Verify mesh generation and node classification without SDF.
"""

import math
import sys

import numpy as np
import torch

from manuscript_experiments.example2_rabbit_poisson.chart_fem_solver import ChartFEMSolver


class IdentityDecoder(torch.nn.Module):
    """Identity map: φ(ξ) = seed + ξ₁t₁ + ξ₂t₂ + ξ₃n (i.e., linear frame)."""

    def __init__(self):
        super().__init__()
        # Dummy parameter so it looks like a module
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        xi: torch.Tensor,
        seed: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        n: torch.Tensor,
        chart_scale: torch.Tensor,
    ) -> torch.Tensor:
        return (
            seed.unsqueeze(0)
            + xi[:, 0:1] * t1.unsqueeze(0)
            + xi[:, 1:2] * t2.unsqueeze(0)
            + xi[:, 2:3] * n.unsqueeze(0)
        )


def manufactured_u_np(x: np.ndarray) -> np.ndarray:
    pi = math.pi
    return np.sin(pi * x[:, 0]) * np.sin(pi * x[:, 1]) * np.sin(pi * x[:, 2])


def forcing_f_np(x: np.ndarray) -> np.ndarray:
    pi = math.pi
    return 3.0 * pi**2 * manufactured_u_np(x)


def test_identity_map_convergence():
    """Test O(h²) convergence on [-r, r]³ with identity map and no SDF filtering."""
    print("=" * 60)
    print("Test: Identity-map Poisson convergence")
    print("=" * 60)

    device = torch.device("cpu")
    dtype = torch.float64
    r = 0.4  # keep solution non-trivial (not at zero crossings of sin)

    decoder = IdentityDecoder().to(device=device, dtype=dtype)
    seed = torch.zeros(3, device=device, dtype=dtype)
    t1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    t2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    n_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    support_r = torch.tensor(r, device=device, dtype=dtype)

    n_cells_list = [4, 8, 16, 32]
    errors = []
    hs = []

    for nc in n_cells_list:
        h = 2.0 * r / nc

        solver = ChartFEMSolver(
            chart_id=0,
            decoder=decoder,
            seed=seed,
            t1=t1,
            t2=t2,
            n_vec=n_vec,
            support_r=support_r,
            n_cells=nc,
            sdf_oracle=None,  # No SDF filtering — use full cube
            device=device,
            dtype=dtype,
        )

        # With identity map, J = I, so A = I, det = 1
        solver.compute_diffusion_tensors()
        solver.assemble(forcing_fn=forcing_f_np)

        # All boundary nodes are artificial (no SDF), so treat them as Dirichlet
        # with exact values
        bc = {}
        all_boundary = set(solver.art_bc_nodes.tolist()) | set(solver.phys_bc_nodes.tolist())
        if len(all_boundary) > 0:
            x_bc = solver.nodes[list(all_boundary)]
            u_bc = manufactured_u_np(x_bc)
            for idx, val in zip(all_boundary, u_bc):
                bc[int(idx)] = float(val)

        solver.solve(bc)

        # Evaluate error at interior nodes
        x_all = solver.nodes
        u_exact = manufactured_u_np(x_all)
        err = np.abs(solver.u - u_exact)
        l2_err = np.sqrt(np.mean(err**2))
        max_err = np.max(err)

        errors.append(l2_err)
        hs.append(h)

        print(
            f"  n_cells={nc:3d}  h={h:.4f}  nodes={solver.n_nodes:6d}  "
            f"L²-err={l2_err:.4e}  max-err={max_err:.4e}"
        )

    # Check convergence rate
    print("\nConvergence rates:")
    rates = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i - 1] > 0:
            rate = math.log(errors[i - 1] / errors[i]) / math.log(hs[i - 1] / hs[i])
            rates.append(rate)
            print(f"  h={hs[i]:.4f} -> rate = {rate:.2f}")
        else:
            print(f"  h={hs[i]:.4f} -> rate = N/A (zero error)")

    # For P1 elements, expect rate ≈ 2
    if rates:
        avg_rate = np.mean(rates[-2:]) if len(rates) >= 2 else rates[-1]
        print(f"\nAverage rate (last 2): {avg_rate:.2f}")
        if avg_rate >= 1.5:
            print("✓ PASS: convergence rate ≥ 1.5 (expected ~2.0 for P1)")
        else:
            print(f"✗ FAIL: convergence rate {avg_rate:.2f} < 1.5")
            return False
    return True


def test_mesh_generation():
    """Test basic mesh generation without SDF."""
    print("\n" + "=" * 60)
    print("Test: Mesh generation (no SDF)")
    print("=" * 60)

    device = torch.device("cpu")
    dtype = torch.float64

    decoder = IdentityDecoder().to(device=device, dtype=dtype)
    seed = torch.zeros(3, device=device, dtype=dtype)
    t1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    t2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    n_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    r = 0.5
    support_r = torch.tensor(r, device=device, dtype=dtype)
    nc = 4

    solver = ChartFEMSolver(
        chart_id=0,
        decoder=decoder,
        seed=seed,
        t1=t1,
        t2=t2,
        n_vec=n_vec,
        support_r=support_r,
        n_cells=nc,
        sdf_oracle=None,
        device=device,
        dtype=dtype,
    )

    expected_nodes = (nc + 1) ** 3
    expected_tets = nc**3 * 6

    print(f"  Nodes: {solver.n_nodes} (expected {expected_nodes})")
    print(f"  Elements: {solver.n_elements} (expected {expected_tets})")
    print(f"  Phys BC nodes: {len(solver.phys_bc_nodes)}")
    print(f"  Art BC nodes: {len(solver.art_bc_nodes)}")
    print(f"  Interior nodes: {len(solver.interior_nodes)}")

    ok = True
    if solver.n_nodes != expected_nodes:
        print(f"  ✗ FAIL: expected {expected_nodes} nodes, got {solver.n_nodes}")
        ok = False
    if solver.n_elements != expected_tets:
        print(f"  ✗ FAIL: expected {expected_tets} elements, got {solver.n_elements}")
        ok = False

    # All boundary + interior should equal total nodes
    total_classified = len(solver.phys_bc_nodes) + len(solver.art_bc_nodes) + len(solver.interior_nodes)
    if total_classified != solver.n_nodes:
        print(f"  ✗ FAIL: classified {total_classified} nodes, but have {solver.n_nodes}")
        ok = False

    if ok:
        print("  ✓ PASS")
    return ok


def test_interpolation():
    """Test that FEM interpolation recovers the solution at node points."""
    print("\n" + "=" * 60)
    print("Test: Interpolation accuracy")
    print("=" * 60)

    device = torch.device("cpu")
    dtype = torch.float64

    decoder = IdentityDecoder().to(device=device, dtype=dtype)
    seed = torch.zeros(3, device=device, dtype=dtype)
    t1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    t2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    n_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    r = 0.4
    support_r = torch.tensor(r, device=device, dtype=dtype)
    nc = 8

    solver = ChartFEMSolver(
        chart_id=0,
        decoder=decoder,
        seed=seed,
        t1=t1,
        t2=t2,
        n_vec=n_vec,
        support_r=support_r,
        n_cells=nc,
        sdf_oracle=None,
        device=device,
        dtype=dtype,
    )

    # Set solution to a known function at nodes
    solver.u = manufactured_u_np(solver.nodes)

    # Evaluate at node positions (should be exact)
    u_at_nodes = solver.evaluate_at(solver.nodes)
    err = np.abs(u_at_nodes - solver.u)
    max_err = np.max(err)

    print(f"  Max interpolation error at nodes: {max_err:.4e}")
    if max_err < 1e-10:
        print("  ✓ PASS")
        return True
    else:
        print(f"  ✗ FAIL: max error {max_err:.4e} > 1e-10")
        return False


if __name__ == "__main__":
    results = []
    results.append(test_mesh_generation())
    results.append(test_interpolation())
    results.append(test_identity_map_convergence())

    print("\n" + "=" * 60)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
