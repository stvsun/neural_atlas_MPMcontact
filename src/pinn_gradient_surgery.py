"""
Physics-Informed Neural Network with Gradient Surgery (PyTorch version)

Key ideas:
1. Gradient Surgery (PCGrad): resolves conflicts between objectives.
2. Symmetry Loss: enforces radial symmetry.
3. Multi-objective optimization with conflict resolution.

This implementation replaces the previous NumPy/manual-backprop model with
PyTorch autograd-based training.
"""

import argparse
import copy
import json
import math
import os
import random
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch


torch.set_default_dtype(torch.float64)

# Problem parameters
a, b = 2.0, 1.0
A11, A22 = a / b, b / a  # 2.0, 0.5
DOMAIN_A, DOMAIN_B = a, b
DEVICE = torch.device("cpu")
DTYPE = torch.float64


# ============================================================================
# Utilities
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def relative_improvement_over_window(values, window):
    if window <= 0 or len(values) < 2 * window:
        return None
    prev_window = values[-2 * window : -window]
    curr_window = values[-window:]
    prev_mean = sum(prev_window) / window
    curr_mean = sum(curr_window) / window
    return (prev_mean - curr_mean) / max(abs(prev_mean), 1e-12)


def update_gradnorm_weights(
    task_losses,
    task_grad_norms,
    task_weights,
    initial_task_losses,
    alpha=1.5,
    step_size=0.025,
    eps=1e-8,
    min_weight=0.1,
):
    """
    Update task weights using a GradNorm-style balancing rule.

    The update is multiplicative and keeps the average task weight at 1.0.
    """
    loss_ratios = task_losses / torch.clamp(initial_task_losses, min=1e-4)
    inverse_train_rates = loss_ratios / torch.clamp(loss_ratios.mean(), min=eps)
    inverse_train_rates = torch.clamp(inverse_train_rates, min=0.5, max=2.0)

    weighted_norms = task_weights * task_grad_norms
    avg_norm = torch.clamp(weighted_norms.mean(), min=eps)
    target_norms = avg_norm * torch.pow(inverse_train_rates, alpha)

    ratio = torch.clamp(
        target_norms / torch.clamp(weighted_norms, min=eps),
        min=0.5,
        max=2.0,
    )
    updated_weights = task_weights * torch.pow(ratio, step_size)
    updated_weights = torch.clamp(updated_weights, min=min_weight)
    updated_weights = updated_weights * (
        task_weights.numel() / torch.clamp(updated_weights.sum(), min=eps)
    )
    return updated_weights.detach()


def normalized_coordinates(x, y):
    return x / DOMAIN_A, y / DOMAIN_B


def normalized_radius(x, y):
    x_n, y_n = normalized_coordinates(x, y)
    return torch.sqrt(x_n**2 + y_n**2 + 1e-8)


# ============================================================================
# Neural Network (PyTorch)
# ============================================================================
class NeuralNetwork(torch.nn.Module):
    """
    Network: [2] -> [32] -> [32] -> [32] -> [1]
    Activation: tanh
    """

    def __init__(self, layer_sizes=(2, 32, 32, 32, 1)):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.layers = torch.nn.ModuleList()

        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            layer = torch.nn.Linear(in_dim, out_dim)
            std = math.sqrt(2.0 / in_dim)
            with torch.no_grad():
                layer.weight.normal_(0.0, std)
                layer.bias.zero_()
            self.layers.append(layer)

    def forward(self, x):
        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx < len(self.layers) - 1:
                out = torch.tanh(out)
        return out

    def parameter_count(self):
        return sum(param.numel() for param in self.parameters())

    def copy(self):
        first_param = next(self.parameters())
        model_copy = NeuralNetwork(layer_sizes=self.layer_sizes).to(
            device=first_param.device, dtype=first_param.dtype
        )
        model_copy.load_state_dict(copy.deepcopy(self.state_dict()))
        return model_copy


# ============================================================================
# Gradient Surgery (PCGrad)
# ============================================================================
def pcgrad(gradients_list):
    """
    Project Conflicting Gradients (PCGrad)

    If two gradients have negative dot product, project one onto the plane
    orthogonal to the other.
    """
    num_tasks = len(gradients_list)
    grad_pc = [grad.clone() for grad in gradients_list]

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                continue

            g_i = grad_pc[i]
            g_j = gradients_list[j]
            dot_product = torch.dot(g_i, g_j)

            if dot_product < 0:
                norm_squared = torch.dot(g_j, g_j)
                if norm_squared > 1e-12:
                    grad_pc[i] = g_i - (dot_product / norm_squared) * g_j

    return grad_pc


# ============================================================================
# Physics and losses
# ============================================================================
def source_term(x, y):
    x_n, y_n = normalized_coordinates(x, y)
    r = normalized_radius(x, y)
    c_x = A11 / (DOMAIN_A**2)
    c_y = A22 / (DOMAIN_B**2)
    return -((c_x + c_y) / r - (c_x * x_n**2 + c_y * y_n**2) / r**3)


def exact_solution(x, y):
    return normalized_radius(x, y)


def flatten_gradients(gradients, params):
    flat_parts = []
    for grad, param in zip(gradients, params):
        if grad is None:
            flat_parts.append(torch.zeros_like(param).reshape(-1))
        else:
            flat_parts.append(grad.reshape(-1))
    return torch.cat(flat_parts)


def unflatten_for_model(nn, flat_grad):
    grads = []
    idx = 0
    for param in nn.parameters():
        n_param = param.numel()
        grads.append(flat_grad[idx : idx + n_param].view_as(param))
        idx += n_param
    return grads


def compute_loss_components(nn, x_int, y_int, x_bc, y_bc, x_sym, y_sym, theta_sym):
    losses = {}

    # PDE residual loss using autograd second derivatives
    x_int_var = x_int.clone().detach()
    y_int_var = y_int.clone().detach()
    X_int = torch.cat([x_int_var, y_int_var], dim=1).requires_grad_(True)

    v = nn(X_int)
    grad_v = torch.autograd.grad(
        v,
        X_int,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
    )[0]

    v_x = grad_v[:, 0:1]
    v_y = grad_v[:, 1:2]

    grad_vx = torch.autograd.grad(
        v_x,
        X_int,
        grad_outputs=torch.ones_like(v_x),
        create_graph=True,
    )[0]
    grad_vy = torch.autograd.grad(
        v_y,
        X_int,
        grad_outputs=torch.ones_like(v_y),
        create_graph=True,
    )[0]

    v_xx = grad_vx[:, 0:1]
    v_yy = grad_vy[:, 1:2]

    operator = -A11 * v_xx - A22 * v_yy
    f = source_term(X_int[:, 0:1], X_int[:, 1:2])
    residual = operator - f
    losses["pde"] = torch.mean(residual**2)

    # Boundary loss
    X_bc = torch.cat([x_bc, y_bc], dim=1)
    v_bc = nn(X_bc)
    losses["bc"] = torch.mean((v_bc - 1.0) ** 2)

    # Symmetry loss in normalized (ellipse) coordinates
    X_sym = torch.cat([x_sym, y_sym], dim=1)
    v_sym = nn(X_sym)

    x_sym_n, y_sym_n = normalized_coordinates(x_sym, y_sym)
    x_rot_n = x_sym_n * torch.cos(theta_sym) - y_sym_n * torch.sin(theta_sym)
    y_rot_n = x_sym_n * torch.sin(theta_sym) + y_sym_n * torch.cos(theta_sym)
    x_rot = DOMAIN_A * x_rot_n
    y_rot = DOMAIN_B * y_rot_n
    X_rot = torch.cat([x_rot, y_rot], dim=1)
    v_rot = nn(X_rot)

    losses["symmetry"] = torch.mean((v_sym - v_rot) ** 2)

    # Center condition
    X_cen = torch.zeros((1, 2), device=X_bc.device, dtype=X_bc.dtype)
    v_cen = nn(X_cen)
    losses["center"] = torch.mean(v_cen**2)

    return losses


# ============================================================================
# Data generation
# ============================================================================
def generate_training_data(n_int=400, n_bc=100, n_sym=50, device=DEVICE, dtype=DTYPE):
    # Interior points in elliptical annulus: 0.05 < r_norm < 0.95
    interior_batches = []
    n_collected = 0

    while n_collected < n_int:
        n_candidates = max(n_int * 2, 64)
        candidates = torch.empty((n_candidates, 2), device=device, dtype=dtype)
        candidates[:, 0] = (
            2.0 * DOMAIN_A * torch.rand((n_candidates,), device=device, dtype=dtype) - DOMAIN_A
        )
        candidates[:, 1] = (
            2.0 * DOMAIN_B * torch.rand((n_candidates,), device=device, dtype=dtype) - DOMAIN_B
        )
        radius = normalized_radius(candidates[:, 0], candidates[:, 1])
        keep = (radius > 0.05) & (radius < 0.95)
        valid = candidates[keep]

        if valid.numel() == 0:
            continue

        interior_batches.append(valid)
        n_collected += valid.shape[0]

    interior = torch.cat(interior_batches, dim=0)[:n_int]
    x_int = interior[:, 0:1]
    y_int = interior[:, 1:2]

    # Boundary points on ellipse
    theta_bc = torch.linspace(0.0, 2.0 * math.pi, n_bc + 1, device=device, dtype=dtype)[:-1]
    x_bc = (DOMAIN_A * torch.cos(theta_bc)).unsqueeze(1)
    y_bc = (DOMAIN_B * torch.sin(theta_bc)).unsqueeze(1)

    # Symmetry points with equal normalized radius
    r_sym = torch.empty((n_sym, 1), device=device, dtype=dtype).uniform_(0.2, 0.9)
    theta_sym = torch.empty((n_sym, 1), device=device, dtype=dtype).uniform_(0.0, 2.0 * math.pi)
    x_sym = DOMAIN_A * r_sym * torch.cos(theta_sym)
    y_sym = DOMAIN_B * r_sym * torch.sin(theta_sym)

    rotation_angles = torch.empty((n_sym, 1), device=device, dtype=dtype).uniform_(
        0.0, 2.0 * math.pi
    )

    return (x_int, y_int), (x_bc, y_bc), (x_sym, y_sym, rotation_angles)


# ============================================================================
# Training
# ============================================================================
def train_with_gradient_surgery(
    nn,
    n_epochs=2000,
    lr=1e-3,
    use_surgery=True,
    use_gradnorm=False,
    training_data=None,
    verbose=True,
    log_every=400,
    min_epochs=500,
    early_stop_window=100,
    early_stop_min_rel_improve=1e-3,
    early_stop_patience=3,
    gradnorm_alpha=1.5,
    gradnorm_lr=0.025,
    target_total_loss=None,
):
    """
    Train PINN with optional Gradient Surgery and GradNorm.

    Stops early when relative improvement of total loss over a sliding window
    becomes too small for `early_stop_patience` consecutive checks, or when
    `target_total_loss` is reached.
    """
    if verbose:
        print(f"\n{'=' * 80}")
        training_label = f"{'WITH' if use_surgery else 'WITHOUT'} Gradient Surgery"
        if use_gradnorm:
            training_label += " + GradNorm"
        print(f"Training: {training_label}")
        print(f"{'=' * 80}")

    if training_data is None:
        first_param = next(nn.parameters())
        training_data = generate_training_data(device=first_param.device, dtype=first_param.dtype)

    (x_int, y_int), (x_bc, y_bc), (x_sym, y_sym, theta_sym) = training_data

    if verbose:
        print("\nTraining data:")
        print(f"  Interior: {len(x_int)}")
        print(f"  Boundary: {len(x_bc)}")
        print(f"  Symmetry: {len(x_sym)}")

    history = {
        "total_loss": [],
        "pde": [],
        "bc": [],
        "symmetry": [],
        "center": [],
        "conflicts_detected": [],
    }

    task_names = ["pde", "bc", "symmetry", "center"]
    params = list(nn.parameters())
    n_tasks = len(task_names)
    task_weights = torch.ones(n_tasks, dtype=params[0].dtype, device=params[0].device)
    gradnorm_task_indices = [0, 1, 2]  # Keep center objective as fixed auxiliary weight.
    initial_task_losses = None

    if use_gradnorm:
        for name in task_names:
            history[f"weight_{name}"] = []

    low_improve_streak = 0
    stopped_early = False
    stop_reason = "max_epochs"

    start_time = time.time()

    for epoch in range(n_epochs):
        losses = compute_loss_components(nn, x_int, y_int, x_bc, y_bc, x_sym, y_sym, theta_sym)

        # Per-task gradients
        grad_list = []
        for idx, task_name in enumerate(task_names):
            task_grads = torch.autograd.grad(
                losses[task_name],
                params,
                retain_graph=idx < len(task_names) - 1,
                create_graph=False,
                allow_unused=True,
            )
            grad_list.append(flatten_gradients(task_grads, params))

        task_loss_tensor = torch.stack([losses[name].detach() for name in task_names])
        grad_norms = torch.stack([torch.norm(grad.detach(), p=2) for grad in grad_list])

        if use_gradnorm:
            if initial_task_losses is None:
                initial_task_losses = torch.clamp(
                    task_loss_tensor[gradnorm_task_indices], min=1e-4
                )
            updated_primary_weights = update_gradnorm_weights(
                task_losses=task_loss_tensor[gradnorm_task_indices],
                task_grad_norms=grad_norms[gradnorm_task_indices],
                task_weights=task_weights[gradnorm_task_indices],
                initial_task_losses=initial_task_losses,
                alpha=gradnorm_alpha,
                step_size=gradnorm_lr,
            )
            task_weights[gradnorm_task_indices] = updated_primary_weights
            task_weights[3] = 1.0
            for idx, name in enumerate(task_names):
                history[f"weight_{name}"].append(float(task_weights[idx].item()))

        # Count conflicts
        conflicts = 0
        for i in range(len(grad_list)):
            for j in range(i + 1, len(grad_list)):
                if torch.dot(grad_list[i], grad_list[j]).item() < 0:
                    conflicts += 1
        history["conflicts_detected"].append(conflicts)

        weighted_grad_list = [task_weights[i] * grad_list[i] for i in range(n_tasks)]

        # Apply PCGrad if requested
        if use_surgery:
            weighted_grad_list = pcgrad(weighted_grad_list)

        # Combine and apply gradient
        combined_grad = torch.stack(weighted_grad_list, dim=0).mean(dim=0)
        grads_per_param = unflatten_for_model(nn, combined_grad)

        with torch.no_grad():
            for param, grad in zip(params, grads_per_param):
                param.add_(grad, alpha=-lr)

        total_loss = sum(losses.values())
        history["total_loss"].append(float(total_loss.item()))
        history["pde"].append(float(losses["pde"].item()))
        history["bc"].append(float(losses["bc"].item()))
        history["symmetry"].append(float(losses["symmetry"].item()))
        history["center"].append(float(losses["center"].item()))

        epoch_id = epoch + 1
        total_loss_value = history["total_loss"][-1]

        if target_total_loss is not None and total_loss_value <= target_total_loss:
            stopped_early = True
            stop_reason = (
                f"target total loss reached "
                f"({total_loss_value:.3e} <= {target_total_loss:.3e})"
            )
            if verbose:
                print(
                    f"Target reached at epoch {epoch_id}/{n_epochs}: "
                    f"total_loss={total_loss_value:.3e}"
                )
            break

        rel_improve = None
        if (
            epoch_id >= max(min_epochs, 2 * early_stop_window)
            and early_stop_window > 0
            and early_stop_patience > 0
        ):
            rel_improve = relative_improvement_over_window(
                history["total_loss"], early_stop_window
            )
            if rel_improve is not None and rel_improve < early_stop_min_rel_improve:
                low_improve_streak += 1
            else:
                low_improve_streak = 0

        if verbose and (epoch + 1) % log_every == 0:
            elapsed = time.time() - start_time
            weight_suffix = ""
            if use_gradnorm:
                weight_suffix = (
                    " | W:"
                    f" pde={task_weights[0].item():.2f},"
                    f" bc={task_weights[1].item():.2f},"
                    f" sym={task_weights[2].item():.2f},"
                    f" ctr={task_weights[3].item():.2f}"
                )
            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"Total: {total_loss.item():.4f} | "
                f"PDE: {losses['pde'].item():.4f} | "
                f"BC: {losses['bc'].item():.4f} | "
                f"Sym: {losses['symmetry'].item():.4f} | "
                f"Conflicts: {conflicts} | "
                f"Time: {elapsed:.1f}s"
                f"{weight_suffix}"
            )

        if low_improve_streak >= early_stop_patience:
            stopped_early = True
            stop_reason = (
                f"low relative improvement (< {early_stop_min_rel_improve:.2e}) "
                f"for {early_stop_patience} checks, window={early_stop_window}"
            )
            if verbose:
                final_rel = 0.0 if rel_improve is None else rel_improve
                print(
                    f"Early stop at epoch {epoch_id}/{n_epochs}: "
                    f"relative improvement={final_rel:.3e}, "
                    f"streak={low_improve_streak}"
                )
            break

    total_time = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {total_time:.1f}s")
        if stopped_early:
            print(f"Stop reason: {stop_reason}")

    history["train_time_sec"] = float(total_time)
    history["epochs_ran"] = len(history["total_loss"])
    history["stopped_early"] = bool(stopped_early)
    history["stop_reason"] = stop_reason
    history["final_total_loss"] = (
        history["total_loss"][-1] if history["total_loss"] else float("nan")
    )
    history["target_total_loss"] = target_total_loss
    return history


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_network(nn, title="PINN", verbose=True):
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Evaluation: {title}")
        print(f"{'=' * 80}")

    first_param = next(nn.parameters())
    device = first_param.device
    dtype = first_param.dtype

    n_grid = 50
    x_grid = torch.linspace(-DOMAIN_A, DOMAIN_A, n_grid, device=device, dtype=dtype)
    y_grid = torch.linspace(-DOMAIN_B, DOMAIN_B, n_grid, device=device, dtype=dtype)
    X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing="xy")

    radius = normalized_radius(X_grid, Y_grid)
    mask = radius <= 0.95

    x_flat = X_grid[mask].unsqueeze(1)
    y_flat = Y_grid[mask].unsqueeze(1)
    X_test = torch.cat([x_flat, y_flat], dim=1)

    with torch.no_grad():
        v_pred = nn(X_test)

    v_exact = exact_solution(x_flat, y_flat)
    error = torch.abs(v_pred - v_exact)

    l2_err = torch.sqrt(torch.mean(error**2)).item()
    max_err = torch.max(error).item()

    if verbose:
        print(f"L2 error: {l2_err:.6e}")
        print(f"Max error: {max_err:.6e}")

    v_pred_grid = torch.full_like(X_grid, float("nan"))
    v_exact_grid = torch.full_like(X_grid, float("nan"))
    v_pred_grid[mask] = v_pred.squeeze(1)
    v_exact_grid[mask] = v_exact.squeeze(1)

    return (
        X_grid.detach().cpu().numpy(),
        Y_grid.detach().cpu().numpy(),
        v_pred_grid.detach().cpu().numpy(),
        v_exact_grid.detach().cpu().numpy(),
        float(l2_err),
        float(max_err),
    )


# ============================================================================
# Visualization
# ============================================================================
def create_visualization(
    X_grid,
    Y_grid,
    V_base,
    V_surg,
    V_exact,
    history_baseline,
    history_surgery,
    l2_base,
    l2_surgery,
    output_path,
):
    fig = plt.figure(figsize=(18, 14))
    objective_names = ["pde", "bc", "symmetry", "center"]
    objective_colors = {
        "pde": "tab:orange",
        "bc": "tab:green",
        "symmetry": "tab:purple",
        "center": "tab:brown",
    }

    def _positive(values):
        return [max(v, 1e-12) for v in values]

    def _add_oval_outline(ax):
        ax.add_patch(
            Ellipse(
                (0.0, 0.0),
                width=2.0 * DOMAIN_A,
                height=2.0 * DOMAIN_B,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
        ax.set_xlim(-1.05 * DOMAIN_A, 1.05 * DOMAIN_A)
        ax.set_ylim(-1.05 * DOMAIN_B, 1.05 * DOMAIN_B)

    ax1 = fig.add_subplot(3, 3, 1)
    im1 = ax1.contourf(X_grid, Y_grid, V_base, levels=15, cmap="viridis")
    _add_oval_outline(ax1)
    ax1.set_title("Baseline (No Surgery)")
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1, label="v")

    ax2 = fig.add_subplot(3, 3, 2)
    im2 = ax2.contourf(X_grid, Y_grid, V_surg, levels=15, cmap="viridis")
    _add_oval_outline(ax2)
    ax2.set_title("With Gradient Surgery")
    ax2.set_aspect("equal")
    plt.colorbar(im2, ax=ax2, label="v")

    ax3 = fig.add_subplot(3, 3, 3)
    im3 = ax3.contourf(X_grid, Y_grid, V_exact, levels=15, cmap="viridis")
    _add_oval_outline(ax3)
    ax3.set_title("Exact Solution")
    ax3.set_aspect("equal")
    plt.colorbar(im3, ax=ax3, label="v")

    epochs_base = range(1, len(history_baseline["total_loss"]) + 1)
    epochs_surg = range(1, len(history_surgery["total_loss"]) + 1)

    ax4 = fig.add_subplot(3, 3, 4)
    ax4.semilogy(
        epochs_base, _positive(history_baseline["total_loss"]), "r-", lw=2, label="Baseline"
    )
    ax4.semilogy(
        epochs_surg, _positive(history_surgery["total_loss"]), "b-", lw=2, label="Surgery"
    )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Total Loss")
    ax4.set_title("Training Loss")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(epochs_base, history_baseline["conflicts_detected"], "r-", lw=2, label="Baseline")
    ax5.plot(epochs_surg, history_surgery["conflicts_detected"], "b-", lw=2, label="Surgery")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Conflicts")
    ax5.set_title("Gradient Conflicts")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    improvement = 100.0 * (l2_base - l2_surgery) / max(l2_base, 1e-12)
    avg_conf_base = tensor_mean(history_baseline["conflicts_detected"])
    avg_conf_surg = tensor_mean(history_surgery["conflicts_detected"])

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.axis("off")
    ax6.text(
        0.02,
        0.95,
        (
            "Summary\n\n"
            f"Domain: ellipse a={DOMAIN_A}, b={DOMAIN_B}\n"
            f"L2 baseline: {l2_base:.6e}\n"
            f"L2 surgery:  {l2_surgery:.6e}\n"
            f"Improvement: {improvement:.2f}%\n\n"
            f"Avg conflicts (base): {avg_conf_base:.2f}\n"
            f"Avg conflicts (surg): {avg_conf_surg:.2f}"
        ),
        va="top",
        family="monospace",
    )

    ax7 = fig.add_subplot(3, 3, 7)
    for name in objective_names:
        ax7.semilogy(
            epochs_base,
            _positive(history_baseline[name]),
            lw=1.8,
            color=objective_colors[name],
            label=name,
        )
    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("Loss")
    ax7.set_title("Baseline Objective Curves")
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    ax8 = fig.add_subplot(3, 3, 8)
    for name in objective_names:
        ax8.semilogy(
            epochs_surg,
            _positive(history_surgery[name]),
            lw=1.8,
            color=objective_colors[name],
            label=name,
        )
    ax8.set_xlabel("Epoch")
    ax8.set_ylabel("Loss")
    ax8.set_title("Surgery Objective Curves")
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    ax9 = fig.add_subplot(3, 3, 9)
    for name in objective_names:
        ax9.semilogy(
            epochs_base,
            _positive(history_baseline[name]),
            "--",
            lw=1.2,
            color=objective_colors[name],
            alpha=0.7,
            label=f"{name} (base)",
        )
        ax9.semilogy(
            epochs_surg,
            _positive(history_surgery[name]),
            "-",
            lw=1.6,
            color=objective_colors[name],
            alpha=0.95,
            label=f"{name} (surg)",
        )
    ax9.set_xlabel("Epoch")
    ax9.set_ylabel("Loss")
    ax9.set_title("Objective Curves (Comparison)")
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Experiments
# ============================================================================
def run_single_comparison(
    n_epochs=2000,
    lr=1e-3,
    n_int=400,
    n_bc=100,
    n_sym=50,
    seed=42,
    make_plot=True,
    output_dir=".",
    min_epochs=500,
    early_stop_window=100,
    early_stop_min_rel_improve=1e-3,
    early_stop_patience=3,
    use_gradnorm=False,
    gradnorm_alpha=1.5,
    gradnorm_lr=0.025,
    target_total_loss=None,
):
    set_seed(seed)
    training_data = generate_training_data(
        n_int=n_int,
        n_bc=n_bc,
        n_sym=n_sym,
        device=DEVICE,
        dtype=DTYPE,
    )

    initial_model = NeuralNetwork().to(device=DEVICE, dtype=DTYPE)
    nn_baseline = initial_model.copy()
    nn_surgery = initial_model.copy()

    print(f"\n{'=' * 80}")
    print("EXPERIMENT: Impact of Gradient Surgery (Fair Comparison)")
    print(f"{'=' * 80}")
    print(f"Seed: {seed}")
    print(f"Max epochs: {n_epochs}, LR: {lr}")
    print(
        "Optimization: Baseline="
        f"{'GradNorm' if use_gradnorm else 'uniform weighting'}, "
        "Surgery=PCGrad"
        f"{' + GradNorm' if use_gradnorm else ''}"
    )
    print(
        "Early stopping: "
        f"min_epochs={min_epochs}, window={early_stop_window}, "
        f"min_rel_improve={early_stop_min_rel_improve}, patience={early_stop_patience}"
    )
    if target_total_loss is not None:
        print(f"Target total loss: {target_total_loss:.2e}")

    history_baseline = train_with_gradient_surgery(
        nn_baseline,
        n_epochs=n_epochs,
        lr=lr,
        use_surgery=False,
        use_gradnorm=use_gradnorm,
        training_data=training_data,
        verbose=True,
        log_every=max(1, n_epochs // 5),
        min_epochs=min_epochs,
        early_stop_window=early_stop_window,
        early_stop_min_rel_improve=early_stop_min_rel_improve,
        early_stop_patience=early_stop_patience,
        gradnorm_alpha=gradnorm_alpha,
        gradnorm_lr=gradnorm_lr,
        target_total_loss=target_total_loss,
    )
    X_grid, Y_grid, V_base, V_exact, l2_base, max_base = evaluate_network(
        nn_baseline, "Baseline", verbose=True
    )

    history_surgery = train_with_gradient_surgery(
        nn_surgery,
        n_epochs=n_epochs,
        lr=lr,
        use_surgery=True,
        use_gradnorm=use_gradnorm,
        training_data=training_data,
        verbose=True,
        log_every=max(1, n_epochs // 5),
        min_epochs=min_epochs,
        early_stop_window=early_stop_window,
        early_stop_min_rel_improve=early_stop_min_rel_improve,
        early_stop_patience=early_stop_patience,
        gradnorm_alpha=gradnorm_alpha,
        gradnorm_lr=gradnorm_lr,
        target_total_loss=target_total_loss,
    )
    _, _, V_surg, _, l2_surg, max_surg = evaluate_network(
        nn_surgery, "With Gradient Surgery", verbose=True
    )

    improvement = 100.0 * (l2_base - l2_surg) / max(l2_base, 1e-12)
    result = {
        "seed": seed,
        "n_epochs": n_epochs,
        "n_epochs_max": n_epochs,
        "lr": lr,
        "l2_baseline": float(l2_base),
        "l2_surgery": float(l2_surg),
        "max_baseline": float(max_base),
        "max_surgery": float(max_surg),
        "l2_improvement_percent": float(improvement),
        "avg_conflicts_baseline": float(tensor_mean(history_baseline["conflicts_detected"])),
        "avg_conflicts_surgery": float(tensor_mean(history_surgery["conflicts_detected"])),
        "final_total_loss_baseline": float(history_baseline["final_total_loss"]),
        "final_total_loss_surgery": float(history_surgery["final_total_loss"]),
        "epochs_ran_baseline": int(history_baseline["epochs_ran"]),
        "epochs_ran_surgery": int(history_surgery["epochs_ran"]),
        "stopped_early_baseline": bool(history_baseline["stopped_early"]),
        "stopped_early_surgery": bool(history_surgery["stopped_early"]),
        "stop_reason_baseline": history_baseline["stop_reason"],
        "stop_reason_surgery": history_surgery["stop_reason"],
        "train_time_baseline_sec": float(history_baseline["train_time_sec"]),
        "train_time_surgery_sec": float(history_surgery["train_time_sec"]),
    }

    print(f"\n{'Method':<24} {'L2 Error':<16} {'Max Error':<16}")
    print("-" * 58)
    print(f"{'Baseline':<24} {l2_base:<16.6e} {max_base:<16.6e}")
    print(f"{'Gradient Surgery':<24} {l2_surg:<16.6e} {max_surg:<16.6e}")
    print(f"{'Improvement':<24} {improvement:<16.3f}%")
    print(
        f"Epochs run (base/surg): "
        f"{history_baseline['epochs_ran']}/{history_surgery['epochs_ran']}"
    )
    print(
        f"Final total loss (base/surg): "
        f"{history_baseline['final_total_loss']:.3e}/"
        f"{history_surgery['final_total_loss']:.3e}"
    )

    if make_plot:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pinn_gradient_surgery_results.png")
        create_visualization(
            X_grid,
            Y_grid,
            V_base,
            V_surg,
            V_exact,
            history_baseline,
            history_surgery,
            l2_base,
            l2_surg,
            output_path,
        )
        print(f"Saved visualization: {output_path}")

    return result


def run_benchmark(
    trials=5,
    n_epochs=800,
    lr=1e-3,
    n_int=400,
    n_bc=100,
    n_sym=50,
    base_seed=42,
    output_json="pinn_gradient_surgery_benchmark.json",
    min_epochs=500,
    early_stop_window=100,
    early_stop_min_rel_improve=1e-3,
    early_stop_patience=3,
    use_gradnorm=False,
    gradnorm_alpha=1.5,
    gradnorm_lr=0.025,
    target_total_loss=None,
):
    all_trials = []
    print(f"\n{'=' * 80}")
    print("BENCHMARK: Baseline vs Gradient Surgery")
    print(f"{'=' * 80}")
    print(f"Trials: {trials}, Epochs per trial: {n_epochs}, LR: {lr}")

    for k in range(trials):
        seed = base_seed + k
        set_seed(seed)

        training_data = generate_training_data(
            n_int=n_int,
            n_bc=n_bc,
            n_sym=n_sym,
            device=DEVICE,
            dtype=DTYPE,
        )

        initial_model = NeuralNetwork().to(device=DEVICE, dtype=DTYPE)
        nn_baseline = initial_model.copy()
        nn_surgery = initial_model.copy()

        history_base = train_with_gradient_surgery(
            nn_baseline,
            n_epochs=n_epochs,
            lr=lr,
            use_surgery=False,
            use_gradnorm=use_gradnorm,
            training_data=training_data,
            verbose=False,
            min_epochs=min_epochs,
            early_stop_window=early_stop_window,
            early_stop_min_rel_improve=early_stop_min_rel_improve,
            early_stop_patience=early_stop_patience,
            gradnorm_alpha=gradnorm_alpha,
            gradnorm_lr=gradnorm_lr,
            target_total_loss=target_total_loss,
        )
        _, _, _, _, l2_base, max_base = evaluate_network(nn_baseline, title="Baseline", verbose=False)

        history_surg = train_with_gradient_surgery(
            nn_surgery,
            n_epochs=n_epochs,
            lr=lr,
            use_surgery=True,
            use_gradnorm=use_gradnorm,
            training_data=training_data,
            verbose=False,
            min_epochs=min_epochs,
            early_stop_window=early_stop_window,
            early_stop_min_rel_improve=early_stop_min_rel_improve,
            early_stop_patience=early_stop_patience,
            gradnorm_alpha=gradnorm_alpha,
            gradnorm_lr=gradnorm_lr,
            target_total_loss=target_total_loss,
        )
        _, _, _, _, l2_surg, max_surg = evaluate_network(
            nn_surgery, title="With Surgery", verbose=False
        )

        improvement = 100.0 * (l2_base - l2_surg) / max(l2_base, 1e-12)
        trial_result = {
            "trial": k + 1,
            "seed": seed,
            "l2_baseline": float(l2_base),
            "l2_surgery": float(l2_surg),
            "max_baseline": float(max_base),
            "max_surgery": float(max_surg),
            "improvement_percent": float(improvement),
            "avg_conflicts_baseline": float(tensor_mean(history_base["conflicts_detected"])),
            "avg_conflicts_surgery": float(tensor_mean(history_surg["conflicts_detected"])),
            "final_total_loss_baseline": float(history_base["final_total_loss"]),
            "final_total_loss_surgery": float(history_surg["final_total_loss"]),
            "epochs_ran_baseline": int(history_base["epochs_ran"]),
            "epochs_ran_surgery": int(history_surg["epochs_ran"]),
            "train_time_baseline_sec": float(history_base["train_time_sec"]),
            "train_time_surgery_sec": float(history_surg["train_time_sec"]),
        }
        all_trials.append(trial_result)

        print(
            f"Trial {k + 1}/{trials} (seed={seed}) | "
            f"L2 base={l2_base:.5e}, L2 surg={l2_surg:.5e}, "
            f"improve={improvement:.2f}% | "
            f"epochs(base/surg)={history_base['epochs_ran']}/{history_surg['epochs_ran']}"
        )

    l2_base_vals = torch.tensor([item["l2_baseline"] for item in all_trials], dtype=DTYPE)
    l2_surg_vals = torch.tensor([item["l2_surgery"] for item in all_trials], dtype=DTYPE)
    imp_vals = torch.tensor([item["improvement_percent"] for item in all_trials], dtype=DTYPE)
    epochs_base_vals = torch.tensor([item["epochs_ran_baseline"] for item in all_trials], dtype=DTYPE)
    epochs_surg_vals = torch.tensor([item["epochs_ran_surgery"] for item in all_trials], dtype=DTYPE)
    final_loss_base_vals = torch.tensor(
        [item["final_total_loss_baseline"] for item in all_trials], dtype=DTYPE
    )
    final_loss_surg_vals = torch.tensor(
        [item["final_total_loss_surgery"] for item in all_trials], dtype=DTYPE
    )

    summary = {
        "trials": trials,
        "epochs_per_trial": n_epochs,
        "lr": lr,
        "l2_baseline_mean": float(l2_base_vals.mean().item()),
        "l2_baseline_std": float(l2_base_vals.std(unbiased=False).item()),
        "l2_surgery_mean": float(l2_surg_vals.mean().item()),
        "l2_surgery_std": float(l2_surg_vals.std(unbiased=False).item()),
        "improvement_percent_mean": float(imp_vals.mean().item()),
        "improvement_percent_std": float(imp_vals.std(unbiased=False).item()),
        "epochs_ran_baseline_mean": float(epochs_base_vals.mean().item()),
        "epochs_ran_surgery_mean": float(epochs_surg_vals.mean().item()),
        "final_total_loss_baseline_mean": float(final_loss_base_vals.mean().item()),
        "final_total_loss_surgery_mean": float(final_loss_surg_vals.mean().item()),
    }

    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(f"Baseline L2: {summary['l2_baseline_mean']:.6e} ± {summary['l2_baseline_std']:.6e}")
    print(f"Surgery  L2: {summary['l2_surgery_mean']:.6e} ± {summary['l2_surgery_std']:.6e}")
    print(
        f"Improvement: {summary['improvement_percent_mean']:.2f}% ± "
        f"{summary['improvement_percent_std']:.2f}%"
    )
    print(
        f"Avg epochs run (base/surg): "
        f"{summary['epochs_ran_baseline_mean']:.1f}/{summary['epochs_ran_surgery_mean']:.1f}"
    )
    print(
        f"Final total loss (base/surg): "
        f"{summary['final_total_loss_baseline_mean']:.3e}/"
        f"{summary['final_total_loss_surgery_mean']:.3e}"
    )

    payload = {"summary": summary, "trials_data": all_trials}
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved benchmark report: {os.path.abspath(output_json)}")

    return payload


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch PINN with PCGrad: fair comparison and multi-seed benchmark"
    )
    parser.add_argument("--mode", choices=["compare", "benchmark"], default="compare")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--benchmark-epochs", type=int, default=800)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-int", type=int, default=400)
    parser.add_argument("--n-bc", type=int, default=100)
    parser.add_argument("--n-sym", type=int, default=50)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--benchmark-json", default="pinn_gradient_surgery_benchmark.json")
    parser.add_argument("--min-epochs", type=int, default=500)
    parser.add_argument("--early-stop-window", type=int, default=100)
    parser.add_argument("--early-stop-min-rel-improve", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--target-total-loss", type=float, default=None)
    parser.add_argument("--use-gradnorm", action="store_true")
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-lr", type=float, default=0.025)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    print("=" * 80)
    print("PINN with Gradient Surgery and Symmetry Enforcement (PyTorch)")
    print("=" * 80)
    print("\nProblem: Anisotropic diffusion on oval (elliptical) domain")
    print(f"A = diag({A11}, {A22})")
    print(f"Ellipse semiaxes: a={DOMAIN_A}, b={DOMAIN_B}")
    print("Exact solution: v(x,y) = sqrt((x/a)^2 + (y/b)^2)")
    print(f"Backend: PyTorch {torch.__version__} on {DEVICE.type}")

    args = parse_args()
    if args.mode == "benchmark":
        run_benchmark(
            trials=args.trials,
            n_epochs=args.benchmark_epochs,
            lr=args.lr,
            n_int=args.n_int,
            n_bc=args.n_bc,
            n_sym=args.n_sym,
            base_seed=args.seed,
            output_json=args.benchmark_json,
            min_epochs=args.min_epochs,
            early_stop_window=args.early_stop_window,
            early_stop_min_rel_improve=args.early_stop_min_rel_improve,
            early_stop_patience=args.early_stop_patience,
            use_gradnorm=args.use_gradnorm,
            gradnorm_alpha=args.gradnorm_alpha,
            gradnorm_lr=args.gradnorm_lr,
            target_total_loss=args.target_total_loss,
        )
    else:
        run_single_comparison(
            n_epochs=args.epochs,
            lr=args.lr,
            n_int=args.n_int,
            n_bc=args.n_bc,
            n_sym=args.n_sym,
            seed=args.seed,
            make_plot=not args.no_plot,
            output_dir=args.output_dir,
            min_epochs=args.min_epochs,
            early_stop_window=args.early_stop_window,
            early_stop_min_rel_improve=args.early_stop_min_rel_improve,
            early_stop_patience=args.early_stop_patience,
            use_gradnorm=args.use_gradnorm,
            gradnorm_alpha=args.gradnorm_alpha,
            gradnorm_lr=args.gradnorm_lr,
            target_total_loss=args.target_total_loss,
        )


if __name__ == "__main__":
    main()
