"""
Module 5: Visualization
========================
Produces all figures for the project report and presentation.

Figures:
  1. Payoff matrix heatmap
  2. Strategy convergence (FP vs BRD vs Nash)
  3. Replicator dynamics trajectory (simplex + time series)
  4. Phase portrait on 2D simplex (Hawk-Dove)
  5. ESS vs Nash comparison bar chart
  6. ML feature importance
  7. Convergence speed prediction (actual vs predicted)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ── style ────────────────────────────────────────────────────────────────────

COLORS = {
    "attacker": "#E85D24",
    "defender": "#1D9E75",
    "nash":     "#7F77DD",
    "ess":      "#BA7517",
    "stable":   "#639922",
    "unstable": "#E24B4A",
}

def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ── Figure 1: Payoff heatmap ─────────────────────────────────────────────────

def plot_payoff_matrix(A: np.ndarray, title: str = "Payoff matrix (attacker)",
                        ax=None, save_path: str = None):
    set_style()
    fig, ax = (plt.subplots(figsize=(5, 4)) if ax is None else (ax.figure, ax))
    n, m = A.shape
    im = ax.imshow(A, cmap="RdYlGn", aspect="auto",
                   vmin=-abs(A).max(), vmax=abs(A).max())
    plt.colorbar(im, ax=ax, fraction=0.04)

    for i in range(n):
        for j in range(m):
            ax.text(j, i, f"{A[i,j]:.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if abs(A[i,j]) > abs(A).max() * 0.5 else "black")

    ax.set_xticks(range(m))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"D{j+1}" for j in range(m)])
    ax.set_yticklabels([f"A{i+1}" for i in range(n)])
    ax.set_xlabel("Defender strategy")
    ax.set_ylabel("Attacker strategy")
    ax.set_title(title, fontsize=13)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 2: Convergence comparison ────────────────────────────────────────

def plot_convergence(fp_result: dict, brd_result: dict,
                     nash_p: np.ndarray, nash_q: np.ndarray,
                     save_path: str = None):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Learning dynamics convergence to Nash equilibrium", fontsize=14)

    T = len(fp_result["payoffs"])
    t = np.arange(T)
    n = nash_p.shape[0]

    # Top-left: Attacker strategy evolution (FP)
    ax = axes[0, 0]
    emp_p = fp_result["attacker_empirical"]
    for i in range(n):
        ax.plot(t, emp_p[:, i], label=f"A{i+1}", linewidth=1.5)
    for i, v in enumerate(nash_p):
        ax.axhline(v, linestyle="--", linewidth=1, alpha=0.5,
                   color=ax.lines[i].get_color())
    ax.set_title("Fictitious play — attacker strategy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Strategy probability")
    ax.legend(fontsize=9)

    # Top-right: Attacker strategy evolution (BRD)
    ax = axes[0, 1]
    emp_p_brd = brd_result["attacker_empirical"]
    for i in range(n):
        ax.plot(t, emp_p_brd[:, i], label=f"A{i+1}", linewidth=1.5)
    for i, v in enumerate(nash_p):
        ax.axhline(v, linestyle="--", linewidth=1, alpha=0.5,
                   color=ax.lines[i].get_color())
    ax.set_title("Best response dynamics — attacker strategy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Strategy probability")
    ax.legend(fontsize=9)

    # Bottom-left: Payoff over time
    ax = axes[1, 0]
    ax.plot(t, fp_result["payoffs"], label="FP payoff", color=COLORS["attacker"], linewidth=1.5)
    ax.plot(t, brd_result["payoffs"], label="BRD payoff", color=COLORS["defender"], linewidth=1.5)
    ax.set_title("Realized payoff vs time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Payoff")
    ax.legend()

    # Bottom-right: Distance to Nash over time
    ax = axes[1, 1]
    fp_emp_p = fp_result["attacker_empirical"]
    brd_emp_p = brd_result["attacker_empirical"]
    fp_dist = np.linalg.norm(fp_emp_p - nash_p, axis=1)
    brd_dist = np.linalg.norm(brd_emp_p - nash_p, axis=1)
    ax.semilogy(t, fp_dist + 1e-9, label="FP distance", color=COLORS["attacker"], linewidth=1.5)
    ax.semilogy(t, brd_dist + 1e-9, label="BRD distance", color=COLORS["defender"], linewidth=1.5)
    ax.set_title("Distance from Nash equilibrium (log scale)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||p_t - p*||")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 3: Replicator dynamics trajectory ─────────────────────────────────

def plot_replicator_trajectory(trajectory: np.ndarray, labels: list,
                                title: str = "Replicator dynamics",
                                save_path: str = None):
    set_style()
    n_strategies = trajectory.shape[1]
    T = trajectory.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13)

    # Left: time series
    ax = axes[0]
    strategy_colors = [COLORS["attacker"], COLORS["defender"], COLORS["nash"]]
    for i in range(n_strategies):
        color = strategy_colors[i % len(strategy_colors)]
        ax.plot(t, trajectory[:, i], label=labels[i], color=color, linewidth=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Population frequency")
    ax.set_title("Strategy frequencies over time")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    # Right: phase portrait (only for 2-strategy games)
    ax = axes[1]
    if n_strategies == 2:
        x_vals = trajectory[:, 0]
        ax.plot(x_vals, linewidth=2, color=COLORS["attacker"])
        ax.axhline(trajectory[-1, 0], linestyle="--", color=COLORS["ess"],
                   linewidth=1.5, label=f"ESS ≈ {trajectory[-1,0]:.2f}")
        ax.set_xlabel("Time step")
        ax.set_ylabel(f"Frequency of {labels[0]}")
        ax.set_title("Phase portrait (2-strategy)")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
    else:
        # For 3 strategies: show trajectory in 2D projection
        ax.scatter(trajectory[:, 0], trajectory[:, 1],
                   c=np.arange(T), cmap="viridis", s=8, alpha=0.6)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], "r*", markersize=14, label="Fixed point")
        ax.set_xlabel(f"Frequency {labels[0]}")
        ax.set_ylabel(f"Frequency {labels[1]}")
        ax.set_title("Trajectory projection (strategies 1 vs 2)")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 4: Simplex phase portrait (Hawk-Dove) ─────────────────────────────

def plot_hawkdove_phase(V: float = 4.0, C: float = 6.0, n_trajectories: int = 8,
                         save_path: str = None):
    """Phase portrait on [0,1] simplex for Hawk-Dove game."""
    from evolutionary_dynamics import hawk_dove_matrix, simulate_replicator, hawk_dove_ess
    set_style()
    A = hawk_dove_matrix(V, C)
    p_ess = hawk_dove_ess(V, C)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Multiple trajectories from different initial conditions
    init_points = np.linspace(0.05, 0.95, n_trajectories)
    for x0_hawk in init_points:
        x0 = np.array([x0_hawk, 1 - x0_hawk])
        res = simulate_replicator(A, x0, T=2000, dt=0.01)
        traj = res["trajectory"][:, 0]
        t = res["time"]
        ax.plot(t, traj, alpha=0.6, linewidth=1.5,
                color=COLORS["attacker"] if x0_hawk > p_ess else COLORS["defender"])

    ax.axhline(p_ess, linestyle="--", color=COLORS["ess"],
               linewidth=2.5, label=f"ESS = V/C = {p_ess:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency of Hawk")
    ax.set_title(f"Hawk-Dove replicator dynamics (V={V}, C={C})", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    red_patch = mpatches.Patch(color=COLORS["attacker"], alpha=0.6, label="Start above ESS")
    blue_patch = mpatches.Patch(color=COLORS["defender"], alpha=0.6, label="Start below ESS")
    ax.legend(handles=[ax.lines[0], red_patch, blue_patch], fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 5: Nash vs ESS comparison ────────────────────────────────────────

def plot_nash_vs_ess(nash_result: dict, ess_state: np.ndarray, labels: list,
                     save_path: str = None):
    set_style()
    n = len(nash_result["p_star"])
    x = np.arange(n)
    width = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Nash equilibrium vs Evolutionary Stable Strategy", fontsize=13)

    for ax, (player, nash_strat, title) in zip(axes, [
        ("Attacker", nash_result["p_star"], "Attacker (row player)"),
        ("Defender", nash_result["q_star"], "Defender (col player)"),
    ]):
        bars1 = ax.bar(x - width/2, nash_strat, width, label="Nash eq.",
                       color=COLORS["nash"], alpha=0.85)
        if len(ess_state) == n:
            bars2 = ax.bar(x + width/2, ess_state, width, label="Replicator fixed pt.",
                           color=COLORS["ess"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels[:n])
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 6: ML feature importance ─────────────────────────────────────────

def plot_feature_importance(importances: np.ndarray, feature_names: list,
                             title: str = "Feature importance (Random Forest)",
                             save_path: str = None):
    set_style()
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [COLORS["stable"] if v > np.median(importances) else "#888"
              for v in sorted_imp]
    ax.bar(range(len(sorted_imp)), sorted_imp, color=colors, alpha=0.85)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Importance")
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Figure 7: Convergence speed prediction ───────────────────────────────────

def plot_speed_prediction(y_test: np.ndarray, y_pred: np.ndarray,
                           r2: float, save_path: str = None):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.5, color=COLORS["attacker"], s=25)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual convergence iteration")
    ax.set_ylabel("Predicted convergence iteration")
    ax.set_title(f"Convergence speed prediction (R² = {r2:.3f})", fontsize=11)
    ax.legend()

    ax = axes[1]
    residuals = y_test - y_pred
    ax.hist(residuals, bins=25, color=COLORS["defender"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
