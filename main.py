"""
Main pipeline: Strategic Learning in Games
==========================================
Runs all modules end-to-end and saves results.

Run: python main.py

Output: figures/ directory with all plots + printed results table.
"""

import os
import numpy as np

# ── setup ────────────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

from game_model import get_default_payoff_matrix, game_summary, uniform_strategy
from classical_solvers import solve_minimax, find_pure_nash
from learning_dynamics import fictitious_play, best_response_dynamics, convergence_metrics
from evolutionary_dynamics import (
    hawk_dove_matrix, hawk_dove_ess, simulate_replicator,
    find_fixed_points, is_ess
)
from ml_layer import (
    generate_convergence_dataset, train_stability_classifier,
    train_convergence_speed_regressor, generate_ess_dataset
)
from visualization import (
    plot_payoff_matrix, plot_convergence, plot_replicator_trajectory,
    plot_hawkdove_phase, plot_nash_vs_ess, plot_feature_importance,
    plot_speed_prediction
)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ═══════════════════════════════════════════════════════════
# PART 1: Game model + classical solution
# ═══════════════════════════════════════════════════════════
separator("PART 1: Game Model + Classical Solution")

A = get_default_payoff_matrix()
info = game_summary(A)
print(f"Game: {info['n_attacker_strategies']}×{info['n_defender_strategies']} zero-sum")
print(f"Payoff range: {info['payoff_range'][0]:.1f} to {info['payoff_range'][1]:.1f}")

pure_ne = find_pure_nash(A)
print(f"Pure Nash equilibria: {pure_ne if pure_ne else 'None (mixed strategy required)'}")

minimax = solve_minimax(A)
print(f"\nMinimax solution (LP):")
print(f"  Attacker NE strategy:  {np.round(minimax['p_star'], 3)}")
print(f"  Defender NE strategy:  {np.round(minimax['q_star'], 3)}")
print(f"  Game value (minimax):  {minimax['game_value']:.4f}")
print(f"  Verification E[p*,q*]: {minimax['verification']:.4f}")

plot_payoff_matrix(A, save_path="figures/fig1_payoff_matrix.png")
print("→ Saved: figures/fig1_payoff_matrix.png")


# ═══════════════════════════════════════════════════════════
# PART 2: Learning dynamics
# ═══════════════════════════════════════════════════════════
separator("PART 2: Learning Dynamics (FP vs BRD)")

T_learn = 600

fp_result = fictitious_play(A, T=T_learn)
brd_result = best_response_dynamics(A, T=T_learn)

fp_metrics = convergence_metrics(fp_result, minimax["p_star"], minimax["q_star"],
                                  minimax["game_value"])
brd_metrics = convergence_metrics(brd_result, minimax["p_star"], minimax["q_star"],
                                   minimax["game_value"])

print(f"\nFictitious Play after {T_learn} iterations:")
print(f"  Final attacker strategy: {np.round(fp_result['final_p'], 3)}")
print(f"  Final defender strategy: {np.round(fp_result['final_q'], 3)}")
print(f"  Converged: {fp_metrics['converged']}  "
      f"(iteration {fp_metrics['converged_at']})")
print(f"  Final payoff gap from v*: {fp_metrics['final_payoff_gap']:.4f}")

print(f"\nBest Response Dynamics after {T_learn} iterations:")
print(f"  Final attacker strategy: {np.round(brd_result['final_p'], 3)}")
print(f"  Converged: {brd_metrics['converged']}  "
      f"(iteration {brd_metrics['converged_at']})")
print(f"  Final payoff gap from v*: {brd_metrics['final_payoff_gap']:.4f}")

plot_convergence(fp_result, brd_result, minimax["p_star"], minimax["q_star"],
                 save_path="figures/fig2_convergence.png")
print("\n→ Saved: figures/fig2_convergence.png")


# ═══════════════════════════════════════════════════════════
# PART 3: Evolutionary dynamics
# ═══════════════════════════════════════════════════════════
separator("PART 3: Evolutionary Game Dynamics")

# 3a: Hawk-Dove (2-strategy, analytic ESS known)
V, C = 4.0, 6.0
A_hd = hawk_dove_matrix(V, C)
p_ess_analytic = hawk_dove_ess(V, C)
print(f"\nHawk-Dove game (V={V}, C={C})")
print(f"  Analytic ESS (p_hawk*) = V/C = {p_ess_analytic:.4f}")

x0_hd = np.array([0.3, 0.7])
hd_result = simulate_replicator(A_hd, x0_hd, T=3000)
print(f"  Replicator converges to: p_hawk = {hd_result['final_state'][0]:.4f}")
print(f"  Is ESS: {is_ess(A_hd, hd_result['final_state'])}")

plot_hawkdove_phase(V, C, save_path="figures/fig4_hawkdove_phase.png")
print("→ Saved: figures/fig4_hawkdove_phase.png")

# 3b: 3-strategy game (same A as cybersecurity game, symmetrised)
A_sym = (A + A.T) / 2  # Symmetrise for single-population evolutionary game
x0_3 = np.array([0.4, 0.35, 0.25])
rep_result = simulate_replicator(A_sym, x0_3, T=2000)
labels_3 = ["Attack Heavy", "Balanced", "Attack Light"]

print(f"\n3-strategy evolutionary game (symmetrised payoff)")
print(f"  Initial state:    {x0_3}")
print(f"  Final state:      {np.round(rep_result['final_state'], 3)}")
print(f"  Is final ESS:     {is_ess(A_sym, rep_result['final_state'])}")

plot_replicator_trajectory(rep_result["trajectory"], labels_3,
                            title="Replicator dynamics — 3-strategy game",
                            save_path="figures/fig3_replicator.png")
print("→ Saved: figures/fig3_replicator.png")

# Fixed point analysis
print("\nSearching for fixed points (20 random initialisations)...")
fps = find_fixed_points(A_sym, T=1000, n_trials=20)
print(f"  Found {len(fps)} unique fixed point(s):")
for k, fp in enumerate(fps):
    print(f"    FP{k+1}: {np.round(fp['state'], 3)}  ESS={fp['is_ess']}")


# ═══════════════════════════════════════════════════════════
# PART 4: Nash vs ESS comparison
# ═══════════════════════════════════════════════════════════
separator("PART 4: Nash vs ESS Comparison")

ess_state = rep_result["final_state"][:3]
plot_nash_vs_ess(minimax, ess_state, labels_3,
                 save_path="figures/fig5_nash_vs_ess.png")
print(f"  Nash attacker strategy: {np.round(minimax['p_star'], 3)}")
print(f"  ESS (replicator FP):    {np.round(ess_state, 3)}")
print("→ Saved: figures/fig5_nash_vs_ess.png")


# ═══════════════════════════════════════════════════════════
# PART 5: ML layer
# ═══════════════════════════════════════════════════════════
separator("PART 5: ML — Stability Classification + Speed Prediction")

print("\nGenerating dataset (200 random games)...")
data = generate_convergence_dataset(n_samples=200, T=300)
print(f"  Stable games: {data['y_stable'].sum()}  |  Unstable: {(1-data['y_stable']).sum()}")

print("\nTraining stability classifier...")
clf_results = train_stability_classifier(data)
print(f"  Logistic Regression — CV accuracy: "
      f"{clf_results['lr_cv_mean']:.3f} ± {clf_results['lr_cv_std']:.3f}")
print(f"  Random Forest       — CV accuracy: "
      f"{clf_results['rf_cv_mean']:.3f} ± {clf_results['rf_cv_std']:.3f}")
print(f"  RF test accuracy:    {clf_results['rf_test_accuracy']:.3f}")

print("\nTraining convergence speed regressor...")
reg_results = train_convergence_speed_regressor(data)
print(f"  R² score: {reg_results['r2_score']:.3f}")

all_feature_names = (data["feature_names_matrix"] + data["feature_names_learning"])
plot_feature_importance(clf_results["rf_feature_importance"],
                         all_feature_names,
                         title="Feature importance — stability classifier",
                         save_path="figures/fig6_feature_importance.png")

plot_speed_prediction(reg_results["y_test"], reg_results["y_pred"],
                       reg_results["r2_score"],
                       save_path="figures/fig7_speed_prediction.png")
print("→ Saved: figures/fig6_feature_importance.png")
print("→ Saved: figures/fig7_speed_prediction.png")

# ESS classification
print("\nTraining ESS classifier (Hawk-Dove variants)...")
ess_clf = generate_ess_dataset(n_samples=150)
print(f"  Logistic Regression CV accuracy: "
      f"{ess_clf['cv_scores'].mean():.3f} ± {ess_clf['cv_scores'].std():.3f}")


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
separator("RESULTS SUMMARY")
print(f"""
  Game type:          Zero-sum bimatrix (Attacker vs Defender)
  Minimax value:      {minimax['game_value']:.4f}
  FP convergence:     {'Yes' if fp_metrics['converged'] else 'No'} at iter {fp_metrics['converged_at']}
  BRD convergence:    {'Yes' if brd_metrics['converged'] else 'No'} at iter {brd_metrics['converged_at']}
  Hawk-Dove ESS:      p_hawk* = {p_ess_analytic:.4f}  (V/C = {V}/{C})
  RF classifier:      {clf_results['rf_cv_mean']:.1%} accuracy (5-fold CV)
  Speed regressor:    R² = {reg_results['r2_score']:.3f}

  Saved figures:
    fig1_payoff_matrix.png
    fig2_convergence.png
    fig3_replicator.png
    fig4_hawkdove_phase.png
    fig5_nash_vs_ess.png
    fig6_feature_importance.png
    fig7_speed_prediction.png
""")
