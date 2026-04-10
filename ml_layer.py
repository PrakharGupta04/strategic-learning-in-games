"""
Module 4: ML Layer
===================
Uses sklearn to extract patterns from game-theoretic simulations.

Why ML here (not just math)?
  - Analytical convergence proofs tell us *whether* FP converges in zero-sum games
    (Robinson's theorem), but not *how fast* for a given matrix.
  - ML learns the mapping from payoff matrix structure → convergence speed.
  - Stability classification: given a random game, predict if replicator dynamics
    converge to an ESS or cycle — analytically hard for n > 2.

Tasks:
  1. Classification: stable vs unstable fixed points (ESS vs non-ESS) from
     payoff matrix features.
  2. Regression: predict FP convergence iteration from matrix statistics.

Features are intentionally interpretable — no black boxes.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from game_model import random_payoff_matrix, uniform_strategy
from classical_solvers import solve_minimax
from learning_dynamics import fictitious_play, best_response_dynamics, convergence_metrics
from evolutionary_dynamics import (
    hawk_dove_matrix, simulate_replicator, is_ess, find_fixed_points
)


# ── feature extraction ───────────────────────────────────────────────────────

def payoff_matrix_features(A: np.ndarray) -> np.ndarray:
    """
    Extract interpretable features from a payoff matrix.
    These capture the 'hardness' of the game for learning algorithms.
    """
    n = A.shape[0]
    return np.array([
        A.mean(),                              # Average payoff
        A.std(),                               # Payoff spread
        A.max() - A.min(),                     # Payoff range
        np.abs(A).max(),                       # Max absolute payoff
        np.linalg.matrix_rank(A),             # Matrix rank
        float(np.linalg.det(A[:min(n,3), :min(n,3)])),  # Determinant (up to 3x3)
        (A > 0).mean(),                        # Fraction of positive entries
        np.abs(A - A.mean()).mean(),            # Mean absolute deviation
        float(np.linalg.norm(A, 'fro')),       # Frobenius norm
        float(np.abs(np.linalg.eigvals(A[:min(n,3), :min(n,3)])).max()),  # Spectral radius
    ])


def learning_features(A: np.ndarray, T: int = 300) -> np.ndarray:
    """
    Features derived from running FP for T steps.
    Captures empirical convergence behaviour.
    """
    try:
        minimax = solve_minimax(A)
        fp_hist = fictitious_play(A, T=T)
        metrics = convergence_metrics(
            fp_hist, minimax["p_star"], minimax["q_star"], minimax["game_value"]
        )
        brd_hist = best_response_dynamics(A, T=T)
        brd_metrics = convergence_metrics(
            brd_hist, minimax["p_star"], minimax["q_star"], minimax["game_value"]
        )
        return np.array([
            metrics["final_payoff_gap"],
            metrics["mean_dist_to_nash"],
            metrics["oscillation_p"],
            metrics["oscillation_q"],
            metrics["payoff_variance"],
            float(metrics["converged"]),
            brd_metrics["oscillation_p"],
            brd_metrics["final_payoff_gap"],
        ])
    except Exception:
        return np.zeros(8)


# ── dataset generation ───────────────────────────────────────────────────────

def generate_convergence_dataset(n_samples: int = 200, n_strategies: int = 3,
                                  T: int = 300, seed: int = 42) -> dict:
    """
    Generate labelled dataset for ML tasks.

    Labels:
      - y_stable: 1 if FP converges fast (< T//2 iters), else 0 (slow/unstable)
      - y_speed: number of iterations until convergence (regression target)

    We use a tighter epsilon to distinguish fast vs slow convergence, creating
    a naturally balanced binary label across the dataset.
    """
    rng = np.random.default_rng(seed)
    X_matrix = []
    X_learning = []
    y_stable = []
    y_speed = []

    print(f"Generating {n_samples} game instances...")
    for i in range(n_samples):
        A = random_payoff_matrix(n_strategies, seed=rng.integers(1e6))
        try:
            mf = payoff_matrix_features(A)
            lf = learning_features(A, T=T)
            X_matrix.append(mf)
            X_learning.append(lf)

            minimax = solve_minimax(A)
            fp_hist = fictitious_play(A, T=T)
            # Use strict epsilon (0.01) so fast-converging games are clearly separated
            metrics = convergence_metrics(
                fp_hist, minimax["p_star"], minimax["q_star"], minimax["game_value"],
                epsilon=0.01
            )
            # Label: 1 = fast convergence (< T//2), 0 = slow / non-converged
            converged_fast = metrics["converged"] and metrics["converged_at"] < T // 2
            y_stable.append(int(converged_fast))
            y_speed.append(metrics["converged_at"])

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_samples} done")
        except Exception:
            continue

    return {
        "X_matrix": np.array(X_matrix),
        "X_learning": np.array(X_learning),
        "X_combined": np.hstack([X_matrix, X_learning]),
        "y_stable": np.array(y_stable),
        "y_speed": np.array(y_speed),
        "feature_names_matrix": [
            "mean_payoff", "payoff_std", "payoff_range", "max_abs_payoff",
            "matrix_rank", "determinant", "frac_positive", "mean_abs_dev",
            "frobenius_norm", "spectral_radius"
        ],
        "feature_names_learning": [
            "fp_payoff_gap", "fp_dist_nash", "fp_osc_p", "fp_osc_q",
            "fp_payoff_var", "fp_converged", "brd_osc_p", "brd_payoff_gap"
        ],
    }


# ── ML models ────────────────────────────────────────────────────────────────

def train_stability_classifier(data: dict) -> dict:
    """
    Task 1: Classify whether a game's FP dynamics converge (stable) or not.
    Model: Logistic Regression (interpretable) + Random Forest (powerful).
    """
    X = data["X_combined"]
    y = data["y_stable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression — interpretable, fast, good baseline
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_cv = cross_val_score(lr_pipe, X, y, cv=5, scoring='accuracy')

    # Random Forest — captures nonlinear interactions between features
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_cv = cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy')

    return {
        "lr_model": lr_pipe,
        "rf_model": rf_pipe,
        "lr_cv_mean": lr_cv.mean(),
        "lr_cv_std": lr_cv.std(),
        "rf_cv_mean": rf_cv.mean(),
        "rf_cv_std": rf_cv.std(),
        "lr_test_accuracy": lr_pipe.score(X_test, y_test),
        "rf_test_accuracy": rf_pipe.score(X_test, y_test),
        "rf_feature_importance": rf_pipe.named_steps["clf"].feature_importances_,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_convergence_speed_regressor(data: dict) -> dict:
    """
    Task 2: Predict how many iterations FP takes to converge.
    Target: log-transformed convergence speed (better-distributed regression target).
    Only uses matrix features (not learning features) to simulate 'prediction before run'.
    """
    X = data["X_matrix"]
    # Log-transform to make the target better distributed and reduce outlier influence
    y_raw = data["y_speed"].astype(float)
    y = np.log1p(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    rf_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(n_estimators=150, max_depth=6,
                                       min_samples_leaf=3, random_state=42))
    ])
    rf_reg.fit(X_train, y_train)

    y_pred = rf_reg.predict(X_test)
    residuals = y_test - y_pred
    # Report R² on log scale
    r2 = rf_reg.score(X_test, y_test)

    return {
        "model": rf_reg,
        "r2_score": r2,
        "y_test": y_test,
        "y_pred": y_pred,
        "residuals": residuals,
        "feature_importance": rf_reg.named_steps["reg"].feature_importances_,
        "note": "Targets are log(1 + convergence_iteration)",
    }


# ── ESS classification (evolutionary) ───────────────────────────────────────

def generate_ess_dataset(n_samples: int = 150, seed: int = 7) -> dict:
    """
    For 2x2 symmetric games: classify if replicator dynamics reach an ESS.
    Features: entries of the 2x2 symmetric payoff matrix.
    Label: 1 if interior ESS exists, 0 if corner attractors.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_samples):
        V = rng.uniform(1, 8)
        C = rng.uniform(1, 10)
        A = hawk_dove_matrix(V, C)
        result = simulate_replicator(A, x0=np.array([0.5, 0.5]), T=3000)
        x_final = result["final_state"]
        ess = is_ess(A, x_final)

        # Features: raw payoff entries + ratio
        X.append([A[0, 0], A[0, 1], A[1, 0], A[1, 1], V, C, V / (C + 1e-9)])
        y.append(int(ess and 0.05 < x_final[0] < 0.95))  # Interior ESS

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(max_iter=300)
    lr.fit(X_train, y_train)

    return {
        "model": lr,
        "accuracy": lr.score(X_test, y_test),
        "cv_scores": cross_val_score(lr, X, y, cv=5),
        "X": X,
        "y": y,
    }
