"""
Module 1: Game Model
====================
Defines the two-player zero-sum game between Attacker and Defender.

Game setup:
- Players: Attacker (row), Defender (column)
- Strategies: Attacker chooses targets {T1, T2, T3}, Defender allocates resources to nodes
- Payoff: Attacker gains = Defender loses (zero-sum)
- Matrix A[i][j] = attacker payoff when attacker plays i, defender plays j
"""

import numpy as np


# ── payoff matrix ────────────────────────────────────────────────────────────

def get_default_payoff_matrix() -> np.ndarray:
    """
    3x3 zero-sum payoff matrix (attacker's perspective).
    Row i = attacker strategy, Column j = defender strategy.
    Entry A[i,j] = attacker gain = defender loss.

    Interpretation:
      - Attacker targets: {Critical infra, Database, API gateway}
      - Defender allocates: {Heavy on infra, Balanced, Heavy on API}
    """
    return np.array([
        [ 3, -1,  2],
        [-1,  4, -2],
        [ 2, -3,  5],
    ], dtype=float)


def random_payoff_matrix(n: int = 3, seed: int = None) -> np.ndarray:
    """Generate a random n×n zero-sum payoff matrix in [-5, 5]."""
    rng = np.random.default_rng(seed)
    return rng.integers(-5, 6, size=(n, n)).astype(float)


# ── mixed strategy utilities ─────────────────────────────────────────────────

def expected_payoff(A: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """
    E[payoff] = p^T A q
    p: attacker mixed strategy (row player)
    q: defender mixed strategy (col player)
    """
    return float(p @ A @ q)


def is_valid_mixed_strategy(s: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.all(s >= -tol) and abs(s.sum() - 1.0) < tol)


def uniform_strategy(n: int) -> np.ndarray:
    return np.ones(n) / n


# ── game info ────────────────────────────────────────────────────────────────

def game_summary(A: np.ndarray) -> dict:
    n, m = A.shape
    return {
        "n_attacker_strategies": n,
        "n_defender_strategies": m,
        "payoff_range": (A.min(), A.max()),
        "is_zero_sum": True,
        "attacker_labels": [f"A{i+1}" for i in range(n)],
        "defender_labels": [f"D{j+1}" for j in range(m)],
    }
