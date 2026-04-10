"""
Module 2b: Learning Dynamics
==============================
Simulates how rational agents *learn* Nash equilibrium without central coordination.

Algorithms:
  1. Best Response Dynamics (BRD)
     - Each agent plays a pure best response to opponent's current strategy
     - Fast but may cycle in non-zero-sum games
     - In zero-sum games, may not converge but orbits the Nash equilibrium

  2. Fictitious Play (FP) [Brown, 1951]
     - Each agent plays best response to the empirical frequency of opponent's past play
     - Converges to Nash in zero-sum games (proved by Robinson, 1951)
     - Convergence is slow (O(1/t) oscillations around equilibrium)

Both algorithms track strategy histories, enabling convergence analysis and ML features.
"""

import numpy as np
from game_model import expected_payoff


# ── best response utilities ──────────────────────────────────────────────────

def best_response_attacker(A: np.ndarray, q: np.ndarray) -> int:
    """Pure best response for attacker given defender's mixed strategy q."""
    return int(np.argmax(A @ q))


def best_response_defender(A: np.ndarray, p: np.ndarray) -> int:
    """Pure best response for defender given attacker's mixed strategy p.
    Defender minimizes attacker payoff, so minimizes p^T A e_j."""
    return int(np.argmin(p @ A))


# ── Best Response Dynamics ───────────────────────────────────────────────────

def best_response_dynamics(A: np.ndarray, T: int = 500, seed: int = 42) -> dict:
    """
    Alternating best response dynamics.
    At each step, one player best-responds to the opponent's pure strategy.

    Returns history of:
      - attacker_strategies: list of pure strategy indices (length T)
      - defender_strategies: list of pure strategy indices (length T)
      - attacker_empirical: empirical frequency distributions over time
      - defender_empirical: empirical frequency distributions over time
      - payoffs: realized payoffs at each step
    """
    rng = np.random.default_rng(seed)
    n, m = A.shape

    # Initialise with uniform distributions
    p_emp = np.ones(n) / n
    q_emp = np.ones(m) / m

    att_hist, def_hist = [], []
    att_emp_hist, def_emp_hist = [], []
    payoff_hist = []

    att_counts = np.ones(n)  # Laplace smoothing to avoid div-by-zero at t=0
    def_counts = np.ones(m)

    for t in range(T):
        # Attacker best-responds to current empirical defender distribution
        i = best_response_attacker(A, q_emp)
        # Defender best-responds to current empirical attacker distribution
        j = best_response_defender(A, p_emp)

        att_counts[i] += 1
        def_counts[j] += 1

        p_emp = att_counts / att_counts.sum()
        q_emp = def_counts / def_counts.sum()

        att_hist.append(i)
        def_hist.append(j)
        att_emp_hist.append(p_emp.copy())
        def_emp_hist.append(q_emp.copy())
        payoff_hist.append(expected_payoff(A, p_emp, q_emp))

    return {
        "method": "best_response_dynamics",
        "attacker_strategies": att_hist,
        "defender_strategies": def_hist,
        "attacker_empirical": np.array(att_emp_hist),
        "defender_empirical": np.array(def_emp_hist),
        "payoffs": np.array(payoff_hist),
        "final_p": p_emp,
        "final_q": q_emp,
    }


# ── Fictitious Play ──────────────────────────────────────────────────────────

def fictitious_play(A: np.ndarray, T: int = 500, seed: int = 42) -> dict:
    """
    Fictitious Play: each player best-responds to the opponent's time-averaged strategy.
    """
    rng = np.random.default_rng(seed)
    n, m = A.shape

    # Random initial actions
    i = rng.integers(n)
    j = rng.integers(m)

    att_counts = np.zeros(n)
    def_counts = np.zeros(m)
    att_counts[i] += 1
    def_counts[j] += 1

    att_hist, def_hist = [i], [j]
    att_emp_hist = [att_counts / att_counts.sum()]
    def_emp_hist = [def_counts / def_counts.sum()]
    payoff_hist = [A[i, j]]

    for t in range(1, T):
        p_emp = att_counts / att_counts.sum()
        q_emp = def_counts / def_counts.sum()

        # Best responses
        i = best_response_attacker(A, q_emp)
        j = best_response_defender(A, p_emp)

        att_counts[i] += 1
        def_counts[j] += 1

        p_emp_new = att_counts / att_counts.sum()
        q_emp_new = def_counts / def_counts.sum()

        att_hist.append(i)
        def_hist.append(j)
        att_emp_hist.append(p_emp_new.copy())
        def_emp_hist.append(q_emp_new.copy())
        payoff_hist.append(expected_payoff(A, p_emp_new, q_emp_new))

    # 🔥 NEW: Save attacker strategy history for animation
    p_history = np.array(att_emp_hist)
    np.save("fp_history.npy", p_history)

    return {
        "method": "fictitious_play",
        "attacker_strategies": att_hist,
        "defender_strategies": def_hist,
        "attacker_empirical": p_history,
        "defender_empirical": np.array(def_emp_hist),
        "payoffs": np.array(payoff_hist),
        "final_p": att_counts / att_counts.sum(),
        "final_q": def_counts / def_counts.sum(),
    }


# ── convergence analysis ─────────────────────────────────────────────────────

def convergence_metrics(history: dict, nash_p: np.ndarray, nash_q: np.ndarray,
                        game_value: float, epsilon: float = 0.05) -> dict:
    """
    Computes convergence features — used as ML input features.
    """
    emp_p = history["attacker_empirical"]  # shape (T, n)
    emp_q = history["defender_empirical"]  # shape (T, m)
    payoffs = history["payoffs"]
    T = len(payoffs)

    # L2 distance from Nash over time
    dist_p = np.linalg.norm(emp_p - nash_p, axis=1)
    dist_q = np.linalg.norm(emp_q - nash_q, axis=1)

    # Payoff gap from game value
    payoff_gap = np.abs(payoffs - game_value)

    # Convergence iteration (first t where gap < epsilon, stays < epsilon)
    converged_at = None
    for t in range(10, T):
        if np.all(payoff_gap[t:min(t + 20, T)] < epsilon):
            converged_at = t
            break

    # Strategy oscillation: mean absolute change in empirical distribution
    osc_p = np.mean(np.abs(np.diff(emp_p, axis=0)))
    osc_q = np.mean(np.abs(np.diff(emp_q, axis=0)))

    return {
        "converged": converged_at is not None,
        "converged_at": converged_at if converged_at else T,
        "final_payoff_gap": float(payoff_gap[-1]),
        "mean_dist_to_nash": float((dist_p + dist_q).mean()),
        "oscillation_p": float(osc_p),
        "oscillation_q": float(osc_q),
        "payoff_variance": float(payoffs.var()),
        "dist_p_history": dist_p,
        "dist_q_history": dist_q,
        "payoff_gap_history": payoff_gap,
    }
