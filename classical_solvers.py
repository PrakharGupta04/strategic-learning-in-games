"""
Module 2a: Classical Solvers
=============================
Solves zero-sum games analytically using:
  1. Minimax Theorem (Von Neumann, 1928)
  2. Linear Programming (reduction of matrix game to LP)

Key concepts:
  - Minimax value v*: max_p min_q p^T A q = min_q max_p p^T A q
  - Nash equilibrium (p*, q*): neither player can unilaterally improve
  - In zero-sum games, Nash equilibrium = minimax solution
"""

import numpy as np
from scipy.optimize import linprog
from game_model import expected_payoff


# ── pure strategy analysis ───────────────────────────────────────────────────

def find_pure_nash(A: np.ndarray):
    """
    Find pure strategy Nash equilibria by checking best-response conditions.
    (i*, j*) is a pure NE iff A[i*,j*] >= A[i,j*] for all i  (attacker BR)
                               and A[i*,j*] <= A[i*,j] for all j (defender BR)
    Returns list of (row, col) pairs; empty list if none exist.
    """
    n, m = A.shape
    equilibria = []
    for i in range(n):
        for j in range(m):
            # Attacker cannot gain more by switching row
            attacker_br = all(A[i, j] >= A[k, j] for k in range(n))
            # Defender cannot lose less by switching column (minimize loss)
            defender_br = all(A[i, j] <= A[i, l] for l in range(m))
            if attacker_br and defender_br:
                equilibria.append((i, j))
    return equilibria


def find_saddle_point(A: np.ndarray):
    """
    Saddle point = pure NE. Value = minimax = maximin.
    Returns (row, col, value) or None.
    """
    pure_ne = find_pure_nash(A)
    if pure_ne:
        i, j = pure_ne[0]
        return i, j, A[i, j]
    return None


# ── LP-based minimax solver ──────────────────────────────────────────────────

def solve_attacker_lp(A: np.ndarray):
    """
    Attacker maximizes game value v subject to:
      A^T p >= v * 1  (each defender strategy held against: attacker gets >= v)
      sum(p) = 1, p >= 0

    Reformulated for linprog (minimization):
      min  -v
      s.t. -A^T p + v <= 0   =>  each col: sum_i A[i,j]*p[i] >= v
           sum(p) = 1
           p >= 0, v free

    Variables: x = [p_0, ..., p_{n-1}, v]  (n+1 variables)
    """
    n, m = A.shape
    # Shift matrix to make all entries positive (ensures v > 0 for LP stability)
    shift = -A.min() + 1 if A.min() <= 0 else 0
    B = A + shift

    # Objective: minimize -v  =>  c = [0,...,0, -1]
    c = np.zeros(n + 1)
    c[-1] = -1.0

    # Inequality: -B^T p + v <= 0  =>  for each j: -sum_i B[i,j]*p[i] + v <= 0
    A_ub = np.zeros((m, n + 1))
    for j in range(m):
        A_ub[j, :n] = -B[:, j]
        A_ub[j, n] = 1.0
    b_ub = np.zeros(m)

    # Equality: sum(p) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Bounds: p_i >= 0, v unbounded
    bounds = [(0, None)] * n + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if res.success:
        p_star = res.x[:n]
        p_star = np.clip(p_star, 0, None)
        p_star /= p_star.sum()
        v_shifted = res.x[n]
        v_star = v_shifted - shift
        return p_star, v_star
    raise ValueError(f"LP failed for attacker: {res.message}")


def solve_defender_lp(A: np.ndarray):
    """
    Defender minimizes game value v subject to:
      A q <= v * 1  (each attacker strategy: defender keeps loss <= v)
      sum(q) = 1, q >= 0

    Variables: y = [q_0, ..., q_{m-1}, v]
    """
    n, m = A.shape
    shift = -A.min() + 1 if A.min() <= 0 else 0
    B = A + shift

    # Objective: minimize v  =>  c = [0,...,0, 1]
    c = np.zeros(m + 1)
    c[-1] = 1.0

    # Inequality: B q - v <= 0  =>  for each i: sum_j B[i,j]*q[j] - v <= 0
    A_ub = np.zeros((n, m + 1))
    for i in range(n):
        A_ub[i, :m] = B[i, :]
        A_ub[i, m] = -1.0
    b_ub = np.zeros(n)

    # Equality: sum(q) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if res.success:
        q_star = res.x[:m]
        q_star = np.clip(q_star, 0, None)
        q_star /= q_star.sum()
        v_shifted = res.x[m]
        v_star = v_shifted - shift
        return q_star, v_star
    raise ValueError(f"LP failed for defender: {res.message}")


def solve_minimax(A: np.ndarray) -> dict:
    """
    Full minimax solution. Returns Nash equilibrium strategies and game value.
    """
    p_star, v_attacker = solve_attacker_lp(A)
    q_star, v_defender = solve_defender_lp(A)
    v_star = (v_attacker + v_defender) / 2.0  # Should be equal; average for numerical stability
    saddle = find_saddle_point(A)
    return {
        "p_star": p_star,         # Attacker Nash strategy
        "q_star": q_star,         # Defender Nash strategy
        "game_value": v_star,     # Minimax value
        "has_saddle_point": saddle is not None,
        "saddle_point": saddle,
        "verification": expected_payoff(A, p_star, q_star),
    }
