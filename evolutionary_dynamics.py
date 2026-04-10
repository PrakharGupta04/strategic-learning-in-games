"""
Module 3: Evolutionary Game Dynamics
======================================
Models how strategies evolve in a *population* of agents over time.

Key concepts:
  - Replicator Dynamics: frequency of strategy i grows proportional to
    how much better it does vs the population average.
    dx_i/dt = x_i * (f_i(x) - f̄(x))
    where f_i = fitness of strategy i, f̄ = average fitness

  - Evolutionary Stable Strategy (ESS): a strategy x* that, if adopted
    by the whole population, cannot be invaded by any mutant strategy.
    ESS is a refinement of Nash equilibrium.

  - Relation to Nash: Every ESS is a Nash equilibrium, but not every
    Nash equilibrium is an ESS.

Application: Cybersecurity population where agents gradually learn whether
to play "attack heavy / light" or "defend heavy / light" based on payoffs.
"""

import numpy as np
from typing import Callable


# ── fitness functions ────────────────────────────────────────────────────────

def fitness(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    f_i(x) = (A x)_i  — expected payoff of strategy i against population x.
    A: n×n symmetric payoff matrix (for single-population game)
    x: population state (mixed strategy / frequency vector)
    """
    return A @ x


def average_fitness(A: np.ndarray, x: np.ndarray) -> float:
    """f̄(x) = x^T A x — average fitness in population."""
    return float(x @ A @ x)


# ── replicator dynamics ──────────────────────────────────────────────────────

def replicator_step(A: np.ndarray, x: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    One Euler step of replicator dynamics:
      x_i(t+dt) = x_i(t) + dt * x_i(t) * (f_i(x) - f̄(x))
    Then re-normalise to stay on the simplex.
    """
    f = fitness(A, x)
    f_bar = float(x @ f)
    dx = x * (f - f_bar)
    x_new = x + dt * dx
    x_new = np.clip(x_new, 0, None)
    s = x_new.sum()
    return x_new / s if s > 0 else x


def simulate_replicator(A: np.ndarray, x0: np.ndarray,
                        T: int = 2000, dt: float = 0.01) -> dict:
    """
    Full simulation of replicator dynamics.
    Returns trajectory of population states and fitness values.

    A: symmetric payoff matrix (n×n)
    x0: initial population state (n-vector, sums to 1)
    """
    n = len(x0)
    trajectory = [x0.copy()]
    fitness_hist = []
    avg_fitness_hist = []

    x = x0.copy()
    for _ in range(T):
        f = fitness(A, x)
        f_bar = average_fitness(A, x)
        fitness_hist.append(f.copy())
        avg_fitness_hist.append(f_bar)
        x = replicator_step(A, x, dt)
        trajectory.append(x.copy())

    return {
        "trajectory": np.array(trajectory),           # shape (T+1, n)
        "fitness_history": np.array(fitness_hist),     # shape (T, n)
        "avg_fitness_history": np.array(avg_fitness_hist),  # shape (T,)
        "final_state": trajectory[-1],
        "time": np.arange(T + 1) * dt,
    }


# ── ESS detection ────────────────────────────────────────────────────────────

def is_ess(A: np.ndarray, x_star: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if x_star is an ESS using the two-condition definition:
      For all y ≠ x_star:
        (1) x_star^T A x_star > y^T A x_star  (strict NE condition), OR
        (2) x_star^T A x_star = y^T A x_star  AND  x_star^T A y > y^T A y

    Tests on a grid of candidate mutant strategies.
    """
    n = len(x_star)
    v_star = float(x_star @ A @ x_star)

    # Test pure strategy mutants (most important candidates)
    for i in range(n):
        y = np.zeros(n)
        y[i] = 1.0
        if np.allclose(y, x_star, atol=tol):
            continue
        payoff_vs_star = float(y @ A @ x_star)
        if payoff_vs_star > v_star + tol:
            return False  # Mutant invades: not ESS
        if abs(payoff_vs_star - v_star) < tol:
            # Neutrally stable: check secondary condition
            if float(y @ A @ y) >= float(x_star @ A @ y) - tol:
                return False
    return True


def find_fixed_points(A: np.ndarray, T: int = 2000, dt: float = 0.01,
                      n_trials: int = 30, seed: int = 0) -> list:
    """
    Find fixed points of replicator dynamics by running many random initialisations.
    Returns list of unique fixed points and whether each is an ESS.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    fixed_points = []

    for _ in range(n_trials):
        x0 = rng.dirichlet(np.ones(n))
        result = simulate_replicator(A, x0, T=T, dt=dt)
        x_final = result["final_state"]

        # Check if this is a new fixed point (not already found)
        is_new = True
        for fp in fixed_points:
            if np.linalg.norm(fp["state"] - x_final) < 0.02:
                is_new = False
                break

        if is_new:
            ess = is_ess(A, x_final)
            fixed_points.append({
                "state": x_final,
                "is_ess": ess,
                "avg_fitness": average_fitness(A, x_final),
            })

    return fixed_points


# ── hawk-dove payoff matrix ──────────────────────────────────────────────────

def hawk_dove_matrix(V: float = 4.0, C: float = 6.0) -> np.ndarray:
    """
    Classic Hawk-Dove game.
    V = value of resource, C = cost of injury
    Strategies: [Hawk, Dove]

    Payoffs:
      H vs H: (V-C)/2  — fight, expected cost
      H vs D: V        — hawk wins, dove retreats
      D vs H: 0        — dove retreats
      D vs D: V/2      — share resource
    """
    return np.array([
        [(V - C) / 2,  V],
        [0,            V / 2],
    ])


def hawk_dove_ess(V: float = 4.0, C: float = 6.0) -> float:
    """
    Analytical ESS for Hawk-Dove: p* = V/C (fraction of hawks at equilibrium).
    Valid only when C > V (cost of fighting exceeds value).
    """
    if C <= V:
        return 1.0  # All hawks is ESS
    return V / C
