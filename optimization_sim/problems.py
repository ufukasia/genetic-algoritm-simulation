from __future__ import annotations

import numpy as np

from .models import PSOProblem

def schwefel_objective(x: np.ndarray) -> float:
    """Schwefel: yaniltici, global minimum merkezden cok uzakta."""
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def ackley_shifted_objective(x: np.ndarray) -> float:
    """Ackley (shifted): duz plato + keskin ibre seklinde minimum."""
    shift = np.array([2.8, -3.5])
    z = x - shift
    n = len(z)
    sum_sq = np.sum(z * z)
    sum_cos = np.sum(np.cos(2.0 * np.pi * z))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
                 - np.exp(sum_cos / n) + 20.0 + np.e)

def rastrigin_shifted_rotated_objective(x: np.ndarray) -> float:
    """Rastrigin (shifted + 30 derece rotated): cok yerel minimumlu."""
    shift = np.array([-3.2, 4.1])
    theta = np.radians(30.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    z = rot @ (x - shift)
    n = len(z)
    return float(10.0 * n + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z)))

def rosenbrock_wide_objective(x: np.ndarray) -> float:
    """Rosenbrock genis uzayda: dar, kavisli vadi."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))

def levy_objective(x: np.ndarray) -> float:
    """Levy fonksiyonu: karmasik cok-modlu."""
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)

def build_pso_problem(problem_name: str) -> PSOProblem:
    if problem_name == "Schwefel":
        lb = np.array([-500.0, -500.0])
        ub = np.array([500.0, 500.0])
        return PSOProblem(
            problem_name, lb, ub, schwefel_objective,
            (
                "f(x) = 418.9829*n - sum(xi * sin(sqrt(|xi|)))\n"
                "Global minimum: f(420.9687, 420.9687) = 0\n"
                "En yaniltici benchmark: minimum, arama uzayinin koseninde!\n"
                "Suru, merkezden cok uzaktaki global minimumu bulmak zorunda."
            ),
            np.array([420.9687, 420.9687]), 0.0,
        )

    if problem_name == "Ackley (shifted)":
        lb = np.array([-30.0, -30.0])
        ub = np.array([30.0, 30.0])
        return PSOProblem(
            problem_name, lb, ub, ackley_shifted_objective,
            (
                "f(x) = -20*exp(-0.2*sqrt(sum/n)) - exp(sum_cos/n) + 20 + e\n"
                "Global minimum: f(2.8, -3.5) = 0 (shifted)\n"
                "Duz plato + dar ibre seklinde minimum. Suru platoda kaybolabilir."
            ),
            np.array([2.8, -3.5]), 0.0,
        )

    if problem_name == "Rastrigin (shifted+rotated)":
        lb = np.array([-15.0, -15.0])
        ub = np.array([15.0, 15.0])
        return PSOProblem(
            problem_name, lb, ub, rastrigin_shifted_rotated_objective,
            (
                "f(x) = 10*n + sum(zi^2 - 10*cos(2*pi*zi)), z = R*(x-shift)\n"
                "Global minimum: f(-3.2, 4.1) = 0 (shifted + 30 derece rotated)\n"
                "50+ yerel minimum! Suru yerel minimumlara takilabilir."
            ),
            np.array([-3.2, 4.1]), 0.0,
        )

    if problem_name == "Rosenbrock (genis)":
        lb = np.array([-30.0, -30.0])
        ub = np.array([30.0, 30.0])
        return PSOProblem(
            problem_name, lb, ub, rosenbrock_wide_objective,
            (
                "f(x) = 100*(x2-x1^2)^2 + (x1-1)^2\n"
                "Global minimum: f(1, 1) = 0\n"
                "Genis uzayda dar, kavisli 'muz' vadisi.\n"
                "Suru vadiyi bulsa bile minimum noktaya yurumek cok zor."
            ),
            np.array([1.0, 1.0]), 0.0,
        )

    if problem_name == "Levy":
        lb = np.array([-10.0, -10.0])
        ub = np.array([10.0, 10.0])
        return PSOProblem(
            problem_name, lb, ub, levy_objective,
            (
                "f(x) = sin^2(pi*w1) + sum((wi-1)^2*(1+10*sin^2(pi*wi+1)))\n"
                "         + (wn-1)^2*(1+sin^2(2*pi*wn)), w = 1+(x-1)/4\n"
                "Global minimum: f(1, 1) = 0\n"
                "Karmasik, cok-modlu peyzaj. Simetrik gorunur ama yaniltici."
            ),
            np.array([1.0, 1.0]), 0.0,
        )

    raise ValueError(f"Desteklenmeyen problem: {problem_name}")
