from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EARTH_RADIUS_KM = 6371.0088
PSO_ALGORITHM_NAME = "Particle Swarm Optimization (PSO)"
BO_ALGORITHM_NAME = "Bayesian Optimization (BO)"
CMA_ES_ALGORITHM_NAME = "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"
TR_ASCII = str.maketrans(
    {
        "\u00C7": "C",
        "\u011E": "G",
        "\u0130": "I",
        "I": "I",
        "\u00D6": "O",
        "\u015E": "S",
        "\u00DC": "U",
        "\u00E7": "c",
        "\u011F": "g",
        "\u0131": "i",
        "i": "i",
        "\u00F6": "o",
        "\u015F": "s",
        "\u00FC": "u",
    }
)

PSO_PROBLEM_LABELS = [
    "Schwefel",
    "Ackley (shifted)",
    "Rastrigin (shifted+rotated)",
    "Rosenbrock (genis)",
    "Levy",
]
CONTINUOUS_BENCHMARK_SET = tuple(PSO_PROBLEM_LABELS)
TSP_BENCHMARK_NAME = "TSP-81IL-IZMIR"
COMPARISON_SEED = 42

@dataclass
class GAConfig:
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    elitism: int
    selection_method: str
    tournament_size: int
    crossover_method: str
    mutation_operator: str
    route_update_every: int
    analytics_update_every: int
    heatmap_update_every: int
    frame_delay: float
    random_seed: int

@dataclass
class SAConfig:
    iterations: int
    initial_temperature: float
    cooling_rate: float
    min_temperature: float
    neighbor_operator: str
    two_opt_every: int
    stagnation_limit: int
    reheat_ratio: float
    route_update_every: int
    analytics_update_every: int
    moves_update_every: int
    frame_delay: float
    random_seed: int

@dataclass
class TabuConfig:
    iterations: int
    candidate_pool_size: int
    tabu_tenure: int
    aspiration_enabled: bool
    stagnation_limit: int
    kick_ratio: float
    route_update_every: int
    analytics_update_every: int
    moves_update_every: int
    frame_delay: float
    random_seed: int

@dataclass
class ACOConfig:
    ant_count: int
    iterations: int
    alpha: float
    beta: float
    evaporation_rate: float
    pheromone_constant: float
    elitist_weight: int
    candidate_k: int
    two_opt_every: int
    route_update_every: int
    analytics_update_every: int
    ants_update_every: int
    frame_delay: float
    random_seed: int

@dataclass
class PSOConfig:
    problem_name: str
    swarm_size: int
    iterations: int
    inertia_weight: float
    cognitive_coeff: float
    social_coeff: float
    velocity_clamp_ratio: float
    route_update_every: int
    analytics_update_every: int
    frame_delay: float
    random_seed: int

@dataclass
class BOConfig:
    problem_name: str
    n_initial: int
    n_iterations: int
    kernel_type: str
    acquisition_type: str
    kappa: float
    xi: float
    route_update_every: int
    analytics_update_every: int
    frame_delay: float
    random_seed: int


@dataclass
class CMAESConfig:
    problem_name: str
    population_size: int
    iterations: int
    initial_sigma: float
    sigma_decay: float
    elite_ratio: float
    route_update_every: int
    analytics_update_every: int
    frame_delay: float
    random_seed: int

@dataclass
class PSOProblem:
    name: str
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    objective: callable
    description: str
    global_min: np.ndarray
    global_min_value: float
