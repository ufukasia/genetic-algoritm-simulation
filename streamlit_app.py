from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from scipy.stats import norm as scipy_norm


EARTH_RADIUS_KM = 6371.0088
PSO_ALGORITHM_NAME = "Particle Swarm Optimization (PSO)"
BO_ALGORITHM_NAME = "Bayesian Optimization (BO)"
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
class PSOProblem:
    name: str
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    objective: callable
    description: str
    global_min: np.ndarray
    global_min_value: float


PSO_PROBLEM_LABELS = [
    "Schwefel",
    "Ackley (shifted)",
    "Rastrigin (shifted+rotated)",
    "Rosenbrock (genis)",
    "Levy",
]


def normalize_city_name(value: str) -> str:
    return str(value).strip().translate(TR_ASCII).upper()


def parse_coordinate(raw_value: str) -> float:
    token = str(raw_value).strip().replace(" ", "").replace(",", ".")
    if not token:
        return np.nan

    if token.isdigit() and len(token) >= 7:
        token = f"{token[:2]}.{token[2:]}"

    value = float(token)
    if abs(value) > 180 and token.replace(".", "").isdigit():
        value = float(token.replace(".", "")) / 1_000_000
    return value


# ---------- PSO Benchmark Fonksiyonlari ----------

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


@st.cache_data(show_spinner=False)
def load_turkiye_cities(csv_path: str) -> pd.DataFrame:
    names = ["plate", "city", "lat_raw", "lon_raw"]
    encodings = ["utf-8", "cp1254", "latin1"]
    data = None
    last_error = None

    for encoding in encodings:
        try:
            data = pd.read_csv(
                csv_path,
                sep=";",
                header=None,
                names=names,
                dtype=str,
                encoding=encoding,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if data is None:
        raise RuntimeError(f"81il.csv okunamadi: {last_error}") from last_error

    data["plate"] = data["plate"].astype(int)
    data["city"] = data["city"].str.strip()
    data["lat"] = data["lat_raw"].apply(parse_coordinate)
    data["lon"] = data["lon_raw"].apply(parse_coordinate)
    data["city_key"] = data["city"].apply(normalize_city_name)
    data = data.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return data


@st.cache_data(show_spinner=False)
def build_distance_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat = np.radians(latitudes)
    lon = np.radians(longitudes)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(
        dlon / 2.0
    ) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def create_initial_population(
    population_size: int, available_cities: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    chromosomes = [rng.permutation(available_cities) for _ in range(population_size)]
    return np.asarray(chromosomes, dtype=np.int16)


def evaluate_population(
    population: np.ndarray, start_idx: int, distance_matrix: np.ndarray
) -> np.ndarray:
    start_col = np.full((population.shape[0], 1), start_idx, dtype=np.int16)
    full_route = np.hstack((start_col, population, start_col))
    return distance_matrix[full_route[:, :-1], full_route[:, 1:]].sum(axis=1)


def route_distance(chromosome: np.ndarray, start_idx: int, distance_matrix: np.ndarray) -> float:
    sequence = np.concatenate(([start_idx], chromosome, [start_idx]))
    return float(distance_matrix[sequence[:-1], sequence[1:]].sum())


def select_parent_indices(
    distances: np.ndarray,
    method: str,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pop_size = len(distances)
    if method == "Turnuva":
        picks = np.empty(pop_size, dtype=np.int32)
        for i in range(pop_size):
            candidates = rng.integers(0, pop_size, size=tournament_size)
            picks[i] = candidates[np.argmin(distances[candidates])]
        return picks

    fitness = 1.0 / (distances + 1e-9)
    probs = fitness / fitness.sum()
    return rng.choice(pop_size, size=pop_size, p=probs)


def ordered_crossover(
    parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int]]:
    n = len(parent1)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=np.int16)
    child[i : j + 1] = parent1[i : j + 1]

    used = set(child[i : j + 1].tolist())
    fill_values = [gene for gene in parent2 if gene not in used]
    fill_idx = 0
    for pos in range(n):
        if child[pos] == -1:
            child[pos] = fill_values[fill_idx]
            fill_idx += 1
    return child, (i, j)


def pmx_crossover(
    parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int]]:
    n = len(parent1)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=np.int16)
    child[i : j + 1] = parent1[i : j + 1]

    p2_positions = {int(gene): idx for idx, gene in enumerate(parent2)}
    child_values = set(child[i : j + 1].tolist())

    for pos in range(i, j + 1):
        gene = int(parent2[pos])
        if gene in child_values:
            continue

        mapped_pos = pos
        while True:
            mapped_gene = int(parent1[mapped_pos])
            mapped_pos = p2_positions[mapped_gene]
            if child[mapped_pos] == -1:
                child[mapped_pos] = gene
                break

    for idx in range(n):
        if child[idx] == -1:
            child[idx] = parent2[idx]

    return child, (i, j)


def mutate_chromosome(
    chromosome: np.ndarray,
    mutation_rate: float,
    operator: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict | None]:
    if rng.random() >= mutation_rate:
        return chromosome, None

    n = len(chromosome)
    mutated = chromosome.copy()
    i, j = sorted(rng.choice(n, size=2, replace=False))
    before = mutated.copy()

    if operator == "Swap":
        mutated[i], mutated[j] = mutated[j], mutated[i]
    elif operator == "Inversion":
        mutated[i : j + 1] = mutated[i : j + 1][::-1]
    else:
        segment = mutated[i : j + 1].copy()
        rng.shuffle(segment)
        mutated[i : j + 1] = segment

    event = {
        "operator": operator,
        "positions": f"{i}-{j}",
        "before": before[i : j + 1].tolist(),
        "after": mutated[i : j + 1].tolist(),
    }
    return mutated, event


def apply_two_opt_segment(chromosome: np.ndarray, i: int, j: int) -> np.ndarray:
    mutated = chromosome.copy()
    mutated[i : j + 1] = mutated[i : j + 1][::-1]
    return mutated


def two_opt_delta(
    chromosome: np.ndarray,
    i: int,
    j: int,
    start_idx: int,
    distance_matrix: np.ndarray,
) -> float:
    if i == j:
        return 0.0

    city_a = start_idx if i == 0 else int(chromosome[i - 1])
    city_b = int(chromosome[i])
    city_c = int(chromosome[j])
    city_d = start_idx if j == len(chromosome) - 1 else int(chromosome[j + 1])

    removed = distance_matrix[city_a, city_b] + distance_matrix[city_c, city_d]
    added = distance_matrix[city_a, city_c] + distance_matrix[city_b, city_d]
    return float(added - removed)


def propose_sa_neighbor(
    chromosome: np.ndarray,
    current_distance: float,
    operator: str,
    start_idx: int,
    distance_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, dict]:
    n = len(chromosome)
    use_operator = operator
    if operator == "2-Opt + Swap (onerilen)":
        use_operator = "2-Opt" if rng.random() < 0.75 else "Swap"

    if use_operator == "2-Opt":
        i, j = sorted(rng.choice(n, size=2, replace=False))
        candidate = apply_two_opt_segment(chromosome, int(i), int(j))
        delta = two_opt_delta(chromosome, int(i), int(j), start_idx, distance_matrix)
        event = {
            "operator": "2-Opt",
            "positions": f"{int(i)}-{int(j)}",
            "before": chromosome[int(i) : int(j) + 1].tolist(),
            "after": candidate[int(i) : int(j) + 1].tolist(),
        }
        return candidate, current_distance + delta, event

    mapped = use_operator if use_operator in {"Swap", "Inversion", "Scramble"} else "Scramble"
    candidate, event = mutate_chromosome(chromosome, mutation_rate=1.0, operator=mapped, rng=rng)
    candidate_distance = route_distance(candidate, start_idx, distance_matrix)
    return candidate, candidate_distance, event or {}


def best_two_opt_improvement(
    chromosome: np.ndarray,
    current_distance: float,
    start_idx: int,
    distance_matrix: np.ndarray,
) -> tuple[np.ndarray, float, tuple[int, int] | None]:
    n = len(chromosome)
    best_delta = 0.0
    best_pair: tuple[int, int] | None = None

    for i in range(n - 1):
        for j in range(i + 1, n):
            delta = two_opt_delta(chromosome, i, j, start_idx, distance_matrix)
            if delta < best_delta:
                best_delta = delta
                best_pair = (i, j)

    if best_pair is None:
        return chromosome, current_distance, None

    improved = apply_two_opt_segment(chromosome, best_pair[0], best_pair[1])
    return improved, current_distance + best_delta, best_pair


def sample_two_opt_pairs(
    n: int, candidate_count: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    total_pairs = n * (n - 1) // 2
    if total_pairs <= 0:
        return []

    target = max(1, min(candidate_count, total_pairs))
    if target == total_pairs:
        return [(i, j) for i in range(n - 1) for j in range(i + 1, n)]

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < target:
        i, j = sorted(rng.choice(n, size=2, replace=False))
        pairs.add((int(i), int(j)))
    return list(pairs)


def construct_aco_route(
    start_idx: int,
    available_cities: np.ndarray,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    alpha: float,
    beta: float,
    candidate_k: int,
    nearest_neighbors: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    unvisited = set(int(city) for city in available_cities.tolist())
    route: list[int] = []
    current = start_idx

    for _ in range(len(available_cities)):
        if candidate_k > 0:
            near = [city for city in nearest_neighbors[current][:candidate_k] if city in unvisited]
            candidates = near if near else list(unvisited)
        else:
            candidates = list(unvisited)

        candidate_arr = np.asarray(candidates, dtype=np.int16)
        tau = np.power(pheromone[current, candidate_arr], alpha)
        eta = np.power(heuristic[current, candidate_arr], beta)
        weights = tau * eta
        total = float(weights.sum())

        if not np.isfinite(total) or total <= 0.0:
            next_city = int(rng.choice(candidate_arr))
        else:
            probs = weights / total
            next_city = int(rng.choice(candidate_arr, p=probs))

        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return np.asarray(route, dtype=np.int16)


def add_pheromone_for_route(
    pheromone: np.ndarray,
    route: np.ndarray,
    start_idx: int,
    delta_pheromone: float,
) -> None:
    full_route = np.concatenate(([start_idx], route, [start_idx]))
    from_nodes = full_route[:-1]
    to_nodes = full_route[1:]
    pheromone[from_nodes, to_nodes] += delta_pheromone
    pheromone[to_nodes, from_nodes] += delta_pheromone


def crossover_pair(
    parent1: np.ndarray,
    parent2: np.ndarray,
    method: str,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() >= crossover_rate:
        return parent1.copy(), parent2.copy()

    if method == "PMX":
        child1, _ = pmx_crossover(parent1, parent2, rng)
        child2, _ = pmx_crossover(parent2, parent1, rng)
        return child1, child2

    child1, _ = ordered_crossover(parent1, parent2, rng)
    child2, _ = ordered_crossover(parent2, parent1, rng)
    return child1, child2


def population_diversity(
    population: np.ndarray, best_chromosome: np.ndarray
) -> tuple[float, float]:
    unique_ratio = np.unique(population, axis=0).shape[0] / population.shape[0]
    hamming_to_best = float(np.mean(np.mean(population != best_chromosome, axis=1)))
    return unique_ratio, hamming_to_best


def build_route_figure(
    route_idx: np.ndarray,
    cities: pd.DataFrame,
    generation: int,
    distance_km: float,
    step_label: str = "Nesil",
) -> go.Figure:
    lat = cities.loc[route_idx, "lat"].to_numpy()
    lon = cities.loc[route_idx, "lon"].to_numpy()
    names = cities.loc[route_idx, "city"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="lines+markers",
            line=dict(width=2, color="#d62828"),
            marker=dict(size=5, color="#003049"),
            text=names,
            hovertemplate="%{text}<extra></extra>",
            name="Tur",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[lat[0]],
            lon=[lon[0]],
            mode="markers+text",
            marker=dict(size=12, color="#2a9d8f"),
            text=[f"Baslangic: {names[0]}"],
            textposition="top center",
            name="Baslangic",
        )
    )
    fig.update_layout(
        title=f"{step_label} {generation} | En iyi tur: {distance_km:,.1f} km",
        height=460,
        margin=dict(l=0, r=0, t=55, b=0),
        geo=dict(
            projection_type="mercator",
            showland=True,
            landcolor="#f1f5f9",
            showcountries=True,
            countrycolor="#94a3b8",
            center=dict(lat=39.0, lon=35.0),
            lataxis=dict(range=[35.2, 42.8]),
            lonaxis=dict(range=[25.5, 45.5]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_fitness_figure(best_hist: list[float], avg_hist: list[float]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Nesil": np.arange(1, len(best_hist) + 1),
            "En iyi (km)": best_hist,
            "Ortalama (km)": avg_hist,
        }
    )
    fig = px.line(
        frame,
        x="Nesil",
        y=["En iyi (km)", "Ortalama (km)"],
        labels={"value": "Mesafe (km)", "variable": "Olcum"},
    )
    fig.update_layout(height=290, margin=dict(l=0, r=0, t=35, b=0), legend_title_text="")
    return fig


def build_diversity_figure(unique_hist: list[float], hamming_hist: list[float]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Nesil": np.arange(1, len(unique_hist) + 1),
            "Benzersiz birey orani": unique_hist,
            "Best'e ortalama Hamming": hamming_hist,
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["Nesil"],
            y=frame["Benzersiz birey orani"],
            mode="lines",
            name="Benzersiz oran",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["Nesil"],
            y=frame["Best'e ortalama Hamming"],
            mode="lines",
            name="Hamming",
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=0, r=0, t=35, b=0),
        yaxis=dict(title="Benzersiz oran", range=[0, 1]),
        yaxis2=dict(title="Hamming", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_sa_progress_figure(best_hist: list[float], current_hist: list[float]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Iterasyon": np.arange(1, len(best_hist) + 1),
            "En iyi (km)": best_hist,
            "Mevcut (km)": current_hist,
        }
    )
    fig = px.line(
        frame,
        x="Iterasyon",
        y=["En iyi (km)", "Mevcut (km)"],
        labels={"value": "Mesafe (km)", "variable": "Olcum"},
    )
    fig.update_layout(height=290, margin=dict(l=0, r=0, t=35, b=0), legend_title_text="")
    return fig


def build_sa_temperature_figure(
    temperature_hist: list[float], acceptance_hist: list[float]
) -> go.Figure:
    iterations = np.arange(1, len(temperature_hist) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=temperature_hist,
            mode="lines",
            name="Sicaklik",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=acceptance_hist,
            mode="lines",
            name="Kabul orani",
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=0, r=0, t=35, b=0),
        yaxis=dict(title="Sicaklik"),
        yaxis2=dict(title="Kabul orani", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_tabu_status_figure(
    tabu_size_hist: list[float], aspiration_hist: list[float], stagnation_hist: list[float]
) -> go.Figure:
    iterations = np.arange(1, len(tabu_size_hist) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=tabu_size_hist,
            mode="lines",
            name="Tabu liste boyutu",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=stagnation_hist,
            mode="lines",
            name="Stagnasyon",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=aspiration_hist,
            mode="lines",
            name="Aspiration orani",
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=0, r=0, t=35, b=0),
        yaxis=dict(title="Tabu/Stagnasyon"),
        yaxis2=dict(title="Aspiration", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_sa_moves_figure(move_events: deque) -> go.Figure:
    events = list(move_events)
    if not events:
        fig = go.Figure()
        fig.update_layout(
            title="Son hamle degisimleri",
            height=380,
            margin=dict(l=0, r=0, t=45, b=0),
            xaxis_title="Iterasyon",
            yaxis_title="delta km",
        )
        fig.add_annotation(
            text="Henuz hamle verisi yok",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    frame = pd.DataFrame(events)
    frame["durum"] = frame["accepted"].map({True: "Kabul", False: "Red"})
    fig = px.bar(
        frame,
        x="iteration",
        y="delta_km",
        color="durum",
        color_discrete_map={"Kabul": "#2a9d8f", "Red": "#d62828"},
        hover_data=["temperature", "accepted_worse", "operator", "positions"],
    )
    fig.update_layout(
        title=f"Son {len(events)} hamlenin delta km dagilimi",
        height=380,
        margin=dict(l=0, r=0, t=45, b=0),
        xaxis_title="Iterasyon",
        yaxis_title="delta km",
        legend_title_text="",
    )
    return fig


def build_aco_progress_figure(best_hist: list[float], iter_best_hist: list[float]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Iterasyon": np.arange(1, len(best_hist) + 1),
            "Global en iyi (km)": best_hist,
            "Iterasyon en iyi (km)": iter_best_hist,
        }
    )
    fig = px.line(
        frame,
        x="Iterasyon",
        y=["Global en iyi (km)", "Iterasyon en iyi (km)"],
        labels={"value": "Mesafe (km)", "variable": "Olcum"},
    )
    fig.update_layout(height=290, margin=dict(l=0, r=0, t=35, b=0), legend_title_text="")
    return fig


def build_aco_pheromone_figure(
    pheromone_mean_hist: list[float], pheromone_max_hist: list[float]
) -> go.Figure:
    iterations = np.arange(1, len(pheromone_mean_hist) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=pheromone_mean_hist,
            mode="lines",
            name="Ortalama feromon",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=pheromone_max_hist,
            mode="lines",
            name="Maks feromon",
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=0, r=0, t=35, b=0),
        yaxis=dict(title="Ortalama"),
        yaxis2=dict(title="Maks", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_aco_ant_distance_figure(ant_distances: np.ndarray) -> go.Figure:
    if ant_distances.size == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Karinca mesafe dagilimi",
            height=380,
            margin=dict(l=0, r=0, t=45, b=0),
        )
        return fig

    sorted_distances = np.sort(ant_distances.astype(float))
    sample = sorted_distances[: min(40, len(sorted_distances))]
    frame = pd.DataFrame({"Karinca": np.arange(1, len(sample) + 1), "Tur mesafesi (km)": sample})
    fig = px.bar(frame, x="Karinca", y="Tur mesafesi (km)")
    fig.update_layout(
        title=f"Bu iterasyondaki en iyi {len(sample)} karinca",
        height=380,
        margin=dict(l=0, r=0, t=45, b=0),
        xaxis_title="Karinca sirasi (iyi -> kotu)",
    )
    return fig


def build_aco_swarm_figure(
    cities: pd.DataFrame,
    start_idx: int,
    best_route_idx: np.ndarray,
    pheromone: np.ndarray,
    ant_routes: list[np.ndarray],
    ant_distances: np.ndarray,
    iteration: int,
    best_distance: float,
    max_pheromone_edges: int = 260,
    max_swarm_routes: int = 14,
) -> go.Figure:
    lat_all = cities["lat"].to_numpy()
    lon_all = cities["lon"].to_numpy()
    names_all = cities["city"].to_numpy()
    n_cities = len(cities)

    fig = go.Figure()

    # Draw strongest pheromone edges as a layered network.
    upper_i, upper_j = np.triu_indices(n_cities, k=1)
    upper_values = pheromone[upper_i, upper_j]
    edge_count = min(max_pheromone_edges, len(upper_values))
    if edge_count > 0:
        pick = np.argpartition(upper_values, -edge_count)[-edge_count:]
        edge_i = upper_i[pick]
        edge_j = upper_j[pick]
        edge_v = upper_values[pick]

        v_min = float(np.min(edge_v))
        v_max = float(np.max(edge_v))
        span = max(v_max - v_min, 1e-12)
        edge_norm = (edge_v - v_min) / span

        layer_specs = [
            ("Feromon dusuk", edge_norm < 0.33, "rgba(16,185,129,0.12)", 0.7),
            (
                "Feromon orta",
                (edge_norm >= 0.33) & (edge_norm < 0.66),
                "rgba(16,185,129,0.24)",
                1.5,
            ),
            ("Feromon yuksek", edge_norm >= 0.66, "rgba(5,150,105,0.42)", 2.8),
        ]

        for layer_name, mask, color, width in layer_specs:
            if not np.any(mask):
                continue

            lat_segments: list[float | None] = []
            lon_segments: list[float | None] = []
            text_segments: list[str | None] = []
            layer_i = edge_i[mask]
            layer_j = edge_j[mask]
            layer_v = edge_v[mask]

            for a, b, value in zip(layer_i, layer_j, layer_v):
                a = int(a)
                b = int(b)
                lat_segments.extend([float(lat_all[a]), float(lat_all[b]), None])
                lon_segments.extend([float(lon_all[a]), float(lon_all[b]), None])
                label = f"{names_all[a]} <-> {names_all[b]}<br>Feromon: {float(value):.5f}"
                text_segments.extend([label, label, None])

            fig.add_trace(
                go.Scattergeo(
                    lat=lat_segments,
                    lon=lon_segments,
                    text=text_segments,
                    mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="text",
                    name=layer_name,
                    showlegend=True,
                )
            )

    # Show ant swarm as semi-transparent route bundle (best ants of this iteration).
    if ant_distances.size > 0 and ant_routes:
        top_ant_ids = np.argsort(ant_distances)[: min(max_swarm_routes, len(ant_routes))]
        for rank, ant_id in enumerate(top_ant_ids):
            ant_id = int(ant_id)
            full_route = np.concatenate(([start_idx], ant_routes[ant_id], [start_idx]))
            lat_route = cities.iloc[full_route]["lat"].to_numpy()
            lon_route = cities.iloc[full_route]["lon"].to_numpy()
            fig.add_trace(
                go.Scattergeo(
                    lat=lat_route,
                    lon=lon_route,
                    mode="lines",
                    line=dict(width=1.0, color="rgba(59,130,246,0.10)"),
                    hoverinfo="skip",
                    name="Karinca surusu",
                    showlegend=(rank == 0),
                )
            )

    # Overlay global best route for readability.
    best_route_idx = np.asarray(best_route_idx, dtype=np.int32)
    lat_best = cities.iloc[best_route_idx]["lat"].to_numpy()
    lon_best = cities.iloc[best_route_idx]["lon"].to_numpy()
    name_best = cities.iloc[best_route_idx]["city"].tolist()
    fig.add_trace(
        go.Scattergeo(
            lat=lat_best,
            lon=lon_best,
            mode="lines+markers",
            line=dict(width=3.2, color="#ef4444"),
            marker=dict(size=5, color="#1e3a8a"),
            text=name_best,
            hovertemplate="%{text}<extra></extra>",
            name="Global en iyi tur",
        )
    )

    node_strength = pheromone.sum(axis=1)
    node_min = float(np.min(node_strength))
    node_max = float(np.max(node_strength))
    node_span = max(node_max - node_min, 1e-12)
    node_size = 4.0 + 8.0 * (node_strength - node_min) / node_span
    fig.add_trace(
        go.Scattergeo(
            lat=lat_all,
            lon=lon_all,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_strength,
                colorscale="YlGn",
                opacity=0.75,
                line=dict(width=0.5, color="rgba(15,23,42,0.5)"),
                colorbar=dict(title="Dugum feromon gucu"),
            ),
            text=names_all,
            hovertemplate="%{text}<br>Dugum gucu: %{marker.color:.5f}<extra></extra>",
            name="Feromon dugum yogunlugu",
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lat=[float(lat_all[start_idx])],
            lon=[float(lon_all[start_idx])],
            mode="markers+text",
            marker=dict(size=12, color="#22c55e"),
            text=[f"Baslangic: {names_all[start_idx]}"],
            textposition="top center",
            name="Baslangic",
        )
    )

    fig.update_layout(
        title=f"ACO Swarm | Iterasyon {iteration} | En iyi tur: {best_distance:,.1f} km",
        height=460,
        margin=dict(l=0, r=0, t=55, b=0),
        geo=dict(
            projection_type="mercator",
            showland=True,
            landcolor="#eef2ff",
            showcountries=True,
            countrycolor="#94a3b8",
            center=dict(lat=39.0, lon=35.0),
            lataxis=dict(range=[35.2, 42.8]),
            lonaxis=dict(range=[25.5, 45.5]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_aco_hyperspace_figure(
    cities: pd.DataFrame,
    start_idx: int,
    best_route_idx: np.ndarray,
    pheromone: np.ndarray,
    ant_routes: list[np.ndarray],
    ant_distances: np.ndarray,
    iteration: int,
    best_distance: float,
    max_pheromone_edges: int = 240,
    max_swarm_routes: int = 12,
) -> go.Figure:
    lat_all = cities["lat"].to_numpy(dtype=np.float64)
    lon_all = cities["lon"].to_numpy(dtype=np.float64)
    names_all = cities["city"].to_numpy()
    node_strength = pheromone.sum(axis=1).astype(np.float64)
    strength_min = float(np.min(node_strength))
    strength_max = float(np.max(node_strength))
    strength_span = max(strength_max - strength_min, 1e-12)
    z_all = 5.0 + 95.0 * (node_strength - strength_min) / strength_span

    fig = go.Figure()
    n_cities = len(cities)
    upper_i, upper_j = np.triu_indices(n_cities, k=1)
    upper_values = pheromone[upper_i, upper_j]
    edge_count = min(max_pheromone_edges, len(upper_values))
    if edge_count > 0:
        pick = np.argpartition(upper_values, -edge_count)[-edge_count:]
        edge_i = upper_i[pick]
        edge_j = upper_j[pick]
        edge_v = upper_values[pick]
        v_min = float(np.min(edge_v))
        v_max = float(np.max(edge_v))
        v_span = max(v_max - v_min, 1e-12)
        edge_norm = (edge_v - v_min) / v_span

        layers = [
            ("Feromon dusuk", edge_norm < 0.33, "rgba(34,197,94,0.12)", 1.0),
            ("Feromon orta", (edge_norm >= 0.33) & (edge_norm < 0.66), "rgba(34,197,94,0.24)", 1.9),
            ("Feromon yuksek", edge_norm >= 0.66, "rgba(22,163,74,0.42)", 3.2),
        ]
        for name, mask, color, width in layers:
            if not np.any(mask):
                continue
            xs: list[float | None] = []
            ys: list[float | None] = []
            zs: list[float | None] = []
            texts: list[str | None] = []
            for a, b, value in zip(edge_i[mask], edge_j[mask], edge_v[mask]):
                a = int(a)
                b = int(b)
                xs.extend([float(lon_all[a]), float(lon_all[b]), None])
                ys.extend([float(lat_all[a]), float(lat_all[b]), None])
                zs.extend([float(z_all[a]), float(z_all[b]), None])
                t = f"{names_all[a]} <-> {names_all[b]}<br>Feromon: {float(value):.5f}"
                texts.extend([t, t, None])
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=color, width=width),
                    text=texts,
                    hoverinfo="text",
                    name=name,
                    showlegend=True,
                )
            )

    if ant_distances.size > 0 and ant_routes:
        top_ant_ids = np.argsort(ant_distances)[: min(max_swarm_routes, len(ant_routes))]
        for rank, ant_id in enumerate(top_ant_ids):
            ant_id = int(ant_id)
            full_route = np.concatenate(([start_idx], ant_routes[ant_id], [start_idx]))
            fig.add_trace(
                go.Scatter3d(
                    x=lon_all[full_route],
                    y=lat_all[full_route],
                    z=z_all[full_route],
                    mode="lines",
                    line=dict(width=1.2, color="rgba(59,130,246,0.18)"),
                    hoverinfo="skip",
                    name="Karinca surusu",
                    showlegend=(rank == 0),
                )
            )

    best_route_idx = np.asarray(best_route_idx, dtype=np.int32)
    fig.add_trace(
        go.Scatter3d(
            x=lon_all[best_route_idx],
            y=lat_all[best_route_idx],
            z=z_all[best_route_idx],
            mode="lines+markers",
            line=dict(width=5, color="#ef4444"),
            marker=dict(size=4, color="#1e3a8a"),
            text=names_all[best_route_idx],
            hovertemplate="%{text}<extra></extra>",
            name="Global en iyi tur",
        )
    )

    marker_size = 3.5 + 7.0 * (node_strength - strength_min) / strength_span
    fig.add_trace(
        go.Scatter3d(
            x=lon_all,
            y=lat_all,
            z=z_all,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=node_strength,
                colorscale="YlGn",
                opacity=0.84,
                colorbar=dict(title="Dugum feromon gucu"),
            ),
            text=names_all,
            hovertemplate="%{text}<br>Dugum gucu: %{marker.color:.5f}<extra></extra>",
            name="Feromon dugumleri",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[float(lon_all[start_idx])],
            y=[float(lat_all[start_idx])],
            z=[float(z_all[start_idx])],
            mode="markers+text",
            marker=dict(size=10, color="#22c55e"),
            text=[f"Baslangic: {names_all[start_idx]}"],
            textposition="top center",
            name="Baslangic",
        )
    )

    fig.update_layout(
        title=f"ACO Hyperspace | Iterasyon {iteration} | En iyi tur: {best_distance:,.1f} km",
        height=500,
        margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            xaxis_title="Boylam",
            yaxis_title="Enlem",
            zaxis_title="Feromon yuksekligi",
            bgcolor="rgba(241,245,249,0.65)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_population_heatmap(
    population: np.ndarray,
    distances: np.ndarray,
    cities: pd.DataFrame,
    rows: int = 25,
    cols: int = 45,
) -> go.Figure:
    top_ids = np.argsort(distances)[: min(rows, population.shape[0])]
    top = population[top_ids, : min(cols, population.shape[1])]
    city_labels = cities["city"].to_numpy()
    hover = np.vectorize(lambda idx: city_labels[int(idx)])(top)

    fig = go.Figure(
        data=go.Heatmap(
            z=top,
            customdata=hover,
            colorscale="Turbo",
            hovertemplate="Birey: %{y}<br>Gen pozisyonu: %{x}<br>Sehir: %{customdata}<extra></extra>",
            colorbar=dict(title="Sehir ID"),
        )
    )
    fig.update_layout(
        title=f"En iyi {top.shape[0]} bireyin ilk {top.shape[1]} geni",
        height=380,
        margin=dict(l=0, r=0, t=45, b=0),
        xaxis_title="Gen pozisyonu",
        yaxis_title="Birey sirasi (iyi -> kotu)",
    )
    return fig


def ids_to_city_names(ids: list[int], cities: pd.DataFrame, max_items: int = 5) -> str:
    if not ids:
        return "-"
    names = [cities.iloc[int(i)]["city"] for i in ids[:max_items]]
    tail = "..." if len(ids) > max_items else ""
    return ", ".join(names) + tail


def build_live_info_text(
    generation: int,
    config: GAConfig,
    best_distance: float,
    first_best: float,
    mutation_counter: int,
    crossover_counter: int,
    best_route: np.ndarray,
    mutation_events: deque,
    cities: pd.DataFrame,
) -> str:
    improvement = ((first_best - best_distance) / first_best) * 100 if generation > 1 else 0.0
    route_names = [cities.iloc[int(idx)]["city"] for idx in best_route[:16]]
    route_preview = " -> ".join(route_names)
    if len(best_route) > 16:
        route_preview += " -> ..."

    lines = [
        f"Nesil: {generation}/{config.generations}",
        f"En iyi tur: {best_distance:,.1f} km",
        f"Iyilesme: %{improvement:.2f}",
        f"Mutasyon: {mutation_counter}",
        f"Crossover: {crossover_counter}",
        "",
        "Rota onizleme:",
        route_preview,
        "",
        "Son mutasyonlar:",
    ]

    if not mutation_events:
        lines.append("(Mutasyon olayi henuz yok)")
        return "\n".join(lines)

    for event in list(mutation_events)[::-1]:
        before = ids_to_city_names(event["before"], cities, max_items=4)
        after = ids_to_city_names(event["after"], cities, max_items=4)
        lines.append(
            f"N{event['generation']} | B{event['child_id']} | {event['operator']} {event['positions']} | d={event['delta_km']:+.1f} km"
        )
        lines.append(f"  once: {before}")
        lines.append(f"  sonra: {after}")

    return "\n".join(lines)


def build_sa_live_info_text(
    iteration: int,
    config: SAConfig,
    temperature: float,
    best_distance: float,
    current_distance: float,
    first_best: float,
    accepted_counter: int,
    worse_accepted_counter: int,
    local_search_counter: int,
    reheat_counter: int,
    stagnation_counter: int,
    best_route: np.ndarray,
    move_events: deque,
    cities: pd.DataFrame,
) -> str:
    improvement = ((first_best - best_distance) / first_best) * 100 if iteration > 1 else 0.0
    route_names = [cities.iloc[int(idx)]["city"] for idx in best_route[:16]]
    route_preview = " -> ".join(route_names)
    if len(best_route) > 16:
        route_preview += " -> ..."

    acceptance_ratio = accepted_counter / iteration
    worse_ratio = worse_accepted_counter / iteration
    lines = [
        f"Iterasyon: {iteration}/{config.iterations}",
        f"Sicaklik: {temperature:.6f}",
        f"En iyi tur: {best_distance:,.1f} km",
        f"Mevcut tur: {current_distance:,.1f} km",
        f"Iyilesme: %{improvement:.2f}",
        f"Kabul edilen hamle: {accepted_counter} (%{acceptance_ratio * 100:.2f})",
        f"Kotu ama kabul: {worse_accepted_counter} (%{worse_ratio * 100:.2f})",
        f"2-Opt yerel iyilestirme: {local_search_counter}",
        f"Reheat: {reheat_counter} | Stagnasyon sayaci: {stagnation_counter}",
        "",
        "Rota onizleme:",
        route_preview,
        "",
        "Son hamleler:",
    ]

    if not move_events:
        lines.append("(Hamle kaydi henuz yok)")
        return "\n".join(lines)

    for event in list(move_events)[::-1]:
        before = ids_to_city_names(event["before"], cities, max_items=4)
        after = ids_to_city_names(event["after"], cities, max_items=4)
        decision = "kabul" if event["accepted"] else "red"
        lines.append(
            f"I{event['iteration']} | {event['operator']} {event['positions']} | d={event['delta_km']:+.1f} km | T={event['temperature']:.4f} | {decision}"
        )
        lines.append(f"  once: {before}")
        lines.append(f"  sonra: {after}")

    return "\n".join(lines)


def build_tabu_live_info_text(
    iteration: int,
    config: TabuConfig,
    best_distance: float,
    current_distance: float,
    first_best: float,
    tabu_size: int,
    stagnation_counter: int,
    aspiration_counter: int,
    diversification_counter: int,
    best_route: np.ndarray,
    move_events: deque,
    cities: pd.DataFrame,
) -> str:
    improvement = ((first_best - best_distance) / first_best) * 100 if iteration > 1 else 0.0
    route_names = [cities.iloc[int(idx)]["city"] for idx in best_route[:16]]
    route_preview = " -> ".join(route_names)
    if len(best_route) > 16:
        route_preview += " -> ..."

    aspiration_ratio = aspiration_counter / iteration
    lines = [
        f"Iterasyon: {iteration}/{config.iterations}",
        f"En iyi tur: {best_distance:,.1f} km",
        f"Mevcut tur: {current_distance:,.1f} km",
        f"Iyilesme: %{improvement:.2f}",
        f"Tabu listesi: {tabu_size} | Tenure: {config.tabu_tenure}",
        f"Aspiration kullanimi: {aspiration_counter} (%{aspiration_ratio * 100:.2f})",
        f"Stagnasyon sayaci: {stagnation_counter} | Cesitlilik-kick: {diversification_counter}",
        "",
        "Rota onizleme:",
        route_preview,
        "",
        "Son hamleler:",
    ]

    if not move_events:
        lines.append("(Hamle kaydi henuz yok)")
        return "\n".join(lines)

    for event in list(move_events)[::-1]:
        before = ids_to_city_names(event["before"], cities, max_items=4)
        after = ids_to_city_names(event["after"], cities, max_items=4)
        tabu_state = "tabu" if event.get("was_tabu", False) else "-"
        aspiration_state = "asp" if event.get("aspiration", False) else "-"
        lines.append(
            f"I{event['iteration']} | {event['operator']} {event['positions']} | d={event['delta_km']:+.1f} km | {tabu_state}/{aspiration_state}"
        )
        lines.append(f"  once: {before}")
        lines.append(f"  sonra: {after}")

    return "\n".join(lines)


def build_aco_live_info_text(
    iteration: int,
    config: ACOConfig,
    best_distance: float,
    iteration_best_distance: float,
    first_best: float,
    pheromone_mean: float,
    pheromone_max: float,
    local_search_counter: int,
    best_route: np.ndarray,
    iteration_events: deque,
    cities: pd.DataFrame,
) -> str:
    improvement = ((first_best - best_distance) / first_best) * 100 if iteration > 1 else 0.0
    route_names = [cities.iloc[int(idx)]["city"] for idx in best_route[:16]]
    route_preview = " -> ".join(route_names)
    if len(best_route) > 16:
        route_preview += " -> ..."

    lines = [
        f"Iterasyon: {iteration}/{config.iterations}",
        f"Global en iyi tur: {best_distance:,.1f} km",
        f"Iterasyon en iyi tur: {iteration_best_distance:,.1f} km",
        f"Iyilesme: %{improvement:.2f}",
        f"Karinca sayisi: {config.ant_count}",
        f"Buharlasma: {config.evaporation_rate:.3f}",
        f"2-Opt iyilestirme sayisi: {local_search_counter}",
        f"Feromon ortalama: {pheromone_mean:.6f} | maks: {pheromone_max:.6f}",
        "",
        "Rota onizleme:",
        route_preview,
        "",
        "Son iterasyonlar:",
    ]

    if not iteration_events:
        lines.append("(Iterasyon ozeti henuz yok)")
        return "\n".join(lines)

    for event in list(iteration_events)[::-1]:
        local = "2-opt" if event["local_search"] else "-"
        lines.append(
            f"I{event['iteration']} | iter_best={event['iteration_best']:,.1f} | global={event['global_best']:,.1f} | pher={event['pheromone_mean']:.5f}/{event['pheromone_max']:.5f} | {local}"
        )

    return "\n".join(lines)


def build_pso_3d_surface_figure(
    problem: PSOProblem,
    positions: np.ndarray,
    fitness: np.ndarray,
    velocities: np.ndarray,
    pbest_positions: np.ndarray,
    pbest_fitness: np.ndarray,
    gbest_position: np.ndarray,
    gbest_value: float,
    gbest_history: list[np.ndarray],
    iteration: int,
    grid_resolution: int = 80,
) -> go.Figure:
    """3D yuzey uzerinde parcaciklari, hiz vektorlerini, personal/global best gorsellestirmesi."""
    lb = problem.lower_bounds
    ub = problem.upper_bounds

    x_grid = np.linspace(lb[0], ub[0], grid_resolution)
    y_grid = np.linspace(lb[1], ub[1], grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            Z[i, j] = problem.objective(np.array([X[i, j], Y[i, j]]))

    z_cap = np.percentile(Z, 97)
    Z_capped = np.clip(Z, None, z_cap)

    fig = go.Figure()

    # ---- 1. Fonksiyon yuzeyi (yari seffaf) ----
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z_capped,
            colorscale="Viridis",
            opacity=0.45,
            showscale=False,
            name="f(x1, x2) yuzeyi",
            hoverinfo="skip",
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
            ),
        )
    )

    # ---- 2. Parcaciklar (yuzey uzerinde toplar) ----
    p_z = np.clip(fitness, None, z_cap)
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=p_z,
            mode="markers",
            marker=dict(
                size=7,
                color=fitness,
                colorscale="Plasma",
                cmin=float(np.min(fitness)),
                cmax=float(np.percentile(fitness, 90)),
                opacity=0.95,
                line=dict(width=0.6, color="rgba(255,255,255,0.7)"),
                colorbar=dict(title="f(x)", len=0.5, y=0.75),
            ),
            text=[
                f"Parcacik {i}<br>x1={positions[i, 0]:.4f}<br>x2={positions[i, 1]:.4f}<br>f={fitness[i]:.6f}"
                for i in range(len(positions))
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Parcaciklar (suru)",
        )
    )

    # ---- 3. Hiz vektorleri (oklar) ----
    vel_norms = np.linalg.norm(velocities, axis=1)
    max_vel = float(np.max(vel_norms)) if float(np.max(vel_norms)) > 1e-9 else 1.0
    arrow_scale = 0.12 * (ub[0] - lb[0])
    vel_normalized = velocities / (max_vel + 1e-12) * arrow_scale

    arrow_tips_x = positions[:, 0] + vel_normalized[:, 0]
    arrow_tips_y = positions[:, 1] + vel_normalized[:, 1]
    tip_z = np.array([
        problem.objective(np.array([arrow_tips_x[i], arrow_tips_y[i]]))
        for i in range(len(positions))
    ])
    tip_z = np.clip(tip_z, None, z_cap)

    for i in range(len(positions)):
        fig.add_trace(
            go.Scatter3d(
                x=[positions[i, 0], arrow_tips_x[i]],
                y=[positions[i, 1], arrow_tips_y[i]],
                z=[p_z[i], tip_z[i]],
                mode="lines",
                line=dict(width=3, color="rgba(255,165,0,0.7)"),
                showlegend=i == 0,
                name="Hiz vektoru" if i == 0 else "",
                hoverinfo="skip",
            )
        )

    # ---- 4. Personal best ----
    pb_z = np.clip(pbest_fitness, None, z_cap)
    fig.add_trace(
        go.Scatter3d(
            x=pbest_positions[:, 0],
            y=pbest_positions[:, 1],
            z=pb_z,
            mode="markers",
            marker=dict(
                size=4,
                color="rgba(0,200,255,0.5)",
                symbol="diamond",
                line=dict(width=0.3, color="rgba(0,150,200,0.6)"),
            ),
            text=[
                f"pBest {i}<br>x1={pbest_positions[i, 0]:.4f}<br>x2={pbest_positions[i, 1]:.4f}<br>f={pbest_fitness[i]:.6f}"
                for i in range(len(pbest_positions))
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Personal best (pBest)",
        )
    )

    # ---- 5. Global best yolu (tarihce) ----
    if len(gbest_history) > 1:
        hist_arr = np.array(gbest_history)
        hist_z = np.array([problem.objective(p) for p in hist_arr])
        hist_z = np.clip(hist_z, None, z_cap)
        fig.add_trace(
            go.Scatter3d(
                x=hist_arr[:, 0],
                y=hist_arr[:, 1],
                z=hist_z,
                mode="lines+markers",
                line=dict(width=5, color="rgba(239,68,68,0.65)"),
                marker=dict(size=2, color="rgba(239,68,68,0.4)"),
                name="Global best yolu",
                hoverinfo="skip",
            )
        )

    # ---- 6. Global best noktasi ----
    gb_z = min(float(gbest_value), z_cap)
    fig.add_trace(
        go.Scatter3d(
            x=[gbest_position[0]],
            y=[gbest_position[1]],
            z=[gb_z],
            mode="markers+text",
            marker=dict(
                size=14, symbol="diamond", color="#ef4444",
                line=dict(width=2, color="#ffffff"),
            ),
            text=[f"gBest f={gbest_value:.6f}"],
            textposition="top center",
            textfont=dict(size=11, color="#ef4444"),
            hovertemplate=f"Global Best<br>x1={gbest_position[0]:.4f}<br>x2={gbest_position[1]:.4f}<br>f={gbest_value:.8f}<extra></extra>",
            name="Global best (gBest)",
        )
    )

    # ---- 7. Gercek global minimum ----
    gm = problem.global_min
    gm_z = min(float(problem.global_min_value), z_cap)
    fig.add_trace(
        go.Scatter3d(
            x=[gm[0]], y=[gm[1]], z=[gm_z],
            mode="markers+text",
            marker=dict(size=12, symbol="cross", color="#22c55e", line=dict(width=2, color="#166534")),
            text=[f"Gercek min ({gm[0]:.1f},{gm[1]:.1f})"],
            textposition="bottom center",
            textfont=dict(size=10, color="#166534"),
            hovertemplate=f"Gercek global minimum<br>x1={gm[0]:.4f}<br>x2={gm[1]:.4f}<br>f={problem.global_min_value:.8f}<extra></extra>",
            name="Gercek minimum",
        )
    )

    # ---- Layout ----
    dist_to_min = np.linalg.norm(gbest_position - problem.global_min)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>PSO 3D Gorsellestime</b> | {problem.name} | Iterasyon {iteration}"
                f"<br><span style='font-size:12px;color:#64748b'>"
                f"gBest={gbest_value:.6f} | Min'e uzaklik={dist_to_min:.4f} | Parcacik={len(positions)}</span>"
            ),
            font=dict(size=15),
        ),
        height=620,
        margin=dict(l=0, r=0, t=80, b=0),
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="f(x1, x2)",
            bgcolor="rgba(15,23,42,0.03)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.05, x=0.0,
            font=dict(size=10),
        ),
    )
    return fig


def build_pso_contour_figure(
    problem: PSOProblem,
    positions: np.ndarray,
    fitness: np.ndarray,
    velocities: np.ndarray,
    pbest_positions: np.ndarray,
    gbest_position: np.ndarray,
    gbest_value: float,
    gbest_history: list[np.ndarray],
    iteration: int,
    grid_resolution: int = 100,
) -> go.Figure:
    """Kus bakisi 2D kontur haritasi + parcaciklar + hiz oklari."""
    lb = problem.lower_bounds
    ub = problem.upper_bounds

    x_grid = np.linspace(lb[0], ub[0], grid_resolution)
    y_grid = np.linspace(lb[1], ub[1], grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            Z[i, j] = problem.objective(np.array([X[i, j], Y[i, j]]))

    z_cap = np.percentile(Z, 95)
    Z_capped = np.clip(Z, None, z_cap)

    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=x_grid, y=y_grid, z=Z_capped,
            colorscale="Viridis",
            contours=dict(coloring="heatmap"),
            showscale=True,
            colorbar=dict(title="f(x)", len=0.8),
            opacity=0.7,
            name="Kontur",
            hoverinfo="skip",
        )
    )

    # Hiz oklari (quiver benzeri)
    vel_norms = np.linalg.norm(velocities, axis=1)
    max_vel = float(np.max(vel_norms)) if float(np.max(vel_norms)) > 1e-9 else 1.0
    arrow_scale = 0.06 * (ub[0] - lb[0])

    for i in range(len(positions)):
        vn = vel_norms[i] / (max_vel + 1e-12)
        dx = velocities[i, 0] / (max_vel + 1e-12) * arrow_scale
        dy = velocities[i, 1] / (max_vel + 1e-12) * arrow_scale
        fig.add_annotation(
            x=positions[i, 0] + dx,
            y=positions[i, 1] + dy,
            ax=positions[i, 0],
            ay=positions[i, 1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.2, arrowwidth=1.5,
            arrowcolor=f"rgba(255,165,0,{0.3 + 0.7 * vn:.2f})",
        )

    # Personal best
    fig.add_trace(
        go.Scatter(
            x=pbest_positions[:, 0], y=pbest_positions[:, 1],
            mode="markers",
            marker=dict(size=5, color="rgba(0,200,255,0.4)", symbol="diamond"),
            name="pBest",
        )
    )

    # Parcaciklar
    fig.add_trace(
        go.Scatter(
            x=positions[:, 0], y=positions[:, 1],
            mode="markers",
            marker=dict(
                size=10, color=fitness, colorscale="Plasma",
                cmin=float(np.min(fitness)),
                cmax=float(np.percentile(fitness, 90)),
                line=dict(width=1, color="white"),
            ),
            text=[f"P{i} f={fitness[i]:.4f}" for i in range(len(positions))],
            hovertemplate="%{text}<extra></extra>",
            name="Parcaciklar",
        )
    )

    # Global best tarihce
    if len(gbest_history) > 1:
        hist_arr = np.array(gbest_history)
        fig.add_trace(
            go.Scatter(
                x=hist_arr[:, 0], y=hist_arr[:, 1],
                mode="lines+markers",
                line=dict(width=2.5, color="rgba(239,68,68,0.6)"),
                marker=dict(size=3, color="rgba(239,68,68,0.4)"),
                name="gBest yolu",
            )
        )

    # Global best
    fig.add_trace(
        go.Scatter(
            x=[gbest_position[0]], y=[gbest_position[1]],
            mode="markers+text",
            marker=dict(size=18, symbol="star", color="#ef4444", line=dict(width=2, color="white")),
            text=[f"gBest"],
            textposition="top right",
            name="Global best",
        )
    )

    # Gercek minimum
    gm = problem.global_min
    fig.add_trace(
        go.Scatter(
            x=[gm[0]], y=[gm[1]],
            mode="markers+text",
            marker=dict(size=16, symbol="x", color="#22c55e", line=dict(width=2, color="#166534")),
            text=["Gercek min"],
            textposition="bottom right",
            name="Gercek minimum",
        )
    )

    fig.update_layout(
        title=f"Kus bakisi (kontur) | {problem.name} | Iterasyon {iteration}",
        height=460,
        margin=dict(l=0, r=0, t=55, b=0),
        xaxis_title="x1", yaxis_title="x2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, font=dict(size=10)),
    )
    return fig


def build_pso_progress_figure(best_hist: list[float], mean_hist: list[float]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Iterasyon": np.arange(1, len(best_hist) + 1),
            "Global best": best_hist,
            "Suru ortalamasi": mean_hist,
        }
    )
    fig = px.line(
        frame,
        x="Iterasyon",
        y=["Global best", "Suru ortalamasi"],
        labels={"value": "Maliyet", "variable": "Olcum"},
    )
    fig.update_layout(height=290, margin=dict(l=0, r=0, t=35, b=0), legend_title_text="")
    return fig


def build_pso_dynamics_figure(velocity_hist: list[float], diversity_hist: list[float]) -> go.Figure:
    steps = np.arange(1, len(velocity_hist) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=velocity_hist,
            mode="lines",
            name="Ortalama hiz normu",
            yaxis="y1",
            line=dict(color="#f59e0b"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=diversity_hist,
            mode="lines",
            name="Suru cesitliligi",
            yaxis="y2",
            line=dict(color="#06b6d4"),
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=0, r=0, t=35, b=0),
        yaxis=dict(title="Hiz"),
        yaxis2=dict(title="Cesitlilik", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_pso_live_info_text(
    iteration: int,
    config: PSOConfig,
    problem: PSOProblem,
    best_value: float,
    current_mean: float,
    first_best: float,
    improved_counter: int,
    best_position: np.ndarray,
    velocity_norm: float,
    diversity: float,
) -> str:
    improvement = ((first_best - best_value) / max(abs(first_best), 1e-12)) * 100 if iteration > 1 else 0.0
    dist_to_min = float(np.linalg.norm(best_position - problem.global_min))
    preview = ", ".join([f"{v:.6f}" for v in best_position])

    lines = [
        "=" * 40,
        f"  ITERASYON: {iteration} / {config.iterations}",
        "=" * 40,
        "",
        f"Problem: {problem.name}",
        f"Boyut: 2 (x1, x2)",
        "",
        "--- SONUCLAR ---",
        f"Global best (gBest) : {best_value:.8f}",
        f"Suru ortalamasi     : {current_mean:.8f}",
        f"Iyilesme            : %{improvement:.3f}",
        f"Min'e uzaklik       : {dist_to_min:.6f}",
        "",
        "--- SURU DINAMIKLERI ---",
        f"Iyilesen parcacik   : {improved_counter} / {config.swarm_size}",
        f"Ort. hiz normu      : {velocity_norm:.6f}",
        f"Suru cesitliligi    : {diversity:.6f}",
        "",
        "--- EN IYI POZISYON ---",
        f"x = [{preview}]",
        "",
        "--- PSO PARAMETRELERI ---",
        f"w  (inertia)    = {config.inertia_weight:.2f}",
        f"c1 (cognitive)  = {config.cognitive_coeff:.2f}",
        f"c2 (social)     = {config.social_coeff:.2f}",
        f"Hiz siniri oran = {config.velocity_clamp_ratio:.2f}",
        "",
        "--- PSO FORMULU ---",
        "v = w*v + c1*r1*(pBest-x) + c2*r2*(gBest-x)",
        "x = x + v",
        "",
        "Not:",
        problem.description,
    ]
    return "\n".join(lines)


def configure_sidebar() -> tuple[str, GAConfig | SAConfig | TabuConfig | ACOConfig | PSOConfig | BOConfig, bool, bool]:
    run_button = st.sidebar.button("Calistir", type="primary", use_container_width=True)
    clear_button = st.sidebar.button("Oturumu sifirla", use_container_width=True)
    st.sidebar.divider()
    algorithm = st.sidebar.selectbox(
        "Algoritma secimi",
        [
            "Genetik Algoritma",
            "Tavlama Algoritmasi",
            "Tabu Search Algoritmasi",
            "Karinca Kolonisi Algoritmasi",
            PSO_ALGORITHM_NAME,
            BO_ALGORITHM_NAME,
        ],
        index=0,
    )

    if algorithm == "Genetik Algoritma":
        st.sidebar.header("Genetik Algoritma Parametreleri")
        population_size = st.sidebar.slider("Populasyon", 40, 500, 180, step=10)
        generations_slider = st.sidebar.slider("Nesil sayisi (slider)", 50, 5000, 4000, step=25)
        generations_manual = st.sidebar.number_input(
            "Nesil sayisi (manuel)",
            min_value=50,
            max_value=200000,
            value=int(generations_slider),
            step=50,
        )
        generations = int(generations_manual)
        crossover_rate = st.sidebar.slider("Crossover olasiligi", 0.0, 1.0, 0.92, step=0.01)
        mutation_rate = st.sidebar.slider("Mutasyon olasiligi", 0.0, 0.35, 0.08, step=0.01)
        max_elitism = max(2, population_size // 5)
        elitism = st.sidebar.slider("Elit birey", 1, max_elitism, min(8, max_elitism))
        selection_method = st.sidebar.selectbox("Secim yontemi", ["Turnuva", "Rulet"])
        tournament_size = st.sidebar.slider("Turnuva boyutu", 2, 10, 4)
        crossover_method = st.sidebar.selectbox("Crossover tipi", ["OX", "PMX"])
        mutation_operator = st.sidebar.selectbox("Mutasyon tipi", ["Swap", "Inversion", "Scramble"])
        with st.sidebar.expander("Canli Akis Ayarlari", expanded=True):
            route_update_every = st.slider("Rota guncelleme (nesilde bir)", 1, 10, 1)
            analytics_update_every = st.slider("Grafik guncelleme (nesilde bir)", 1, 20, 3)
            heatmap_update_every = st.slider("Heatmap guncelleme (nesilde bir)", 1, 50, 8)
            frame_delay = st.slider("Kare gecikmesi (sn)", 0.0, 0.30, 0.05, step=0.01)
        random_seed = st.sidebar.number_input(
            "Rastgele tohum", min_value=0, max_value=999999, value=42, key="ga_seed"
        )

        config: GAConfig | SAConfig | TabuConfig | ACOConfig | PSOConfig = GAConfig(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            selection_method=selection_method,
            tournament_size=tournament_size,
            crossover_method=crossover_method,
            mutation_operator=mutation_operator,
            route_update_every=route_update_every,
            analytics_update_every=analytics_update_every,
            heatmap_update_every=heatmap_update_every,
            frame_delay=frame_delay,
            random_seed=int(random_seed),
        )
    elif algorithm == "Tavlama Algoritmasi":
        st.sidebar.header("Tavlama Algoritmasi Parametreleri")
        iterations_slider = st.sidebar.slider(
            "Iterasyon sayisi (slider)", 2000, 120000, 35000, step=500
        )
        iterations_manual = st.sidebar.number_input(
            "Iterasyon sayisi (manuel)",
            min_value=2000,
            max_value=1000000,
            value=int(iterations_slider),
            step=500,
        )
        iterations = int(iterations_manual)
        initial_temperature = st.sidebar.slider(
            "Baslangic sicakligi", 100.0, 10000.0, 3500.0, step=50.0
        )
        cooling_rate = st.sidebar.slider("Soguma orani", 0.9500, 0.99995, 0.9996, step=0.00005)
        min_temperature = st.sidebar.slider("Minimum sicaklik", 0.00001, 5.0, 0.0005, step=0.00001)
        neighbor_operator = st.sidebar.selectbox(
            "Komsu uretim tipi",
            ["2-Opt + Swap (onerilen)", "2-Opt", "Swap", "Inversion", "Scramble"],
        )
        two_opt_every = st.sidebar.slider(
            "2-Opt yerel iyilestirme araligi (0=kapali)", 0, 2000, 125, step=25
        )
        stagnation_limit = st.sidebar.slider("Stagnasyon limiti", 500, 20000, 3500, step=100)
        reheat_ratio = st.sidebar.slider("Reheat katsayisi", 0.10, 1.00, 0.35, step=0.05)
        with st.sidebar.expander("Canli Akis Ayarlari", expanded=True):
            route_update_every = st.slider("Rota guncelleme (iterasyonda bir)", 1, 500, 100)
            analytics_update_every = st.slider(
                "Grafik guncelleme (iterasyonda bir)", 1, 1000, 150
            )
            moves_update_every = st.slider("Hamle grafigi guncelleme", 1, 1000, 150)
            frame_delay = st.slider("Kare gecikmesi (sn)", 0.0, 0.30, 0.00, step=0.01)
        random_seed = st.sidebar.number_input(
            "Rastgele tohum", min_value=0, max_value=999999, value=42, key="sa_seed"
        )

        config = SAConfig(
            iterations=iterations,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            neighbor_operator=neighbor_operator,
            two_opt_every=two_opt_every,
            stagnation_limit=stagnation_limit,
            reheat_ratio=reheat_ratio,
            route_update_every=route_update_every,
            analytics_update_every=analytics_update_every,
            moves_update_every=moves_update_every,
            frame_delay=frame_delay,
            random_seed=int(random_seed),
        )
    elif algorithm == "Tabu Search Algoritmasi":
        st.sidebar.header("Tabu Search Parametreleri")
        iterations_slider = st.sidebar.slider(
            "Iterasyon sayisi (slider)", 500, 120000, 30000, step=500
        )
        iterations_manual = st.sidebar.number_input(
            "Iterasyon sayisi (manuel)",
            min_value=500,
            max_value=1000000,
            value=int(iterations_slider),
            step=500,
        )
        iterations = int(iterations_manual)
        candidate_pool_size = st.sidebar.slider("Aday 2-Opt hamlesi", 20, 1200, 260, step=20)
        tabu_tenure = st.sidebar.slider("Tabu tenure (iterasyon)", 3, 200, 35, step=1)
        aspiration_enabled = st.sidebar.checkbox("Aspiration kriteri", value=True)
        stagnation_limit = st.sidebar.slider("Stagnasyon limiti", 200, 25000, 3500, step=100)
        kick_ratio = st.sidebar.slider("Cesitlilik-kick orani", 0.02, 0.40, 0.12, step=0.01)
        with st.sidebar.expander("Canli Akis Ayarlari", expanded=True):
            route_update_every = st.slider("Rota guncelleme (iterasyonda bir)", 1, 500, 80)
            analytics_update_every = st.slider(
                "Grafik guncelleme (iterasyonda bir)", 1, 1000, 120
            )
            moves_update_every = st.slider("Hamle grafigi guncelleme", 1, 1000, 120)
            frame_delay = st.slider("Kare gecikmesi (sn)", 0.0, 0.30, 0.00, step=0.01)
        random_seed = st.sidebar.number_input(
            "Rastgele tohum", min_value=0, max_value=999999, value=42, key="tabu_seed"
        )

        config = TabuConfig(
            iterations=iterations,
            candidate_pool_size=candidate_pool_size,
            tabu_tenure=tabu_tenure,
            aspiration_enabled=aspiration_enabled,
            stagnation_limit=stagnation_limit,
            kick_ratio=kick_ratio,
            route_update_every=route_update_every,
            analytics_update_every=analytics_update_every,
            moves_update_every=moves_update_every,
            frame_delay=frame_delay,
            random_seed=int(random_seed),
        )
    elif algorithm == "Karinca Kolonisi Algoritmasi":
        st.sidebar.header("Karinca Kolonisi Parametreleri")
        ant_count = st.sidebar.slider("Karinca sayisi", 20, 240, 90, step=5)
        iterations_slider = st.sidebar.slider(
            "Iterasyon sayisi (slider)", 50, 3000, 700, step=25, key="aco_iterations_slider"
        )
        iterations_manual = st.sidebar.number_input(
            "Iterasyon sayisi (manuel)",
            min_value=50,
            max_value=100000,
            value=int(iterations_slider),
            step=25,
            key="aco_iterations_manual",
        )
        iterations = int(iterations_manual)
        alpha = st.sidebar.slider("Feromon agirligi (alpha)", 0.5, 5.0, 1.2, step=0.1)
        beta = st.sidebar.slider("Sezgisel agirlik (beta)", 1.0, 8.0, 4.8, step=0.1)
        evaporation_rate = st.sidebar.slider("Buharlasma orani", 0.05, 0.90, 0.35, step=0.01)
        pheromone_constant = st.sidebar.slider("Feromon sabiti (Q)", 100.0, 20000.0, 4000.0, step=100.0)
        elitist_weight = st.sidebar.slider("Elitist takviye", 0, 10, 2)
        candidate_k = st.sidebar.slider("Aday komsu siniri (0=sinirsiz)", 0, 50, 18)
        two_opt_every = st.sidebar.slider("2-Opt periyodu (0=kapali)", 0, 250, 25, step=5)
        with st.sidebar.expander("Canli Akis Ayarlari", expanded=True):
            route_update_every = st.slider("Rota guncelleme (iterasyonda bir)", 1, 100, 10)
            analytics_update_every = st.slider("Grafik guncelleme (iterasyonda bir)", 1, 200, 20)
            ants_update_every = st.slider("Karinca grafigi guncelleme", 1, 200, 20)
            frame_delay = st.slider("Kare gecikmesi (sn)", 0.0, 0.30, 0.00, step=0.01)
        random_seed = st.sidebar.number_input(
            "Rastgele tohum", min_value=0, max_value=999999, value=42, key="aco_seed"
        )

        config = ACOConfig(
            ant_count=ant_count,
            iterations=iterations,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_constant=pheromone_constant,
            elitist_weight=elitist_weight,
            candidate_k=candidate_k,
            two_opt_every=two_opt_every,
            route_update_every=route_update_every,
            analytics_update_every=analytics_update_every,
            ants_update_every=ants_update_every,
            frame_delay=frame_delay,
            random_seed=int(random_seed),
        )
    elif algorithm == PSO_ALGORITHM_NAME:
        st.sidebar.header("PSO Parametreleri")
        problem_name = st.sidebar.selectbox("Benchmark fonksiyon", PSO_PROBLEM_LABELS, index=0)
        st.sidebar.caption("Tum problemler 2D (x1, x2) olarak calisir ve 3D yuzey uzerinde gorsellestirilir.")

        swarm_size = st.sidebar.slider("Suru parcacik sayisi", 10, 200, 40, step=5)
        iterations_slider = st.sidebar.slider(
            "Iterasyon sayisi (slider)", 10, 2000, 300, step=10, key="pso_iterations_slider"
        )
        iterations_manual = st.sidebar.number_input(
            "Iterasyon sayisi (manuel)",
            min_value=10,
            max_value=50000,
            value=int(iterations_slider),
            step=10,
            key="pso_iterations_manual",
        )
        iterations = int(iterations_manual)
        st.sidebar.subheader("PSO Katsayilari")
        st.sidebar.caption("v = **w**·v + **c1**·r1·(pBest-x) + **c2**·r2·(gBest-x)")
        inertia_weight = st.sidebar.slider("Inertia (w)", 0.10, 0.99, 0.72, step=0.01)
        cognitive_coeff = st.sidebar.slider("Cognitive (c1)", 0.0, 4.0, 1.70, step=0.05)
        social_coeff = st.sidebar.slider("Social (c2)", 0.0, 4.0, 1.70, step=0.05)
        velocity_clamp_ratio = st.sidebar.slider(
            "Hiz siniri orani (vmax/range)", 0.01, 1.00, 0.25, step=0.01
        )
        with st.sidebar.expander("Canli Akis Ayarlari", expanded=True):
            route_update_every = st.slider("3D guncelleme (iterasyonda bir)", 1, 50, 5)
            analytics_update_every = st.slider("Grafik guncelleme (iterasyonda bir)", 1, 100, 10)
            frame_delay = st.slider("Kare gecikmesi (sn)", 0.0, 1.00, 0.10, step=0.01)
        random_seed = st.sidebar.number_input(
            "Rastgele tohum", min_value=0, max_value=999999, value=42, key="pso_seed"
        )
        config = PSOConfig(
            problem_name=problem_name,
            swarm_size=swarm_size,
            iterations=iterations,
            inertia_weight=inertia_weight,
            cognitive_coeff=cognitive_coeff,
            social_coeff=social_coeff,
            velocity_clamp_ratio=velocity_clamp_ratio,
            route_update_every=route_update_every,
            analytics_update_every=analytics_update_every,
            frame_delay=frame_delay,
            random_seed=int(random_seed),
        )
    else:
        st.sidebar.header("Bayesian Optimization Parametreleri")
        problem_name = st.sidebar.selectbox("Benchmark fonksiyon", PSO_PROBLEM_LABELS, index=1, key="bo_problem")
        st.sidebar.caption("Tum problemler 2D (x1, x2) olarak calisir. GP surrogate model uzerinde gorsellestirilir.")

        n_initial = st.sidebar.slider("Baslangic rastgele ornekleme", 3, 30, 8, step=1, key="bo_n_initial")
        n_iterations = st.sidebar.slider("BO iterasyonu (yeni nokta sayisi)", 5, 100, 35, step=1, key="bo_n_iter")

        st.sidebar.subheader("Gaussian Process (GP)")
        kernel_type = st.sidebar.selectbox("Kernel (cekirdek) tipi", ["Matern (nu=2.5)", "Matern (nu=1.5)", "RBF"], index=0, key="bo_kernel")

        st.sidebar.subheader("Acquisition Function")
        st.sidebar.caption("Sonraki noktayi nereye orneklememiz gerektigini belirler.")
        acquisition_type = st.sidebar.selectbox(
            "Acquisition fonksiyonu",
            ["EI (Expected Improvement)", "UCB (Upper Confidence Bound)", "PI (Probability of Improvement)"],
            index=0, key="bo_acq",
        )
        kappa = 2.576
        xi = 0.01
        if "UCB" in acquisition_type:
            kappa = st.sidebar.slider("Kappa (kesif/somuru dengesi)", 0.1, 10.0, 2.576, step=0.1, key="bo_kappa")
        else:
            xi = st.sidebar.slider("Xi (iyilesme esigi)", 0.0, 0.5, 0.01, step=0.005, key="bo_xi")

        with st.sidebar.expander("Canli Akis Ayarlari", expanded=True):
            route_update_every = st.slider("Gorsel guncelleme (iterasyonda bir)", 1, 10, 1, key="bo_route_upd")
            analytics_update_every = st.slider("Grafik guncelleme (iterasyonda bir)", 1, 20, 2, key="bo_analytics_upd")
            frame_delay = st.slider("Kare gecikmesi (sn)", 0.0, 2.00, 0.30, step=0.05, key="bo_delay")
        random_seed = st.sidebar.number_input(
            "Rastgele tohum", min_value=0, max_value=999999, value=42, key="bo_seed"
        )
        config = BOConfig(
            problem_name=problem_name,
            n_initial=n_initial,
            n_iterations=n_iterations,
            kernel_type=kernel_type,
            acquisition_type=acquisition_type,
            kappa=kappa,
            xi=xi,
            route_update_every=route_update_every,
            analytics_update_every=analytics_update_every,
            frame_delay=frame_delay,
            random_seed=int(random_seed),
        )

    return algorithm, config, run_button, clear_button


def run_genetic_algorithm(cities: pd.DataFrame, config: GAConfig) -> dict:
    idx_izmir = cities.index[cities["city_key"] == "IZMIR"].tolist()
    if not idx_izmir:
        raise ValueError("Veri setinde IZMIR bulunamadi.")
    start_idx = int(idx_izmir[0])

    distance_matrix = build_distance_matrix(cities["lat"].to_numpy(), cities["lon"].to_numpy())
    available_cities = np.array([i for i in range(len(cities)) if i != start_idx], dtype=np.int16)
    rng = np.random.default_rng(config.random_seed)

    population = create_initial_population(config.population_size, available_cities, rng)

    best_hist: list[float] = []
    avg_hist: list[float] = []
    unique_hist: list[float] = []
    hamming_hist: list[float] = []
    mutation_events: deque = deque(maxlen=24)
    mutation_counter = 0
    crossover_counter = 0

    best_distance = float("inf")
    best_chromosome = population[0].copy()

    progress = st.progress(0.0)
    left_col, right_col = st.columns([3.4, 1.6], gap="medium")
    with left_col:
        route_ph = st.empty()
        line_left, line_right = st.columns(2)
        fitness_ph = line_left.empty()
        diversity_ph = line_right.empty()
        heatmap_ph = st.empty()
    with right_col:
        st.caption("Canli metin paneli (sabit kutu, kaydirarak incele)")
        info_ph = st.empty()

    for generation in range(1, config.generations + 1):
        distances = evaluate_population(population, start_idx, distance_matrix)
        order = np.argsort(distances)
        population = population[order]
        distances = distances[order]

        current_best = float(distances[0])
        current_avg = float(distances.mean())
        if current_best < best_distance:
            best_distance = current_best
            best_chromosome = population[0].copy()

        unique_ratio, hamming = population_diversity(population, population[0])
        best_hist.append(best_distance)
        avg_hist.append(current_avg)
        unique_hist.append(unique_ratio)
        hamming_hist.append(hamming)

        route_due = (
            generation == 1
            or generation == config.generations
            or generation % config.route_update_every == 0
        )
        analytics_due = (
            generation == 1
            or generation == config.generations
            or generation % config.analytics_update_every == 0
        )
        heatmap_due = (
            generation == 1
            or generation == config.generations
            or generation % config.heatmap_update_every == 0
        )

        if route_due:
            best_route = np.concatenate(([start_idx], best_chromosome, [start_idx]))
            route_fig = build_route_figure(best_route, cities, generation, best_distance)
            route_ph.plotly_chart(
                route_fig,
                use_container_width=True,
                key=f"route_live_{generation}",
            )
            info_text = build_live_info_text(
                generation=generation,
                config=config,
                best_distance=best_distance,
                first_best=best_hist[0],
                mutation_counter=mutation_counter,
                crossover_counter=crossover_counter,
                best_route=best_route,
                mutation_events=mutation_events,
                cities=cities,
            )
            info_ph.text_area(
                "Canli bilgi paneli",
                value=info_text,
                height=780,
                disabled=True,
                label_visibility="collapsed",
            )
            progress.progress(generation / config.generations)
            if config.frame_delay > 0:
                time.sleep(config.frame_delay)

        if analytics_due:
            fitness_ph.plotly_chart(
                build_fitness_figure(best_hist, avg_hist),
                use_container_width=True,
                key=f"fitness_live_{generation}",
            )
            diversity_ph.plotly_chart(
                build_diversity_figure(unique_hist, hamming_hist),
                use_container_width=True,
                key=f"diversity_live_{generation}",
            )

        if heatmap_due:
            heatmap_ph.plotly_chart(
                build_population_heatmap(population, distances, cities),
                use_container_width=True,
                key=f"heatmap_live_{generation}",
            )

        elite = population[: config.elitism].copy()
        parent_indices = select_parent_indices(
            distances, config.selection_method, config.tournament_size, rng
        )
        parents = population[parent_indices]

        children: list[np.ndarray] = []
        child_id = 0
        while len(children) < (config.population_size - config.elitism):
            p1_idx, p2_idx = rng.integers(0, len(parents), size=2)
            parent1 = parents[p1_idx]
            parent2 = parents[p2_idx]

            child1, child2 = crossover_pair(
                parent1, parent2, config.crossover_method, config.crossover_rate, rng
            )
            if not np.array_equal(child1, parent1) or not np.array_equal(child2, parent2):
                crossover_counter += 1

            for child in (child1, child2):
                before_mutation_distance = route_distance(child, start_idx, distance_matrix)
                mutated_child, event = mutate_chromosome(
                    child, config.mutation_rate, config.mutation_operator, rng
                )
                if event is not None:
                    mutation_counter += 1
                    event["generation"] = generation
                    event["child_id"] = child_id
                    after_mutation_distance = route_distance(mutated_child, start_idx, distance_matrix)
                    event["delta_km"] = after_mutation_distance - before_mutation_distance
                    mutation_events.append(event)
                children.append(mutated_child)
                child_id += 1
                if len(children) >= (config.population_size - config.elitism):
                    break

        population = np.vstack((elite, np.asarray(children, dtype=np.int16)))

    final_route = np.concatenate(([start_idx], best_chromosome, [start_idx]))
    return {
        "algorithm": "Genetik Algoritma",
        "best_distance": best_distance,
        "best_route": final_route,
        "history_best": best_hist,
        "history_avg": avg_hist,
        "history_unique": unique_hist,
        "history_hamming": hamming_hist,
        "mutation_counter": mutation_counter,
        "crossover_counter": crossover_counter,
        "completed_generations": config.generations,
    }


def run_simulated_annealing(cities: pd.DataFrame, config: SAConfig) -> dict:
    idx_izmir = cities.index[cities["city_key"] == "IZMIR"].tolist()
    if not idx_izmir:
        raise ValueError("Veri setinde IZMIR bulunamadi.")
    start_idx = int(idx_izmir[0])

    distance_matrix = build_distance_matrix(cities["lat"].to_numpy(), cities["lon"].to_numpy())
    available_cities = np.array([i for i in range(len(cities)) if i != start_idx], dtype=np.int16)
    rng = np.random.default_rng(config.random_seed)

    current_chromosome = rng.permutation(available_cities).astype(np.int16)
    current_distance = route_distance(current_chromosome, start_idx, distance_matrix)
    best_chromosome = current_chromosome.copy()
    best_distance = current_distance

    best_hist: list[float] = []
    current_hist: list[float] = []
    temperature_hist: list[float] = []
    acceptance_hist: list[float] = []
    move_events: deque = deque(maxlen=36)
    accepted_counter = 0
    worse_accepted_counter = 0
    local_search_counter = 0
    reheat_counter = 0
    stagnation_counter = 0
    temperature = config.initial_temperature

    progress = st.progress(0.0)
    left_col, right_col = st.columns([3.4, 1.6], gap="medium")
    with left_col:
        route_ph = st.empty()
        line_left, line_right = st.columns(2)
        progress_ph = line_left.empty()
        temperature_ph = line_right.empty()
        moves_ph = st.empty()
    with right_col:
        st.caption("Canli metin paneli (sabit kutu, kaydirarak incele)")
        info_ph = st.empty()

    for iteration in range(1, config.iterations + 1):
        previous_distance = current_distance
        candidate, candidate_distance, event = propose_sa_neighbor(
            current_chromosome,
            current_distance,
            operator=config.neighbor_operator,
            start_idx=start_idx,
            distance_matrix=distance_matrix,
            rng=rng,
        )
        delta = candidate_distance - previous_distance

        accepted = False
        accepted_worse = False
        if delta <= 0:
            accepted = True
        else:
            threshold = np.exp(-delta / max(temperature, 1e-12))
            if rng.random() < threshold:
                accepted = True
                accepted_worse = True

        if accepted:
            current_chromosome = candidate
            current_distance = candidate_distance
            accepted_counter += 1
            if accepted_worse:
                worse_accepted_counter += 1

        if config.two_opt_every > 0 and iteration % config.two_opt_every == 0:
            before_local_search = current_distance
            improved_route, improved_distance, best_pair = best_two_opt_improvement(
                current_chromosome, current_distance, start_idx, distance_matrix
            )
            if best_pair is not None and improved_distance + 1e-9 < current_distance:
                local_search_counter += 1
                current_chromosome = improved_route
                current_distance = improved_distance
                move_events.append(
                    {
                        "iteration": iteration,
                        "operator": "2-Opt Yerel",
                        "positions": f"{best_pair[0]}-{best_pair[1]}",
                        "before": [],
                        "after": [],
                        "delta_km": current_distance - before_local_search,
                        "temperature": temperature,
                        "accepted": True,
                        "accepted_worse": False,
                    }
                )

        if current_distance + 1e-9 < best_distance:
            best_distance = current_distance
            best_chromosome = current_chromosome.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        event["iteration"] = iteration
        event["delta_km"] = delta
        event["temperature"] = temperature
        event["accepted"] = accepted
        event["accepted_worse"] = accepted_worse
        if "before" not in event:
            event["before"] = []
        if "after" not in event:
            event["after"] = []
        if "operator" not in event:
            event["operator"] = config.neighbor_operator
        if "positions" not in event:
            event["positions"] = "-"
        move_events.append(event)

        if config.stagnation_limit > 0 and stagnation_counter >= config.stagnation_limit:
            reheat_counter += 1
            before_kick = current_distance
            current_chromosome = best_chromosome.copy()
            kick_moves = max(3, len(current_chromosome) // 10)
            for _ in range(kick_moves):
                i, j = sorted(rng.choice(len(current_chromosome), size=2, replace=False))
                current_chromosome[i], current_chromosome[j] = current_chromosome[j], current_chromosome[i]

            current_distance = route_distance(current_chromosome, start_idx, distance_matrix)
            temperature = max(temperature, config.initial_temperature * config.reheat_ratio)
            stagnation_counter = 0
            move_events.append(
                {
                    "iteration": iteration,
                    "operator": "Reheat-Kick",
                    "positions": f"{kick_moves} hamle",
                    "before": [],
                    "after": [],
                    "delta_km": current_distance - before_kick,
                    "temperature": temperature,
                    "accepted": True,
                    "accepted_worse": current_distance > before_kick,
                }
            )

        best_hist.append(best_distance)
        current_hist.append(current_distance)
        temperature_hist.append(temperature)
        acceptance_hist.append(accepted_counter / iteration)

        route_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.route_update_every == 0
        )
        analytics_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.analytics_update_every == 0
        )
        moves_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.moves_update_every == 0
        )

        if route_due:
            best_route = np.concatenate(([start_idx], best_chromosome, [start_idx]))
            route_fig = build_route_figure(
                best_route,
                cities,
                iteration,
                best_distance,
                step_label="Iterasyon",
            )
            route_ph.plotly_chart(
                route_fig,
                use_container_width=True,
                key=f"route_sa_live_{iteration}",
            )
            info_text = build_sa_live_info_text(
                iteration=iteration,
                config=config,
                temperature=temperature,
                best_distance=best_distance,
                current_distance=current_distance,
                first_best=best_hist[0],
                accepted_counter=accepted_counter,
                worse_accepted_counter=worse_accepted_counter,
                local_search_counter=local_search_counter,
                reheat_counter=reheat_counter,
                stagnation_counter=stagnation_counter,
                best_route=best_route,
                move_events=move_events,
                cities=cities,
            )
            info_ph.text_area(
                "Canli bilgi paneli",
                value=info_text,
                height=780,
                disabled=True,
                label_visibility="collapsed",
            )
            progress.progress(iteration / config.iterations)
            if config.frame_delay > 0:
                time.sleep(config.frame_delay)

        if analytics_due:
            progress_ph.plotly_chart(
                build_sa_progress_figure(best_hist, current_hist),
                use_container_width=True,
                key=f"fitness_sa_live_{iteration}",
            )
            temperature_ph.plotly_chart(
                build_sa_temperature_figure(temperature_hist, acceptance_hist),
                use_container_width=True,
                key=f"temperature_sa_live_{iteration}",
            )

        if moves_due:
            moves_ph.plotly_chart(
                build_sa_moves_figure(move_events),
                use_container_width=True,
                key=f"moves_sa_live_{iteration}",
            )

        temperature = max(config.min_temperature, temperature * config.cooling_rate)

    final_route = np.concatenate(([start_idx], best_chromosome, [start_idx]))
    return {
        "algorithm": "Tavlama Algoritmasi",
        "best_distance": best_distance,
        "best_route": final_route,
        "history_best": best_hist,
        "history_current": current_hist,
        "history_temperature": temperature_hist,
        "history_acceptance": acceptance_hist,
        "accepted_moves": accepted_counter,
        "worse_accepted_moves": worse_accepted_counter,
        "local_search_hits": local_search_counter,
        "reheat_count": reheat_counter,
        "completed_iterations": config.iterations,
    }


def run_tabu_search(cities: pd.DataFrame, config: TabuConfig) -> dict:
    idx_izmir = cities.index[cities["city_key"] == "IZMIR"].tolist()
    if not idx_izmir:
        raise ValueError("Veri setinde IZMIR bulunamadi.")
    start_idx = int(idx_izmir[0])

    distance_matrix = build_distance_matrix(cities["lat"].to_numpy(), cities["lon"].to_numpy())
    available_cities = np.array([i for i in range(len(cities)) if i != start_idx], dtype=np.int16)
    rng = np.random.default_rng(config.random_seed)

    current_chromosome = rng.permutation(available_cities).astype(np.int16)
    current_distance = route_distance(current_chromosome, start_idx, distance_matrix)
    best_chromosome = current_chromosome.copy()
    best_distance = current_distance

    best_hist: list[float] = []
    current_hist: list[float] = []
    tabu_size_hist: list[float] = []
    aspiration_hist: list[float] = []
    stagnation_hist: list[float] = []
    move_events: deque = deque(maxlen=36)

    tabu_expiry: dict[tuple[int, int], int] = {}
    aspiration_counter = 0
    diversification_counter = 0
    stagnation_counter = 0

    progress = st.progress(0.0)
    left_col, right_col = st.columns([3.4, 1.6], gap="medium")
    with left_col:
        route_ph = st.empty()
        line_left, line_right = st.columns(2)
        progress_ph = line_left.empty()
        status_ph = line_right.empty()
        moves_ph = st.empty()
    with right_col:
        st.caption("Canli metin paneli (sabit kutu, kaydirarak incele)")
        info_ph = st.empty()

    n = len(current_chromosome)
    for iteration in range(1, config.iterations + 1):
        expired_moves = [move for move, expiry in tabu_expiry.items() if expiry <= iteration]
        for move in expired_moves:
            del tabu_expiry[move]

        candidate_pairs = sample_two_opt_pairs(n, config.candidate_pool_size, rng)
        chosen_move: tuple[int, int] | None = None
        chosen_delta = float("inf")
        chosen_distance = float("inf")
        chosen_was_tabu = False
        chosen_aspiration = False
        fallback_move: tuple[int, int] | None = None
        fallback_delta = float("inf")
        fallback_distance = float("inf")
        fallback_was_tabu = False

        for i, j in candidate_pairs:
            delta = two_opt_delta(current_chromosome, i, j, start_idx, distance_matrix)
            candidate_distance = current_distance + delta
            move = (int(i), int(j))
            tabu_until = tabu_expiry.get(move, 0)
            is_tabu = tabu_until > iteration
            aspiration = bool(config.aspiration_enabled and candidate_distance + 1e-9 < best_distance)
            admissible = (not is_tabu) or aspiration

            if admissible and candidate_distance < chosen_distance:
                chosen_move = move
                chosen_delta = delta
                chosen_distance = candidate_distance
                chosen_was_tabu = is_tabu
                chosen_aspiration = aspiration

            if candidate_distance < fallback_distance:
                fallback_move = move
                fallback_delta = delta
                fallback_distance = candidate_distance
                fallback_was_tabu = is_tabu

        if chosen_move is None and fallback_move is not None:
            chosen_move = fallback_move
            chosen_delta = fallback_delta
            chosen_distance = fallback_distance
            chosen_was_tabu = fallback_was_tabu
            chosen_aspiration = False

        if chosen_move is None:
            best_hist.append(best_distance)
            current_hist.append(current_distance)
            tabu_size_hist.append(float(len(tabu_expiry)))
            aspiration_hist.append(aspiration_counter / iteration)
            stagnation_hist.append(float(stagnation_counter))
            continue

        i, j = chosen_move
        candidate = apply_two_opt_segment(current_chromosome, i, j)
        event = {
            "iteration": iteration,
            "operator": "2-Opt",
            "positions": f"{i}-{j}",
            "before": current_chromosome[i : j + 1].tolist(),
            "after": candidate[i : j + 1].tolist(),
            "delta_km": chosen_delta,
            "temperature": np.nan,
            "accepted": True,
            "accepted_worse": chosen_delta > 0,
            "was_tabu": chosen_was_tabu,
            "aspiration": chosen_aspiration,
        }
        move_events.append(event)

        current_chromosome = candidate
        current_distance = chosen_distance
        tabu_expiry[(i, j)] = iteration + config.tabu_tenure
        if chosen_aspiration:
            aspiration_counter += 1

        if current_distance + 1e-9 < best_distance:
            best_distance = current_distance
            best_chromosome = current_chromosome.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if config.stagnation_limit > 0 and stagnation_counter >= config.stagnation_limit:
            diversification_counter += 1
            before_kick = current_distance
            current_chromosome = best_chromosome.copy()
            kick_moves = max(2, int(len(current_chromosome) * config.kick_ratio))
            for _ in range(kick_moves):
                a, b = sorted(rng.choice(len(current_chromosome), size=2, replace=False))
                current_chromosome[a], current_chromosome[b] = current_chromosome[b], current_chromosome[a]
            current_distance = route_distance(current_chromosome, start_idx, distance_matrix)
            tabu_expiry.clear()
            stagnation_counter = 0
            move_events.append(
                {
                    "iteration": iteration,
                    "operator": "Diversification-Kick",
                    "positions": f"{kick_moves} swap",
                    "before": [],
                    "after": [],
                    "delta_km": current_distance - before_kick,
                    "temperature": np.nan,
                    "accepted": True,
                    "accepted_worse": current_distance > before_kick,
                    "was_tabu": False,
                    "aspiration": False,
                }
            )

            if current_distance + 1e-9 < best_distance:
                best_distance = current_distance
                best_chromosome = current_chromosome.copy()

        best_hist.append(best_distance)
        current_hist.append(current_distance)
        tabu_size_hist.append(float(len(tabu_expiry)))
        aspiration_hist.append(aspiration_counter / iteration)
        stagnation_hist.append(float(stagnation_counter))

        route_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.route_update_every == 0
        )
        analytics_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.analytics_update_every == 0
        )
        moves_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.moves_update_every == 0
        )

        if route_due:
            best_route = np.concatenate(([start_idx], best_chromosome, [start_idx]))
            route_fig = build_route_figure(
                best_route,
                cities,
                iteration,
                best_distance,
                step_label="Iterasyon",
            )
            route_ph.plotly_chart(
                route_fig,
                use_container_width=True,
                key=f"route_tabu_live_{iteration}",
            )
            info_text = build_tabu_live_info_text(
                iteration=iteration,
                config=config,
                best_distance=best_distance,
                current_distance=current_distance,
                first_best=best_hist[0],
                tabu_size=len(tabu_expiry),
                stagnation_counter=stagnation_counter,
                aspiration_counter=aspiration_counter,
                diversification_counter=diversification_counter,
                best_route=best_route,
                move_events=move_events,
                cities=cities,
            )
            info_ph.text_area(
                "Canli bilgi paneli",
                value=info_text,
                height=780,
                disabled=True,
                label_visibility="collapsed",
            )
            progress.progress(iteration / config.iterations)
            if config.frame_delay > 0:
                time.sleep(config.frame_delay)

        if analytics_due:
            progress_ph.plotly_chart(
                build_sa_progress_figure(best_hist, current_hist),
                use_container_width=True,
                key=f"progress_tabu_live_{iteration}",
            )
            status_ph.plotly_chart(
                build_tabu_status_figure(tabu_size_hist, aspiration_hist, stagnation_hist),
                use_container_width=True,
                key=f"status_tabu_live_{iteration}",
            )

        if moves_due:
            moves_ph.plotly_chart(
                build_sa_moves_figure(move_events),
                use_container_width=True,
                key=f"moves_tabu_live_{iteration}",
            )

    final_route = np.concatenate(([start_idx], best_chromosome, [start_idx]))
    return {
        "algorithm": "Tabu Search Algoritmasi",
        "best_distance": best_distance,
        "best_route": final_route,
        "history_best": best_hist,
        "history_current": current_hist,
        "history_tabu_size": tabu_size_hist,
        "history_aspiration": aspiration_hist,
        "history_stagnation": stagnation_hist,
        "completed_iterations": config.iterations,
        "tabu_tenure": config.tabu_tenure,
        "candidate_pool_size": config.candidate_pool_size,
        "aspiration_count": aspiration_counter,
        "diversification_count": diversification_counter,
    }


def run_ant_colony(cities: pd.DataFrame, config: ACOConfig) -> dict:
    idx_izmir = cities.index[cities["city_key"] == "IZMIR"].tolist()
    if not idx_izmir:
        raise ValueError("Veri setinde IZMIR bulunamadi.")
    start_idx = int(idx_izmir[0])

    distance_matrix = build_distance_matrix(cities["lat"].to_numpy(), cities["lon"].to_numpy())
    heuristic = 1.0 / (distance_matrix + 1e-9)
    np.fill_diagonal(heuristic, 0.0)
    pheromone = np.full(distance_matrix.shape, 1.0, dtype=np.float64)
    np.fill_diagonal(pheromone, 0.0)

    nearest_neighbors: list[np.ndarray] = []
    for city_idx in range(len(cities)):
        order = np.argsort(distance_matrix[city_idx])
        nearest_neighbors.append(order[order != city_idx].astype(np.int16))

    available_cities = np.array([i for i in range(len(cities)) if i != start_idx], dtype=np.int16)
    rng = np.random.default_rng(config.random_seed)

    best_chromosome = rng.permutation(available_cities).astype(np.int16)
    best_distance = route_distance(best_chromosome, start_idx, distance_matrix)
    local_search_counter = 0

    best_hist: list[float] = []
    iteration_best_hist: list[float] = []
    pheromone_mean_hist: list[float] = []
    pheromone_max_hist: list[float] = []
    iteration_events: deque = deque(maxlen=20)
    last_ant_distances = np.array([], dtype=np.float64)

    progress = st.progress(0.0)
    left_col, right_col = st.columns([3.4, 1.6], gap="medium")
    with left_col:
        route_ph = st.empty()
        hyperspace_ph = st.empty()
        line_left, line_right = st.columns(2)
        progress_ph = line_left.empty()
        pheromone_ph = line_right.empty()
        ants_ph = st.empty()
    with right_col:
        st.caption("Canli metin paneli (sabit kutu, kaydirarak incele)")
        info_ph = st.empty()

    for iteration in range(1, config.iterations + 1):
        ant_routes: list[np.ndarray] = []
        ant_distances = np.empty(config.ant_count, dtype=np.float64)

        for ant_idx in range(config.ant_count):
            route = construct_aco_route(
                start_idx=start_idx,
                available_cities=available_cities,
                pheromone=pheromone,
                heuristic=heuristic,
                alpha=config.alpha,
                beta=config.beta,
                candidate_k=config.candidate_k,
                nearest_neighbors=nearest_neighbors,
                rng=rng,
            )
            ant_routes.append(route)
            ant_distances[ant_idx] = route_distance(route, start_idx, distance_matrix)

        order = np.argsort(ant_distances)
        best_ant_idx = int(order[0])
        iteration_best_route = ant_routes[best_ant_idx]
        iteration_best_distance = float(ant_distances[best_ant_idx])

        local_search_used = False
        if config.two_opt_every > 0 and iteration % config.two_opt_every == 0:
            improved_route, improved_distance, pair = best_two_opt_improvement(
                iteration_best_route, iteration_best_distance, start_idx, distance_matrix
            )
            if pair is not None and improved_distance + 1e-9 < iteration_best_distance:
                local_search_counter += 1
                local_search_used = True
                iteration_best_route = improved_route
                iteration_best_distance = improved_distance
                ant_routes[best_ant_idx] = improved_route
                ant_distances[best_ant_idx] = improved_distance

        if iteration_best_distance + 1e-9 < best_distance:
            best_distance = iteration_best_distance
            best_chromosome = iteration_best_route.copy()

        pheromone *= 1.0 - config.evaporation_rate
        deposit_count = max(5, config.ant_count // 2)
        for ant_id in order[:deposit_count]:
            ant_distance = float(ant_distances[int(ant_id)])
            if ant_distance <= 0:
                continue
            delta = config.pheromone_constant / ant_distance
            add_pheromone_for_route(
                pheromone=pheromone,
                route=ant_routes[int(ant_id)],
                start_idx=start_idx,
                delta_pheromone=delta,
            )

        if config.elitist_weight > 0 and best_distance > 0:
            elite_delta = config.elitist_weight * (config.pheromone_constant / best_distance)
            add_pheromone_for_route(
                pheromone=pheromone,
                route=best_chromosome,
                start_idx=start_idx,
                delta_pheromone=elite_delta,
            )

        np.fill_diagonal(pheromone, 0.0)
        tau_max = max(1.0, (config.pheromone_constant / max(best_distance, 1.0)) * 20.0)
        np.clip(pheromone, 1e-8, tau_max, out=pheromone)

        upper = pheromone[np.triu_indices_from(pheromone, k=1)]
        pheromone_mean = float(np.mean(upper))
        pheromone_max = float(np.max(upper))

        best_hist.append(best_distance)
        iteration_best_hist.append(iteration_best_distance)
        pheromone_mean_hist.append(pheromone_mean)
        pheromone_max_hist.append(pheromone_max)
        last_ant_distances = ant_distances[order]
        iteration_events.append(
            {
                "iteration": iteration,
                "iteration_best": iteration_best_distance,
                "global_best": best_distance,
                "pheromone_mean": pheromone_mean,
                "pheromone_max": pheromone_max,
                "local_search": local_search_used,
            }
        )

        route_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.route_update_every == 0
        )
        analytics_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.analytics_update_every == 0
        )
        ants_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.ants_update_every == 0
        )

        if route_due:
            best_route = np.concatenate(([start_idx], best_chromosome, [start_idx]))
            route_fig = build_aco_swarm_figure(
                cities,
                start_idx=start_idx,
                best_route_idx=best_route,
                pheromone=pheromone,
                ant_routes=ant_routes,
                ant_distances=ant_distances,
                iteration=iteration,
                best_distance=best_distance,
            )
            route_ph.plotly_chart(
                route_fig,
                use_container_width=True,
                key=f"route_aco_live_{iteration}",
            )
            hyperspace_ph.plotly_chart(
                build_aco_hyperspace_figure(
                    cities=cities,
                    start_idx=start_idx,
                    best_route_idx=best_route,
                    pheromone=pheromone,
                    ant_routes=ant_routes,
                    ant_distances=ant_distances,
                    iteration=iteration,
                    best_distance=best_distance,
                ),
                use_container_width=True,
                key=f"route_aco_hyper_live_{iteration}",
            )
            info_text = build_aco_live_info_text(
                iteration=iteration,
                config=config,
                best_distance=best_distance,
                iteration_best_distance=iteration_best_distance,
                first_best=best_hist[0],
                pheromone_mean=pheromone_mean,
                pheromone_max=pheromone_max,
                local_search_counter=local_search_counter,
                best_route=best_route,
                iteration_events=iteration_events,
                cities=cities,
            )
            info_ph.text_area(
                "Canli bilgi paneli",
                value=info_text,
                height=780,
                disabled=True,
                label_visibility="collapsed",
            )
            progress.progress(iteration / config.iterations)
            if config.frame_delay > 0:
                time.sleep(config.frame_delay)

        if analytics_due:
            progress_ph.plotly_chart(
                build_aco_progress_figure(best_hist, iteration_best_hist),
                use_container_width=True,
                key=f"progress_aco_live_{iteration}",
            )
            pheromone_ph.plotly_chart(
                build_aco_pheromone_figure(pheromone_mean_hist, pheromone_max_hist),
                use_container_width=True,
                key=f"pheromone_aco_live_{iteration}",
            )

        if ants_due:
            ants_ph.plotly_chart(
                build_aco_ant_distance_figure(last_ant_distances),
                use_container_width=True,
                key=f"ants_aco_live_{iteration}",
            )

    final_route = np.concatenate(([start_idx], best_chromosome, [start_idx]))
    return {
        "algorithm": "Karinca Kolonisi Algoritmasi",
        "best_distance": best_distance,
        "best_route": final_route,
        "history_best": best_hist,
        "history_iteration_best": iteration_best_hist,
        "history_pheromone_mean": pheromone_mean_hist,
        "history_pheromone_max": pheromone_max_hist,
        "local_search_hits": local_search_counter,
        "ant_count": config.ant_count,
        "evaporation_rate": config.evaporation_rate,
        "completed_iterations": config.iterations,
    }


def run_particle_swarm(config: PSOConfig) -> dict:
    problem = build_pso_problem(config.problem_name)
    rng = np.random.default_rng(config.random_seed)

    lb = problem.lower_bounds.astype(np.float64)
    ub = problem.upper_bounds.astype(np.float64)
    dim = 2

    positions = rng.uniform(lb, ub, size=(config.swarm_size, dim))
    range_span = np.maximum(ub - lb, 1e-9)
    vmax = config.velocity_clamp_ratio * range_span
    velocities = rng.uniform(-vmax, vmax, size=(config.swarm_size, dim))

    fitness = np.asarray([problem.objective(pos) for pos in positions], dtype=np.float64)
    personal_best_positions = positions.copy()
    personal_best_fitness = fitness.copy()

    best_idx = int(np.argmin(fitness))
    global_best_position = positions[best_idx].copy()
    global_best_value = float(fitness[best_idx])
    improved_counter = 0
    gbest_history: list[np.ndarray] = [global_best_position.copy()]

    best_hist: list[float] = []
    mean_hist: list[float] = []
    velocity_hist: list[float] = []
    diversity_hist: list[float] = []

    progress = st.progress(0.0)
    left_col, right_col = st.columns([3.4, 1.6], gap="medium")
    with left_col:
        surface_ph = st.empty()
        contour_ph = st.empty()
        line_left, line_right = st.columns(2)
        progress_ph = line_left.empty()
        dynamics_ph = line_right.empty()
    with right_col:
        st.caption("Canli bilgi paneli")
        info_ph = st.empty()

    for iteration in range(1, config.iterations + 1):
        r1 = rng.random((config.swarm_size, dim))
        r2 = rng.random((config.swarm_size, dim))

        cognitive = config.cognitive_coeff * r1 * (personal_best_positions - positions)
        social = config.social_coeff * r2 * (global_best_position - positions)
        velocities = config.inertia_weight * velocities + cognitive + social
        velocities = np.clip(velocities, -vmax, vmax)

        positions = positions + velocities
        out_low = positions < lb
        out_high = positions > ub
        out_mask = out_low | out_high
        velocities[out_mask] *= -0.5
        positions = np.clip(positions, lb, ub)

        fitness = np.asarray([problem.objective(pos) for pos in positions], dtype=np.float64)
        improved_mask = fitness < personal_best_fitness
        improved_counter = int(np.count_nonzero(improved_mask))
        if improved_counter > 0:
            personal_best_positions[improved_mask] = positions[improved_mask]
            personal_best_fitness[improved_mask] = fitness[improved_mask]

        iteration_best_idx = int(np.argmin(fitness))
        iteration_best = float(fitness[iteration_best_idx])
        if iteration_best < global_best_value:
            global_best_value = iteration_best
            global_best_position = positions[iteration_best_idx].copy()
        gbest_history.append(global_best_position.copy())

        center = np.mean(positions, axis=0)
        diversity = float(np.mean(np.linalg.norm(positions - center, axis=1)))
        velocity_norm = float(np.mean(np.linalg.norm(velocities, axis=1)))
        mean_value = float(np.mean(fitness))

        best_hist.append(global_best_value)
        mean_hist.append(mean_value)
        velocity_hist.append(velocity_norm)
        diversity_hist.append(diversity)

        route_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.route_update_every == 0
        )
        analytics_due = (
            iteration == 1
            or iteration == config.iterations
            or iteration % config.analytics_update_every == 0
        )

        if route_due:
            surface_ph.plotly_chart(
                build_pso_3d_surface_figure(
                    problem=problem,
                    positions=positions,
                    fitness=fitness,
                    velocities=velocities,
                    pbest_positions=personal_best_positions,
                    pbest_fitness=personal_best_fitness,
                    gbest_position=global_best_position,
                    gbest_value=global_best_value,
                    gbest_history=gbest_history,
                    iteration=iteration,
                ),
                use_container_width=True,
                key=f"pso_surface_live_{iteration}",
            )
            contour_ph.plotly_chart(
                build_pso_contour_figure(
                    problem=problem,
                    positions=positions,
                    fitness=fitness,
                    velocities=velocities,
                    pbest_positions=personal_best_positions,
                    gbest_position=global_best_position,
                    gbest_value=global_best_value,
                    gbest_history=gbest_history,
                    iteration=iteration,
                ),
                use_container_width=True,
                key=f"pso_contour_live_{iteration}",
            )
            info_text = build_pso_live_info_text(
                iteration=iteration,
                config=config,
                problem=problem,
                best_value=global_best_value,
                current_mean=mean_value,
                first_best=best_hist[0],
                improved_counter=improved_counter,
                best_position=global_best_position,
                velocity_norm=velocity_norm,
                diversity=diversity,
            )
            info_ph.text_area(
                "Canli bilgi paneli",
                value=info_text,
                height=780,
                disabled=True,
                label_visibility="collapsed",
            )
            progress.progress(iteration / config.iterations)
            if config.frame_delay > 0:
                time.sleep(config.frame_delay)

        if analytics_due:
            progress_ph.plotly_chart(
                build_pso_progress_figure(best_hist, mean_hist),
                use_container_width=True,
                key=f"pso_progress_live_{iteration}",
            )
            dynamics_ph.plotly_chart(
                build_pso_dynamics_figure(velocity_hist, diversity_hist),
                use_container_width=True,
                key=f"pso_dynamics_live_{iteration}",
            )

    return {
        "algorithm": PSO_ALGORITHM_NAME,
        "problem_name": problem.name,
        "best_value": global_best_value,
        "best_position": global_best_position,
        "history_best": best_hist,
        "history_mean": mean_hist,
        "history_velocity": velocity_hist,
        "history_diversity": diversity_hist,
        "improved_particles_last_iter": improved_counter,
        "completed_iterations": config.iterations,
        "dimensions": 2,
    }


# ============================================================
# BAYESIAN OPTIMIZATION VISUALIZATION & RUN FUNCTIONS
# ============================================================

def _bo_build_kernel(kernel_type: str):
    """Kernel olustur."""
    if "RBF" in kernel_type:
        return C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    elif "1.5" in kernel_type:
        return C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    else:
        return C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)


def _bo_acquisition(X_cand: np.ndarray, gp: GaussianProcessRegressor,
                    y_best: float, acq_type: str, kappa: float, xi: float) -> np.ndarray:
    """Acquisition function degerlerini hesapla."""
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)

    if "UCB" in acq_type:
        return -(mu - kappa * sigma)  # minimize -> negate UCB
    elif "PI" in acq_type:
        z = (y_best - mu - xi) / sigma
        return -scipy_norm.cdf(z)
    else:  # EI
        z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * scipy_norm.cdf(z) + sigma * scipy_norm.pdf(z)
        return -ei  # negate for argmin


def build_bo_surrogate_figure(
    problem: PSOProblem,
    X_sampled: np.ndarray,
    y_sampled: np.ndarray,
    gp: GaussianProcessRegressor,
    next_point: np.ndarray | None,
    iteration: int,
    grid_resolution: int = 60,
):
    """GP surrogate model 3D yuzey + belirsizlik + orneklenen noktalar."""
    lb, ub = problem.lower_bounds, problem.upper_bounds
    x1 = np.linspace(lb[0], ub[0], grid_resolution)
    x2 = np.linspace(lb[1], ub[1], grid_resolution)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    mu, sigma = gp.predict(X_grid, return_std=True)
    Mu = mu.reshape(grid_resolution, grid_resolution)
    Sigma = sigma.reshape(grid_resolution, grid_resolution)

    fig = go.Figure()

    # GP mean surface
    fig.add_trace(go.Surface(
        x=X1, y=X2, z=Mu,
        colorscale="Viridis", opacity=0.7,
        colorbar=dict(title="GP mu(x)", x=1.02, len=0.45, y=0.75),
        name="GP Ortalama (mu)",
        showlegend=True,
    ))

    # GP uncertainty surface (mu + 2*sigma)
    fig.add_trace(go.Surface(
        x=X1, y=X2, z=Mu + 2 * Sigma,
        colorscale="Oranges", opacity=0.25,
        showscale=False,
        name="Belirsizlik (mu+2sigma)",
        showlegend=True,
    ))

    # Sampled points
    z_sampled = gp.predict(X_sampled)
    fig.add_trace(go.Scatter3d(
        x=X_sampled[:, 0], y=X_sampled[:, 1], z=y_sampled,
        mode="markers",
        marker=dict(size=6, color="red", symbol="circle",
                    line=dict(width=1, color="white")),
        name=f"Orneklenen ({len(X_sampled)})",
    ))

    # Next point
    if next_point is not None:
        z_next = gp.predict(next_point.reshape(1, -1))
        fig.add_trace(go.Scatter3d(
            x=[next_point[0]], y=[next_point[1]], z=[z_next[0]],
            mode="markers",
            marker=dict(size=12, color="yellow", symbol="diamond",
                        line=dict(width=2, color="black")),
            name="Sonraki nokta",
        ))

    # True minimum
    fig.add_trace(go.Scatter3d(
        x=[problem.global_min[0]], y=[problem.global_min[1]],
        z=[problem.global_min_value],
        mode="markers",
        marker=dict(size=8, color="#22c55e", symbol="cross",
                    line=dict(width=1, color="white")),
        name="Gercek minimum",
    ))

    best_idx = int(np.argmin(y_sampled))
    best_val = y_sampled[best_idx]
    dist_to_min = np.linalg.norm(X_sampled[best_idx] - problem.global_min)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Bayesian Optimization — GP Surrogate</b> | {problem.name} | Iterasyon {iteration}"
                f"<br><span style='font-size:12px;color:#64748b'>"
                f"Best={best_val:.6f} | Min'e uzaklik={dist_to_min:.4f} | Orneklem={len(X_sampled)}</span>"
            ),
            font=dict(size=14),
        ),
        height=620,
        margin=dict(l=0, r=0, t=80, b=0),
        scene=dict(
            xaxis_title="x1", yaxis_title="x2", zaxis_title="f(x)",
            bgcolor="rgba(15,23,42,0.03)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, x=0.0, font=dict(size=10)),
    )
    return fig


def build_bo_acquisition_contour(
    problem: PSOProblem,
    X_sampled: np.ndarray,
    y_sampled: np.ndarray,
    gp: GaussianProcessRegressor,
    next_point: np.ndarray | None,
    acq_type: str,
    kappa: float,
    xi: float,
    iteration: int,
    grid_resolution: int = 80,
):
    """Acquisition function 2D kontur haritasi."""
    lb, ub = problem.lower_bounds, problem.upper_bounds
    x1 = np.linspace(lb[0], ub[0], grid_resolution)
    x2 = np.linspace(lb[1], ub[1], grid_resolution)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    y_best = float(np.min(y_sampled))
    acq_vals = _bo_acquisition(X_grid, gp, y_best, acq_type, kappa, xi)
    Acq = (-acq_vals).reshape(grid_resolution, grid_resolution)  # flip back for display

    acq_label = acq_type.split(" (")[0]
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=x1, y=x2, z=Acq,
        colorscale="Hot", ncontours=25,
        colorbar=dict(title=acq_label, len=0.6),
        name=acq_label,
    ))

    # Sampled points
    fig.add_trace(go.Scatter(
        x=X_sampled[:, 0], y=X_sampled[:, 1],
        mode="markers",
        marker=dict(size=8, color="cyan", symbol="circle",
                    line=dict(width=1, color="black")),
        name=f"Orneklenen ({len(X_sampled)})",
    ))

    # Next point
    if next_point is not None:
        fig.add_trace(go.Scatter(
            x=[next_point[0]], y=[next_point[1]],
            mode="markers",
            marker=dict(size=14, color="yellow", symbol="star",
                        line=dict(width=2, color="black")),
            name="Sonraki nokta",
        ))

    # True minimum
    fig.add_trace(go.Scatter(
        x=[problem.global_min[0]], y=[problem.global_min[1]],
        mode="markers",
        marker=dict(size=10, color="#22c55e", symbol="cross",
                    line=dict(width=1.5, color="white")),
        name="Gercek minimum",
    ))

    fig.update_layout(
        title=dict(text=f"<b>Acquisition Function ({acq_label})</b> | Iterasyon {iteration}", font=dict(size=13)),
        xaxis_title="x1", yaxis_title="x2",
        height=420,
        margin=dict(l=40, r=10, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.0, font=dict(size=10)),
    )
    return fig


def build_bo_progress_figure(best_hist: list[float], sample_hist: list[float]):
    """BO ilerleme grafigi."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=best_hist, mode="lines+markers",
                             name="En iyi (best)", line=dict(width=2.5, color="#ef4444"),
                             marker=dict(size=4)))
    fig.add_trace(go.Scatter(y=sample_hist, mode="markers",
                             name="Son orneklem", marker=dict(size=5, color="#3b82f6", opacity=0.6)))
    fig.update_layout(
        title="BO Ilerleme: En Iyi Deger",
        xaxis_title="Toplam orneklem", yaxis_title="f(x)",
        height=280, margin=dict(l=40, r=10, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def build_bo_uncertainty_figure(sigma_hist: list[float]):
    """GP belirsizlik (ortalama sigma) grafigi."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sigma_hist, mode="lines+markers",
                             name="Ort. sigma", line=dict(width=2, color="#f59e0b"),
                             marker=dict(size=3)))
    fig.update_layout(
        title="GP Belirsizlik (Ort. Sigma)",
        xaxis_title="Iterasyon", yaxis_title="Ortalama sigma",
        height=280, margin=dict(l=40, r=10, t=40, b=30),
    )
    return fig


def build_bo_live_info_text(
    iteration: int, config: BOConfig, problem: PSOProblem,
    best_value: float, best_position: np.ndarray,
    n_samples: int, avg_sigma: float,
    first_best: float,
) -> str:
    """BO canli bilgi paneli."""
    dist = np.linalg.norm(best_position - problem.global_min)
    improvement = max(0.0, (first_best - best_value) / max(abs(first_best), 1e-12)) * 100.0

    lines = [
        "=" * 48,
        f"  ITERASYON: {iteration} / {config.n_iterations}",
        "=" * 48,
        "",
        f"  Problem: {problem.name}",
        f"  Boyut: 2 (x1, x2)",
        "",
        "--- SONUCLAR ---",
        f"  Best f(x)     : {best_value:.8f}",
        f"  Iyilesme      : %{improvement:.3f}",
        f"  Min'e uzaklik : {dist:.6f}",
        f"  Toplam orneklem: {n_samples}",
        "",
        "--- GP MODEL ---",
        f"  Kernel        : {config.kernel_type}",
        f"  Ort. sigma    : {avg_sigma:.6f}",
        "",
        "--- EN IYI POZISYON ---",
        f"  x = ({best_position[0]:.6f}, {best_position[1]:.6f})",
        "",
        "--- ACQUISITION ---",
        f"  Fonksiyon     : {config.acquisition_type}",
    ]
    if "UCB" in config.acquisition_type:
        lines.append(f"  Kappa         : {config.kappa:.3f}")
    else:
        lines.append(f"  Xi            : {config.xi:.4f}")

    lines += [
        "",
        "--- BO FORMULU ---",
        "  x_next = argmax alpha(x; GP)",
        "  GP: mu(x), sigma(x) -> posterior",
        "  alpha: Acquisition function",
        "",
        "Not:",
        f"  {problem.description.split(chr(10))[0]}",
    ]
    return "\n".join(lines)


def run_bayesian_optimization(config: BOConfig) -> dict:
    """Bayesian Optimization ana dongusu."""
    problem = build_pso_problem(config.problem_name)
    rng = np.random.default_rng(config.random_seed)

    # Kernel
    kernel = _bo_build_kernel(config.kernel_type)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6, normalize_y=True)

    # Initial random sampling
    lb, ub = problem.lower_bounds, problem.upper_bounds
    X_sampled = rng.uniform(lb, ub, size=(config.n_initial, 2))
    y_sampled = np.array([problem.objective(x) for x in X_sampled])

    # Fit initial GP
    gp.fit(X_sampled, y_sampled)

    best_idx = int(np.argmin(y_sampled))
    best_value = float(y_sampled[best_idx])
    best_position = X_sampled[best_idx].copy()
    first_best = best_value

    best_hist = [best_value]
    sample_hist = list(y_sampled)
    sigma_hist = []

    # Layout
    st.subheader(f"Bayesian Optimization — {problem.name}")
    progress = st.progress(0.0)
    left_col, right_col = st.columns([3.4, 1.6], gap="medium")
    with left_col:
        surface_ph = st.empty()
        acq_ph = st.empty()
        line_left, line_right = st.columns(2)
        progress_ph = line_left.empty()
        sigma_ph = line_right.empty()
    with right_col:
        st.caption("Canli bilgi paneli")
        info_ph = st.empty()

    # Candidate grid for acquisition optimization
    n_cand = 5000
    X_candidates_base = rng.uniform(lb, ub, size=(n_cand, 2))

    for iteration in range(1, config.n_iterations + 1):
        # Acquisition optimization
        y_best = float(np.min(y_sampled))
        X_candidates = np.vstack([
            X_candidates_base,
            rng.uniform(lb, ub, size=(1000, 2)),
        ])
        acq_values = _bo_acquisition(X_candidates, gp, y_best, config.acquisition_type, config.kappa, config.xi)
        next_idx = int(np.argmin(acq_values))
        next_point = X_candidates[next_idx]

        # Evaluate
        y_new = problem.objective(next_point)
        X_sampled = np.vstack([X_sampled, next_point.reshape(1, -1)])
        y_sampled = np.append(y_sampled, y_new)

        # Update GP
        gp.fit(X_sampled, y_sampled)

        # Update best
        if y_new < best_value:
            best_value = y_new
            best_position = next_point.copy()

        best_hist.append(best_value)
        sample_hist.append(y_new)

        # Avg sigma
        sigma_sample = gp.predict(X_candidates_base[:500], return_std=True)[1]
        avg_sigma = float(np.mean(sigma_sample))
        sigma_hist.append(avg_sigma)

        # Visualization update
        vis_due = (iteration % config.route_update_every == 0) or iteration == config.n_iterations
        analytics_due = (iteration % config.analytics_update_every == 0) or iteration == config.n_iterations

        if vis_due:
            next_pt_show = next_point if iteration < config.n_iterations else None
            surface_ph.plotly_chart(
                build_bo_surrogate_figure(problem, X_sampled, y_sampled, gp, next_pt_show, iteration),
                use_container_width=True,
                key=f"bo_surface_{iteration}",
            )
            acq_ph.plotly_chart(
                build_bo_acquisition_contour(
                    problem, X_sampled, y_sampled, gp, next_pt_show,
                    config.acquisition_type, config.kappa, config.xi, iteration,
                ),
                use_container_width=True,
                key=f"bo_acq_{iteration}",
            )
            info_text = build_bo_live_info_text(
                iteration=iteration, config=config, problem=problem,
                best_value=best_value, best_position=best_position,
                n_samples=len(X_sampled), avg_sigma=avg_sigma,
                first_best=first_best,
            )
            info_ph.text_area("Canli bilgi paneli", value=info_text, height=780,
                              disabled=True, label_visibility="collapsed")
            progress.progress(iteration / config.n_iterations)
            if config.frame_delay > 0:
                time.sleep(config.frame_delay)

        if analytics_due:
            progress_ph.plotly_chart(
                build_bo_progress_figure(best_hist, sample_hist),
                use_container_width=True,
                key=f"bo_progress_{iteration}",
            )
            sigma_ph.plotly_chart(
                build_bo_uncertainty_figure(sigma_hist),
                use_container_width=True,
                key=f"bo_sigma_{iteration}",
            )

    return {
        "algorithm": BO_ALGORITHM_NAME,
        "problem_name": problem.name,
        "best_value": best_value,
        "best_position": best_position,
        "history_best": best_hist,
        "history_samples": sample_hist,
        "history_sigma": sigma_hist,
        "completed_iterations": config.n_iterations,
        "total_samples": len(X_sampled),
        "dimensions": 2,
    }




def main() -> None:
    st.set_page_config(page_title="Optimizasyon Simulasyonu", page_icon=":round_pushpin:", layout="wide")
    st.markdown(
        "### Ostim Teknik Üniversitesi Yazılım Mühendisliği Akıllı Optimizasyon Algoritmaları"
    )
    st.title("Optimizasyon Cozumu: TSP ve PSO Problem Ailesi")
    st.caption(
        "Bu uygulama, TSP tabanli algoritmalar (GA/SA/Tabu/ACO) ve secili surekli optimizasyon "
        "problemleri icin PSO adimlarini canli olarak gorsellestirir."
    )
    st.markdown(
        """
        <style>
        div[data-testid="stTextArea"] textarea {
            font-family: "Segoe UI", sans-serif !important;
            font-size: 0.86rem !important;
            line-height: 1.35 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: #16a34a !important;
            border-color: #16a34a !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSlider [data-baseweb="slider"] > div > div {
            background-color: #16a34a !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {
            color: #15803d !important;
            font-weight: 600 !important;
        }
        </style>
        """
        ,
        unsafe_allow_html=True,
    )

    algorithm, config, run_button, clear_button = configure_sidebar()
    st.caption(f"Secili algoritma: {algorithm}")

    state_result = st.session_state.get("solver_result")
    state_algorithm = (
        state_result["algorithm"] if isinstance(state_result, dict) and "algorithm" in state_result else ""
    )
    needs_city_data = algorithm not in (PSO_ALGORITHM_NAME, BO_ALGORITHM_NAME)
    if state_algorithm and state_algorithm not in (PSO_ALGORITHM_NAME, BO_ALGORITHM_NAME):
        needs_city_data = True

    cities: pd.DataFrame | None = None
    if needs_city_data:
        csv_path = Path(__file__).with_name("81il.csv")
        if not csv_path.exists():
            st.error("81il.csv bulunamadi. Dosyayi uygulama ile ayni klasore koy.")
            st.stop()

        cities = load_turkiye_cities(str(csv_path))
        if len(cities) != 81:
            st.warning(f"Uyari: Veri setinden {len(cities)} il yuklendi (beklenen 81).")

        with st.expander("Veri ozeti", expanded=False):
            c1, c2 = st.columns(2)
            c1.write(cities[["plate", "city", "lat", "lon"]].head(10))
            c2.write(cities[["plate", "city", "lat", "lon"]].tail(10))
    elif isinstance(config, PSOConfig):
        problem_preview = build_pso_problem(config.problem_name)
        with st.expander("PSO problem ozeti", expanded=False):
            st.write(f"Problem: **{problem_preview.name}**")
            st.write(f"Boyut: **2** (x1, x2)")
            st.write(problem_preview.description)
    elif isinstance(config, BOConfig):
        problem_preview = build_pso_problem(config.problem_name)
        with st.expander("BO problem ozeti", expanded=False):
            st.write(f"Problem: **{problem_preview.name}**")
            st.write(f"Boyut: **2** (x1, x2)")
            st.write(f"Kernel: **{config.kernel_type}**")
            st.write(f"Acquisition: **{config.acquisition_type}**")
            st.write(problem_preview.description)

    if clear_button:
        for key in list(st.session_state.keys()):
            if key.startswith("ga_result") or key.startswith("solver_result"):
                del st.session_state[key]
        st.rerun()

    if run_button:
        if algorithm == "Genetik Algoritma":
            if cities is None:
                st.error("TSP verisi yuklenemedi.")
                st.stop()
            result = run_genetic_algorithm(cities, config)
        elif algorithm == "Tavlama Algoritmasi":
            if cities is None:
                st.error("TSP verisi yuklenemedi.")
                st.stop()
            result = run_simulated_annealing(cities, config)
        elif algorithm == "Tabu Search Algoritmasi":
            if cities is None:
                st.error("TSP verisi yuklenemedi.")
                st.stop()
            result = run_tabu_search(cities, config)
        elif algorithm == "Karinca Kolonisi Algoritmasi":
            if cities is None:
                st.error("TSP verisi yuklenemedi.")
                st.stop()
            result = run_ant_colony(cities, config)
        elif algorithm == PSO_ALGORITHM_NAME:
            result = run_particle_swarm(config)
        else:
            result = run_bayesian_optimization(config)
        st.session_state["solver_result"] = result

    if "solver_result" in st.session_state:
        result = st.session_state["solver_result"]
        st.subheader(f"Final Sonuc - {result['algorithm']}")
        if result["algorithm"] == PSO_ALGORITHM_NAME or result["algorithm"] == BO_ALGORITHM_NAME:
            best_position = np.asarray(result["best_position"], dtype=np.float64)
            col_l, col_r = st.columns([1.2, 2.8], gap="medium")
            with col_l:
                st.write(f"Problem: **{result['problem_name']}**")
                st.write(f"Boyut: **{result['dimensions']}**")
                st.write(f"Tamamlanan iterasyon: **{result['completed_iterations']}**")
                st.write(f"En iyi maliyet: **{result['best_value']:.10f}**")
                if result["algorithm"] == BO_ALGORITHM_NAME:
                    st.write(f"Toplam orneklem sayisi: **{result.get('total_samples', '?')}**")
                else:
                    st.write(f"Son iterasyonda iyilesen parcacik: **{result.get('improved_particles_last_iter', 0)}**")
            with col_r:
                st.caption("En iyi pozisyon vektoru")
                position_text = ", ".join([f"{v:.8f}" for v in best_position.tolist()])
                st.text_area(
                    "Best position",
                    value=position_text,
                    height=120,
                    label_visibility="collapsed",
                )

            vector_df = pd.DataFrame({"Dimension": np.arange(1, len(best_position) + 1), "Value": best_position})
            st.download_button(
                label="Best pozisyonu CSV indir",
                data=vector_df.to_csv(index=False).encode("utf-8"),
                file_name="pso_best_position.csv",
                mime="text/csv",
            )
        else:
            if cities is None:
                st.error("TSP sonucunu gostermek icin sehir verisi bulunamadi.")
                st.stop()
            route_names = [cities.iloc[int(idx)]["city"] for idx in result["best_route"]]
            col_l, col_r = st.columns([1.2, 2.8], gap="medium")
            with col_l:
                st.write(f"En iyi tur: **{result['best_distance']:,.2f} km**")
                if result["algorithm"] == "Genetik Algoritma":
                    st.write(f"Tamamlanan nesil: **{result['completed_generations']}**")
                    st.write(f"Mutasyon sayisi: **{result['mutation_counter']}**")
                    st.write(f"Crossover sayisi: **{result['crossover_counter']}**")
                elif result["algorithm"] == "Tavlama Algoritmasi":
                    acceptance = (result["accepted_moves"] / result["completed_iterations"]) * 100
                    worse_acceptance = (
                        result["worse_accepted_moves"] / result["completed_iterations"]
                    ) * 100
                    st.write(f"Tamamlanan iterasyon: **{result['completed_iterations']}**")
                    st.write(f"Kabul edilen hamle: **{result['accepted_moves']}** (%{acceptance:.2f})")
                    st.write(
                        f"Kotu ama kabul edilen: **{result['worse_accepted_moves']}** (%{worse_acceptance:.2f})"
                    )
                    st.write(f"2-Opt yerel iyilestirme: **{result.get('local_search_hits', 0)}**")
                    st.write(f"Reheat sayisi: **{result.get('reheat_count', 0)}**")
                elif result["algorithm"] == "Tabu Search Algoritmasi":
                    aspiration_ratio = (
                        result["aspiration_count"] / result["completed_iterations"]
                    ) * 100
                    st.write(f"Tamamlanan iterasyon: **{result['completed_iterations']}**")
                    st.write(f"Aday havuzu: **{result['candidate_pool_size']}**")
                    st.write(f"Tabu tenure: **{result['tabu_tenure']}**")
                    st.write(
                        f"Aspiration kullanimi: **{result['aspiration_count']}** (%{aspiration_ratio:.2f})"
                    )
                    st.write(f"Cesitlilik-kick sayisi: **{result['diversification_count']}**")
                else:
                    st.write(f"Tamamlanan iterasyon: **{result['completed_iterations']}**")
                    st.write(f"Karinca sayisi: **{result['ant_count']}**")
                    st.write(f"Buharlasma orani: **{result['evaporation_rate']:.3f}**")
                    st.write(f"2-Opt iyilestirme: **{result.get('local_search_hits', 0)}**")
            with col_r:
                st.caption("Final rota metni (kucuk ve kaydirilabilir kutu)")
                st.text_area(
                    "Final rota",
                    value=" -> ".join(route_names),
                    height=120,
                    label_visibility="collapsed",
                )

            route_df = pd.DataFrame(
                {
                    "Sira": np.arange(1, len(route_names) + 1),
                    "Sehir": route_names,
                }
            )
            st.download_button(
                label="Rotayi CSV indir",
                data=route_df.to_csv(index=False).encode("utf-8"),
                file_name=(
                    "ga_tsp_81il_izmir_rota.csv"
                    if result["algorithm"] == "Genetik Algoritma"
                    else (
                        "sa_tsp_81il_izmir_rota.csv"
                        if result["algorithm"] == "Tavlama Algoritmasi"
                        else (
                            "tabu_tsp_81il_izmir_rota.csv"
                            if result["algorithm"] == "Tabu Search Algoritmasi"
                            else "aco_tsp_81il_izmir_rota.csv"
                        )
                    )
                ),
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
