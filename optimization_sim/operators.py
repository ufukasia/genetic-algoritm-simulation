from __future__ import annotations

import numpy as np

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
