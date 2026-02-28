from __future__ import annotations

import time
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, RBF

from .data import build_distance_matrix
from .models import ACOConfig, BO_ALGORITHM_NAME, BOConfig, CMA_ES_ALGORITHM_NAME, CMAESConfig, GAConfig, PSO_ALGORITHM_NAME, PSOConfig, SAConfig, TabuConfig
from .operators import (
    add_pheromone_for_route,
    apply_two_opt_segment,
    best_two_opt_improvement,
    construct_aco_route,
    create_initial_population,
    crossover_pair,
    evaluate_population,
    mutate_chromosome,
    population_diversity,
    propose_sa_neighbor,
    route_distance,
    sample_two_opt_pairs,
    select_parent_indices,
    two_opt_delta,
)
from .problems import build_pso_problem
from .visualizations import (
    build_aco_ant_distance_figure,
    build_aco_hyperspace_figure,
    build_aco_live_info_text,
    build_aco_pheromone_figure,
    build_aco_progress_figure,
    build_aco_swarm_figure,
    build_bo_acquisition_contour,
    build_bo_live_info_text,
    build_bo_progress_figure,
    build_bo_surrogate_figure,
    build_bo_uncertainty_figure,
    build_diversity_figure,
    build_fitness_figure,
    build_live_info_text,
    build_population_heatmap,
    build_pso_3d_surface_figure,
    build_pso_contour_figure,
    build_pso_dynamics_figure,
    build_pso_live_info_text,
    build_pso_progress_figure,
    build_route_figure,
    build_sa_live_info_text,
    build_sa_moves_figure,
    build_sa_progress_figure,
    build_sa_temperature_figure,
    build_tabu_live_info_text,
    build_tabu_status_figure,
)

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


def run_cma_es(config: CMAESConfig) -> dict:
    problem = build_pso_problem(config.problem_name)
    rng = np.random.default_rng(config.random_seed)

    lb = problem.lower_bounds.astype(np.float64)
    ub = problem.upper_bounds.astype(np.float64)
    dim = 2
    mean = rng.uniform(lb, ub)
    sigma = float(config.initial_sigma)
    cov = np.eye(dim, dtype=np.float64)

    lambda_ = max(4, int(config.population_size))
    mu = max(2, int(lambda_ * config.elite_ratio))
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)

    best_value = float("inf")
    best_position = mean.copy()
    best_hist: list[float] = []
    mean_hist: list[float] = []
    sigma_hist: list[float] = []
    diversity_hist: list[float] = []
    gbest_history: list[np.ndarray] = [best_position.copy()]
    stagnation_counter = 0
    stagnation_limit = max(20, config.iterations // 8)

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
        samples = rng.multivariate_normal(mean=mean, cov=(sigma**2) * cov, size=lambda_)
        positions = np.clip(samples, lb, ub)
        fitness = np.asarray([problem.objective(pos) for pos in positions], dtype=np.float64)
        order = np.argsort(fitness)
        elites = positions[order[:mu]]
        elite_fitness = fitness[order[:mu]]

        mean = np.sum(weights[:, None] * elites, axis=0)
        centered = elites - mean
        cov = (centered.T @ (centered * weights[:, None])) / max(np.sum(weights), 1e-12)
        cov += np.eye(dim) * 1e-9
        iter_best = float(elite_fitness[0])
        if iter_best < best_value:
            best_value = iter_best
            best_position = elites[0].copy()
            stagnation_counter = 0
            sigma = min(5.0, sigma * 1.03)
        else:
            stagnation_counter += 1
            sigma = max(1e-4, sigma * config.sigma_decay)

        if stagnation_counter >= stagnation_limit:
            mean = rng.uniform(lb, ub)
            cov = np.eye(dim, dtype=np.float64)
            sigma = max(config.initial_sigma, sigma * 1.2)
            stagnation_counter = 0

        gbest_history.append(best_position.copy())

        diversity = float(np.mean(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1)))
        mean_value = float(np.mean(fitness))
        best_hist.append(best_value)
        mean_hist.append(mean_value)
        sigma_hist.append(float(sigma))
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
            fake_velocities = np.zeros_like(positions)
            surface_ph.plotly_chart(
                build_pso_3d_surface_figure(
                    problem=problem,
                    positions=positions,
                    fitness=fitness,
                    velocities=fake_velocities,
                    pbest_positions=elites,
                    pbest_fitness=elite_fitness,
                    gbest_position=best_position,
                    gbest_value=best_value,
                    gbest_history=gbest_history,
                    iteration=iteration,
                ),
                use_container_width=True,
                key=f"cma_surface_live_{iteration}",
            )
            contour_ph.plotly_chart(
                build_pso_contour_figure(
                    problem=problem,
                    positions=positions,
                    fitness=fitness,
                    velocities=fake_velocities,
                    pbest_positions=elites,
                    gbest_position=best_position,
                    gbest_value=best_value,
                    gbest_history=gbest_history,
                    iteration=iteration,
                ),
                use_container_width=True,
                key=f"cma_contour_live_{iteration}",
            )
            info_text = (
                f"ITERASYON: {iteration} / {config.iterations}\n"
                f"Problem: {problem.name}\n"
                f"Best: {best_value:.8f}\n"
                f"Ortalama: {mean_value:.8f}\n"
                f"Sigma: {sigma:.6f}\n"
                f"Cesitlilik: {diversity:.6f}\n"
                f"Lambda (pop): {lambda_}\n"
                f"Mu (elit): {mu}\n"
                f"Sigma decay: {config.sigma_decay:.3f}"
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
                key=f"cma_progress_live_{iteration}",
            )
            dynamics_ph.plotly_chart(
                build_pso_dynamics_figure(sigma_hist, diversity_hist),
                use_container_width=True,
                key=f"cma_dynamics_live_{iteration}",
            )

    return {
        "algorithm": CMA_ES_ALGORITHM_NAME,
        "problem_name": problem.name,
        "best_value": best_value,
        "best_position": best_position,
        "history_best": best_hist,
        "history_mean": mean_hist,
        "history_sigma": sigma_hist,
        "history_diversity": diversity_hist,
        "completed_iterations": config.iterations,
        "dimensions": 2,
    }

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
