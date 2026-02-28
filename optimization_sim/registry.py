from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .algorithms import (
    run_ant_colony,
    run_bayesian_optimization,
    run_genetic_algorithm,
    run_particle_swarm,
    run_simulated_annealing,
    run_tabu_search,
)
from .models import BO_ALGORITHM_NAME, PSO_ALGORITHM_NAME


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    needs_city_data: bool
    run: Callable[..., dict[str, Any]]


ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "Genetik Algoritma": AlgorithmSpec("Genetik Algoritma", True, run_genetic_algorithm),
    "Tavlama Algoritmasi": AlgorithmSpec("Tavlama Algoritmasi", True, run_simulated_annealing),
    "Tabu Search Algoritmasi": AlgorithmSpec("Tabu Search Algoritmasi", True, run_tabu_search),
    "Karinca Kolonisi Algoritmasi": AlgorithmSpec("Karinca Kolonisi Algoritmasi", True, run_ant_colony),
    PSO_ALGORITHM_NAME: AlgorithmSpec(PSO_ALGORITHM_NAME, False, run_particle_swarm),
    BO_ALGORITHM_NAME: AlgorithmSpec(BO_ALGORITHM_NAME, False, run_bayesian_optimization),
}


def algorithm_needs_city_data(algorithm_name: str) -> bool:
    spec = ALGORITHM_SPECS.get(algorithm_name)
    return bool(spec and spec.needs_city_data)


def run_selected_algorithm(algorithm_name: str, config: Any, cities: pd.DataFrame | None) -> dict[str, Any]:
    spec = ALGORITHM_SPECS.get(algorithm_name)
    if spec is None:
        raise ValueError(f"Bilinmeyen algoritma: {algorithm_name}")

    if spec.needs_city_data:
        if cities is None:
            raise ValueError("Bu algoritma sehir verisi gerektiriyor.")
        return spec.run(cities, config)

    return spec.run(config)
