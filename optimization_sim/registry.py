from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .algorithms import (
    run_ant_colony,
    run_bayesian_optimization,
    run_cma_es,
    run_genetic_algorithm,
    run_particle_swarm,
    run_simulated_annealing,
    run_tabu_search,
)
from .models import (
    ACOConfig,
    BO_ALGORITHM_NAME,
    BOConfig,
    CMA_ES_ALGORITHM_NAME,
    CMAESConfig,
    COMPARISON_SEED,
    CONTINUOUS_BENCHMARK_SET,
    GAConfig,
    PSO_ALGORITHM_NAME,
    PSOConfig,
    SAConfig,
    TSP_BENCHMARK_NAME,
    TabuConfig,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    needs_city_data: bool
    config_type: type
    run: Callable[..., dict[str, Any]]


ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "Genetik Algoritma": AlgorithmSpec("Genetik Algoritma", True, GAConfig, run_genetic_algorithm),
    "Tavlama Algoritmasi": AlgorithmSpec("Tavlama Algoritmasi", True, SAConfig, run_simulated_annealing),
    "Tabu Search Algoritmasi": AlgorithmSpec("Tabu Search Algoritmasi", True, TabuConfig, run_tabu_search),
    "Karinca Kolonisi Algoritmasi": AlgorithmSpec("Karinca Kolonisi Algoritmasi", True, ACOConfig, run_ant_colony),
    PSO_ALGORITHM_NAME: AlgorithmSpec(PSO_ALGORITHM_NAME, False, PSOConfig, run_particle_swarm),
    BO_ALGORITHM_NAME: AlgorithmSpec(BO_ALGORITHM_NAME, False, BOConfig, run_bayesian_optimization),
    CMA_ES_ALGORITHM_NAME: AlgorithmSpec(CMA_ES_ALGORITHM_NAME, False, CMAESConfig, run_cma_es),
}


def algorithm_needs_city_data(algorithm_name: str) -> bool:
    spec = ALGORITHM_SPECS.get(algorithm_name)
    return bool(spec and spec.needs_city_data)


def _resolve_completed_iterations(result: dict[str, Any]) -> int:
    if "completed_iterations" in result:
        return int(result["completed_iterations"])
    if "completed_generations" in result:
        return int(result["completed_generations"])
    raise ValueError("Sonuc sozlugunde tamamlanma bilgisi yok.")


def _resolve_best_value(result: dict[str, Any]) -> float:
    if "best_value" in result:
        return float(result["best_value"])
    if "best_distance" in result:
        return float(result["best_distance"])
    raise ValueError("Sonuc sozlugunde best degeri yok.")


def _resolve_history(result: dict[str, Any]) -> list[float]:
    if "history_best" in result:
        return [float(v) for v in result["history_best"]]
    raise ValueError("Sonuc sozlugunde history_best yok.")


def _validate_config_contract(spec: AlgorithmSpec, config: Any) -> None:
    if not isinstance(config, spec.config_type):
        raise TypeError(
            f"{spec.name} icin config tipi gecersiz. Beklenen: {spec.config_type.__name__}, "
            f"gelen: {type(config).__name__}"
        )
    if not hasattr(config, "random_seed"):
        raise ValueError(f"{spec.name} config'inde random_seed zorunludur.")
    if int(config.random_seed) != COMPARISON_SEED:
        raise ValueError(
            f"Karsilastirma tutarliligi icin tum algoritmalar {COMPARISON_SEED} seed'i ile calismalidir. "
            f"Gelen seed: {config.random_seed}"
        )

    if hasattr(config, "problem_name"):
        if config.problem_name not in CONTINUOUS_BENCHMARK_SET:
            raise ValueError(
                f"Gecersiz benchmark: {config.problem_name}. Izin verilenler: {', '.join(CONTINUOUS_BENCHMARK_SET)}"
            )


def _normalize_result(algorithm_name: str, config: Any, result: dict[str, Any]) -> dict[str, Any]:
    benchmark_name = getattr(config, "problem_name", TSP_BENCHMARK_NAME)
    result["benchmark"] = benchmark_name
    result["best"] = _resolve_best_value(result)
    result["history"] = _resolve_history(result)
    result["completed_iterations"] = _resolve_completed_iterations(result)
    result["random_seed"] = int(config.random_seed)
    result["benchmark_set"] = list(CONTINUOUS_BENCHMARK_SET)
    result["algorithm_key"] = algorithm_name
    return result


def run_selected_algorithm(algorithm_name: str, config: Any, cities: pd.DataFrame | None) -> dict[str, Any]:
    spec = ALGORITHM_SPECS.get(algorithm_name)
    if spec is None:
        raise ValueError(f"Bilinmeyen algoritma: {algorithm_name}")
    _validate_config_contract(spec, config)

    if spec.needs_city_data:
        if cities is None:
            raise ValueError("Bu algoritma sehir verisi gerektiriyor.")
        raw_result = spec.run(cities, config)
        return _normalize_result(algorithm_name, config, raw_result)

    raw_result = spec.run(config)
    return _normalize_result(algorithm_name, config, raw_result)
