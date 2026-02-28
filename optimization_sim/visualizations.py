from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor

from .models import ACOConfig, BOConfig, GAConfig, PSOConfig, PSOProblem, SAConfig, TabuConfig

def _bo_acquisition(
    X_cand: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    acq_type: str,
    kappa: float,
    xi: float,
) -> np.ndarray:
    """Acquisition function degerlerini hesapla."""
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)

    if "UCB" in acq_type:
        return -(mu - kappa * sigma)  # minimize -> negate UCB
    if "PI" in acq_type:
        z = (y_best - mu - xi) / sigma
        return -scipy_norm.cdf(z)

    # EI
    z = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * scipy_norm.cdf(z) + sigma * scipy_norm.pdf(z)
    return -ei  # negate for argmin

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
