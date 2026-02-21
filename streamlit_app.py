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


EARTH_RADIUS_KM = 6371.0088
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
    route_idx: np.ndarray, cities: pd.DataFrame, generation: int, distance_km: float
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
        title=f"Nesil {generation} | En iyi tur: {distance_km:,.1f} km",
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


def configure_sidebar() -> GAConfig:
    st.sidebar.header("Genetik Algoritma Parametreleri")
    population_size = st.sidebar.slider("Populasyon", 40, 500, 180, step=10)
    generations_slider = st.sidebar.slider("Nesil sayisi (slider)", 50, 5000, 1500, step=25)
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
    st.sidebar.markdown("### Canli Akis Ayarlari")
    route_update_every = st.sidebar.slider("Rota guncelleme (nesilde bir)", 1, 10, 1)
    analytics_update_every = st.sidebar.slider("Grafik guncelleme (nesilde bir)", 1, 20, 3)
    heatmap_update_every = st.sidebar.slider("Heatmap guncelleme (nesilde bir)", 1, 50, 8)
    frame_delay = st.sidebar.slider("Kare gecikmesi (sn)", 0.0, 0.30, 0.05, step=0.01)
    random_seed = st.sidebar.number_input("Rastgele tohum", min_value=0, max_value=999999, value=42)

    return GAConfig(
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


def main() -> None:
    st.set_page_config(page_title="GA TSP Turkiye 81 Il", page_icon=":round_pushpin:", layout="wide")
    st.title("Genetik Algoritma ile TSP: Turkiye'de 81 Il Turu (Izmir baslangic/donus)")
    st.caption(
        "Bu uygulama, Izmir'den baslayip 81 ilin tamamina ugrayan ve tekrar Izmir'e donen "
        "rota icin genetik algoritmanin tum ana adimlarini canli gorsellestirir."
    )
    st.markdown(
        """
        <style>
        div[data-testid="stTextArea"] textarea {
            font-family: "Segoe UI", sans-serif !important;
            font-size: 0.86rem !important;
            line-height: 1.35 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    config = configure_sidebar()

    run_button = st.sidebar.button("Calistir", type="primary", use_container_width=True)
    clear_button = st.sidebar.button("Oturumu sifirla", use_container_width=True)

    if clear_button:
        for key in list(st.session_state.keys()):
            if key.startswith("ga_result"):
                del st.session_state[key]
        st.rerun()

    if run_button:
        result = run_genetic_algorithm(cities, config)
        st.session_state["ga_result"] = result

    if "ga_result" in st.session_state:
        result = st.session_state["ga_result"]
        st.subheader("Final Sonuc")
        route_names = [cities.iloc[int(idx)]["city"] for idx in result["best_route"]]
        col_l, col_r = st.columns([1.2, 2.8], gap="medium")
        with col_l:
            st.write(f"Tamamlanan nesil: **{result['completed_generations']}**")
            st.write(f"En iyi tur: **{result['best_distance']:,.2f} km**")
            st.write(f"Mutasyon sayisi: **{result['mutation_counter']}**")
            st.write(f"Crossover sayisi: **{result['crossover_counter']}**")
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
            file_name="ga_tsp_81il_izmir_rota.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
