from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .data import load_turkiye_cities
from .models import BO_ALGORITHM_NAME, BOConfig, PSO_ALGORITHM_NAME, PSOConfig
from .problems import build_pso_problem
from .registry import algorithm_needs_city_data, run_selected_algorithm
from .results import render_solver_result
from .sidebar import configure_sidebar



def _load_city_data_if_needed(needs_city_data: bool) -> pd.DataFrame | None:
    if not needs_city_data:
        return None

    csv_path = Path(__file__).resolve().parent.parent / "81il.csv"
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

    return cities


def _render_problem_preview(config: PSOConfig | BOConfig) -> None:
    problem_preview = build_pso_problem(config.problem_name)
    if isinstance(config, PSOConfig):
        with st.expander("PSO problem ozeti", expanded=False):
            st.write(f"Problem: **{problem_preview.name}**")
            st.write("Boyut: **2** (x1, x2)")
            st.write(problem_preview.description)
    else:
        with st.expander("BO problem ozeti", expanded=False):
            st.write(f"Problem: **{problem_preview.name}**")
            st.write("Boyut: **2** (x1, x2)")
            st.write(f"Kernel: **{config.kernel_type}**")
            st.write(f"Acquisition: **{config.acquisition_type}**")
            st.write(problem_preview.description)


def _clear_session_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("ga_result") or key.startswith("solver_result"):
            del st.session_state[key]


def main() -> None:
    st.set_page_config(page_title="Optimizasyon Simulasyonu", page_icon=":round_pushpin:", layout="wide")
    st.markdown(
        "### Ostim Teknik ??niversitesi Yaz??l??m M??hendisli??i Ak??ll?? Optimizasyon Algoritmalar??"
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
        """,
        unsafe_allow_html=True,
    )

    algorithm, config, run_button, clear_button = configure_sidebar()
    st.caption(f"Secili algoritma: {algorithm}")

    state_result = st.session_state.get("solver_result")
    state_algorithm = (
        state_result["algorithm"] if isinstance(state_result, dict) and "algorithm" in state_result else ""
    )
    needs_city_data = algorithm_needs_city_data(algorithm)
    if state_algorithm:
        needs_city_data = needs_city_data or algorithm_needs_city_data(state_algorithm)

    cities = _load_city_data_if_needed(needs_city_data)

    if not needs_city_data and isinstance(config, (PSOConfig, BOConfig)):
        _render_problem_preview(config)

    if clear_button:
        _clear_session_state()
        st.rerun()

    if run_button:
        result = run_selected_algorithm(algorithm, config, cities)
        st.session_state["solver_result"] = result

    if "solver_result" in st.session_state:
        render_solver_result(st.session_state["solver_result"], cities)
