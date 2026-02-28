from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .models import BO_ALGORITHM_NAME, PSO_ALGORITHM_NAME



def render_solver_result(result: dict, cities: pd.DataFrame | None) -> None:
    st.subheader(f"Final Sonuc - {result['algorithm']}")
    if result["algorithm"] in (PSO_ALGORITHM_NAME, BO_ALGORITHM_NAME):
        _render_continuous_result(result)
    else:
        _render_tsp_result(result, cities)


def _render_continuous_result(result: dict) -> None:
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
            st.write(
                f"Son iterasyonda iyilesen parcacik: **{result.get('improved_particles_last_iter', 0)}**"
            )
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


def _render_tsp_result(result: dict, cities: pd.DataFrame | None) -> None:
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
            worse_acceptance = (result["worse_accepted_moves"] / result["completed_iterations"]) * 100
            st.write(f"Tamamlanan iterasyon: **{result['completed_iterations']}**")
            st.write(f"Kabul edilen hamle: **{result['accepted_moves']}** (%{acceptance:.2f})")
            st.write(f"Kotu ama kabul edilen: **{result['worse_accepted_moves']}** (%{worse_acceptance:.2f})")
            st.write(f"2-Opt yerel iyilestirme: **{result.get('local_search_hits', 0)}**")
            st.write(f"Reheat sayisi: **{result.get('reheat_count', 0)}**")
        elif result["algorithm"] == "Tabu Search Algoritmasi":
            aspiration_ratio = (result["aspiration_count"] / result["completed_iterations"]) * 100
            st.write(f"Tamamlanan iterasyon: **{result['completed_iterations']}**")
            st.write(f"Aday havuzu: **{result['candidate_pool_size']}**")
            st.write(f"Tabu tenure: **{result['tabu_tenure']}**")
            st.write(f"Aspiration kullanimi: **{result['aspiration_count']}** (%{aspiration_ratio:.2f})")
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
