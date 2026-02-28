from __future__ import annotations

import streamlit as st

from .models import (
    ACOConfig,
    BO_ALGORITHM_NAME,
    BOConfig,
    CMA_ES_ALGORITHM_NAME,
    CMAESConfig,
    GAConfig,
    PSO_ALGORITHM_NAME,
    PSO_PROBLEM_LABELS,
    PSOConfig,
    SAConfig,
    TabuConfig,
)

def configure_sidebar() -> tuple[str, GAConfig | SAConfig | TabuConfig | ACOConfig | PSOConfig | BOConfig | CMAESConfig, bool, bool]:
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
            CMA_ES_ALGORITHM_NAME,
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
    elif algorithm == BO_ALGORITHM_NAME:
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
    else:
        st.sidebar.header("CMA-ES Parametreleri")
        problem_name = st.sidebar.selectbox("Benchmark fonksiyon", PSO_PROBLEM_LABELS, index=1, key="cma_problem")
        st.sidebar.caption("CMA-ES, cok modlu ve zor surekli fonksiyonlarda guclu bir modern yontemdir.")

        population_size = st.sidebar.slider("Populasyon boyutu (lambda)", 6, 120, 24, step=2, key="cma_pop")
        iterations_slider = st.sidebar.slider(
            "Iterasyon sayisi (slider)", 10, 3000, 300, step=10, key="cma_iterations_slider"
        )
        iterations_manual = st.sidebar.number_input(
            "Iterasyon sayisi (manuel)",
            min_value=10,
            max_value=50000,
            value=int(iterations_slider),
            step=10,
            key="cma_iterations_manual",
        )
        iterations = int(iterations_manual)
        initial_sigma = st.sidebar.slider("Baslangic adim boyu (sigma)", 0.01, 5.0, 1.20, step=0.01, key="cma_sigma")
        sigma_decay = st.sidebar.slider("Sigma decay", 0.900, 1.000, 0.995, step=0.001, key="cma_decay")
        elite_ratio = st.sidebar.slider("Elit oran (mu/lambda)", 0.10, 0.80, 0.45, step=0.01, key="cma_elite")
        with st.sidebar.expander("Canli Akis Ayarlari", expanded=True):
            route_update_every = st.slider("Gorsel guncelleme (iterasyonda bir)", 1, 50, 5, key="cma_route_upd")
            analytics_update_every = st.slider("Grafik guncelleme (iterasyonda bir)", 1, 100, 10, key="cma_analytics_upd")
            frame_delay = st.slider("Kare gecikmesi (sn)", 0.0, 1.00, 0.05, step=0.01, key="cma_delay")
        random_seed = st.sidebar.number_input(
            "Rastgele tohum", min_value=0, max_value=999999, value=42, key="cma_seed"
        )
        config = CMAESConfig(
            problem_name=problem_name,
            population_size=population_size,
            iterations=iterations,
            initial_sigma=initial_sigma,
            sigma_decay=sigma_decay,
            elite_ratio=elite_ratio,
            route_update_every=route_update_every,
            analytics_update_every=analytics_update_every,
            frame_delay=frame_delay,
            random_seed=int(random_seed),
        )

    return algorithm, config, run_button, clear_button

