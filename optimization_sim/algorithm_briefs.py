from __future__ import annotations

import streamlit as st

from .models import BO_ALGORITHM_NAME, CMA_ES_ALGORITHM_NAME, PSO_ALGORITHM_NAME


ALGORITHM_BRIEFS: dict[str, dict[str, str]] = {
    "Genetik Algoritma": {
        "what": "Genetik Algoritma (GA), doğal seçilim prensibini sayısal optimizasyona taşır. Çözüm adayları bir popülasyon olarak başlar; iyi bireyler seçilir, çaprazlama ve mutasyon ile yeni nesiller üretilir. Böylece arama tek bir noktaya sıkışmaz ve geniş uzay kontrollü biçimde taranır.",
        "areas": "Rota optimizasyonu (TSP/VRP), çizelgeleme, tasarım optimizasyonu ve hiperparametre aramalarında yaygındır. Kısıtların yoğun olduğu ve analitik çözümün zor olduğu gerçek üretim/saha problemlerinde güçlüdür.",
        "space": "Ayrık kombinatoryal uzayda (permütasyon/sıralama) doğrudan; sürekli uzayda gerçek değer kodlamasıyla çalışır.",
        "params": "population_size, generations, crossover_rate, mutation_rate, elitism, selection_method, tournament_size, crossover_method, mutation_operator",
        "param_effects": (
            "- `selection_method (Turnuva/Rulet)`: Turnuva, güçlü bireyleri daha sık seçerek yakınsamayı hızlandırır; Rulet, olasılıksal seçimle daha zayıf ama umut vadeden bireyleri oyunda tutar ve erken tek tipleşmeyi azaltır.\n"
            "- `tournament_size`: Boyut arttıkça seçilim baskısı artar; hızlı ilerleme sağlar ama çeşitlilik düşebilir.\n"
            "- `crossover_rate`: Yüksek değer, iyi parçaların kombinasyonunu artırır; çok yüksek olursa stabil yapıların korunması zorlaşabilir.\n"
            "- `mutation_rate`: Yerel minimumdan kaçış ve çeşitlilik için kritiktir; düşükse arama körleşir, aşırı yüksekse çözüm rastgeleleşir.\n"
            "- `elitism`: En iyi bireyleri garanti ederek kalite kaybını önler; fazla elitizm çeşitliliği baskılar."
        ),
    },
    "Tavlama Algoritmasi": {
        "what": "Tavlama Algoritması (SA), metal tavlama sürecini taklit eder. Başta yüksek sıcaklıkta kötü hamleleri de belli olasılıkla kabul ederek yerel minimumdan kaçar; sıcaklık düştükçe daha seçici davranır.",
        "areas": "TSP, çizelgeleme, yerleşim (layout) ve parametre tuning problemlerinde tercih edilir. Az parametreyle güçlü sonuç vermesi nedeniyle pratikte çok kullanılır.",
        "space": "Ayrık komşuluk uzayı (swap, inversion, 2-opt) ve sürekli pertürbasyon uzayı.",
        "params": "initial_temperature, cooling_rate, min_temperature, iterations, neighbor_operator, two_opt_every, stagnation_limit, reheat_ratio",
        "param_effects": (
            "- `initial_temperature`: Yüksek başlangıç, daha fazla keşif sağlar; çok düşük değer erken kilitlenmeye yol açar.\n"
            "- `cooling_rate`: 1'e yakın değer daha yavaş soğur ve çözüm kalitesini artırabilir; çok agresif soğuma fırsatları kaçırır.\n"
            "- `neighbor_operator`: Aramanın hamle karakterini belirler; ör. `2-opt` rota problemlerinde büyük kalite katkısı verir.\n"
            "- `stagnation_limit + reheat_ratio`: Uzun süre iyileşme yoksa sistemi yeniden ısıtarak aramayı canlandırır."
        ),
    },
    "Tabu Search Algoritmasi": {
        "what": "Tabu Search, yerel aramayı kısa dönem hafıza ile güçlendirir. Son hamleleri tabu listesine alarak algoritmanın aynı döngülere geri dönmesini engeller.",
        "areas": "Lojistik, rota planlama, atama ve çizelgeleme gibi ayrık karar problemlerinde etkilidir.",
        "space": "Özellikle ayrık kombinatoryal uzay.",
        "params": "iterations, candidate_pool_size, tabu_tenure, aspiration_enabled, stagnation_limit, kick_ratio",
        "param_effects": (
            "- `tabu_tenure`: Tabu süresi uzadıkça döngü riski azalır; çok uzun olursa iyi hamleler gereksiz engellenebilir.\n"
            "- `candidate_pool_size`: Büyük havuz daha iyi hamle bulma şansını artırır ama hesap maliyetini yükseltir.\n"
            "- `aspiration_enabled`: Tabu hamle global iyileştirme getiriyorsa yine de kabul eder; kalite için kritik emniyet valfidir.\n"
            "- `kick_ratio`: Sıkışma anında çözümü sarsarak yeni bölgeleri keşfetmeye zorlar."
        ),
    },
    "Karinca Kolonisi Algoritmasi": {
        "what": "Karınca Kolonisi Algoritması (ACO), karıncaların feromonla yol güçlendirme davranışını modelleyen kolektif öğrenme yaklaşımıdır. İyi yollar zamanla güçlenir, zayıf yollar buharlaşma ile etkisini kaybeder.",
        "areas": "TSP/VRP, robotik yol planlama ve dinamik ağ yönlendirmede güçlüdür. Örneğin 5G/6G ve VANET ortamlarında trafik yükü saniyeler içinde değişirken ACO, feromon benzeri ağırlıklarla paketleri düşük yoğunluklu düğümlere hızla yönlendirerek statik kurallara göre daha iyi adaptasyon sağlar.",
        "space": "Graf tabanlı ayrık yol uzayı.",
        "params": "ant_count, alpha, beta, evaporation_rate, pheromone_constant, elitist_weight, candidate_k, two_opt_every",
        "param_effects": (
            "- `alpha` (feromon etkisi): Geçmişte iyi bulunan yolları ne kadar güçlü takip edeceğini belirler.\n"
            "- `beta` (sezgisel etki): Anlık maliyet bilgisinin (ör. mesafe/yoğunluk) karar üzerindeki ağırlığını ayarlar.\n"
            "- `evaporation_rate`: Yüksek buharlaşma hızlı adaptasyon sağlar; düşük buharlaşma hafızayı korur.\n"
            "- `elitist_weight`: En iyi turun feromon takviyesini artırarak yakınsamayı hızlandırır.\n"
            "- `candidate_k`: Komşu adaylarını sınırlayarak hızı artırır; çok dar tutulursa keşif azalabilir."
        ),
    },
    PSO_ALGORITHM_NAME: {
        "what": "Parçacık Sürü Optimizasyonu (PSO), parçacıkların bireysel en iyi (`pBest`) ve sürü en iyi (`gBest`) bilgisine göre hız/konum güncellediği sürekli optimizasyon yöntemidir.",
        "areas": "Dijital ikiz kalibrasyonu bunun tipik örneğidir: gerçek motor sensör verisi ile simülasyon çıktısı arasındaki hata hızlıca minimize edilir. Ayrıca kontrol parametresi ayarı, fonksiyon minimizasyonu ve ML tuningde yaygındır.",
        "space": "Çok boyutlu sürekli arama uzayı.",
        "params": "swarm_size, iterations, inertia_weight, cognitive_coeff, social_coeff, velocity_clamp_ratio",
        "param_effects": (
            "- `inertia_weight (w)`: Yüksekse keşif artar, düşükse sömürü artar; denge yakınsamayı belirler.\n"
            "- `cognitive_coeff (c1)`: Parçacığın kendi deneyimine bağlı hareketini güçlendirir.\n"
            "- `social_coeff (c2)`: Sürü bilgisini takip eğilimini artırır; çok yüksekse erken tek noktaya çökme riski doğar.\n"
            "- `velocity_clamp_ratio`: Adım büyüklüğünü sınırlayarak taşma/kararsızlığı önler."
        ),
    },
    BO_ALGORITHM_NAME: {
        "what": "Bayesian Optimization (BO), pahalı amaç fonksiyonlarında her denemeden maksimum bilgi çıkarmayı hedefler. Surrogate model (çoğunlukla GP) kurar ve acquisition fonksiyonuyla bir sonraki noktayı seçer.",
        "areas": "Pahalı simülasyon/deney optimizasyonu, hiperparametre seçimi, endüstriyel kalibrasyon ve laboratuvar deney planlamasında çok etkilidir.",
        "space": "Genellikle düşük/orta boyutlu sürekli uzay.",
        "params": "n_initial, n_iterations, kernel_type, acquisition_type, kappa, xi",
        "param_effects": (
            "- `kernel_type`: Fonksiyonun pürüzlülük varsayımını belirler; modelin genelleme davranışını doğrudan etkiler.\n"
            "- `acquisition_type` (`EI/UCB/PI`): Keşif-sömürü stratejisinin karakterini belirler.\n"
            "- `kappa` (UCB): Belirsizlik ödülünü artırarak daha fazla keşif yaptırır.\n"
            "- `xi` (EI/PI): İyileşme eşiğini ayarlayarak daha temkinli ya da daha agresif arama sağlar.\n"
            "- `n_initial`: Başlangıç örneklerinin çeşitliliğini artırır; çok düşükse model yanlı başlayabilir."
        ),
    },
    CMA_ES_ALGORITHM_NAME: {
        "what": "CMA-ES, arama dağılımının ortalamasını ve kovaryansını adaptif biçimde güncelleyerek zor sürekli problemlerde güçlü performans veren modern bir evrimsel yöntemdir.",
        "areas": "Eğrisel, pürüzlü ve gürültülü maliyet yüzeylerinde; model kalibrasyonu, mühendislik tasarımı ve hiperparametre ayarında sık kullanılır.",
        "space": "Sürekli, çok boyutlu arama uzayı.",
        "params": "population_size (lambda), iterations, initial_sigma, sigma_decay, elite_ratio",
        "param_effects": (
            "- `population_size`: Büyük popülasyon daha kararlı istatistik ve daha iyi keşif sağlar; maliyeti artırır.\n"
            "- `initial_sigma`: Başlangıç arama adım boyudur; büyükse geniş keşif, küçükse hızlı yerel odaklanma sağlar.\n"
            "- `sigma_decay`: Adım boyunun iterasyonla küçülme hızını belirler; 1'e yakın değer daha uzun keşif sunar.\n"
            "- `elite_ratio`: Ortalamanın güncellenmesinde kaç iyi örneğin etkili olacağını belirler; yüksek oran stabil, düşük oran agresif davranır."
        ),
    },
}


def render_algorithm_brief(algorithm_name: str) -> None:
    brief = ALGORITHM_BRIEFS.get(algorithm_name)
    if brief is None:
        return

    with st.expander("Algoritma Künyesi ve Uygulama Alanları", expanded=True):
        st.markdown(f"**Algoritma nedir?**  \n{brief['what']}")
        st.markdown(f"**Nerede kullanılır?**  \n{brief['areas']}")
        st.markdown(f"**Hangi uzayda çalışır?**  \n{brief['space']}")
        st.markdown(f"**Temel parametreler**  \n`{brief['params']}`")
        st.markdown(f"**Parametrelerin etkisi (neden önemli?)**  \n{brief['param_effects']}")
