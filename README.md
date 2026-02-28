# Optimization Algorithms Simulation (Streamlit)

Bu proje, Turkiye'deki 81 il icin gezgin satici problemini (TSP) birden fazla yontemle (Genetik Algoritma, Tavlama, Tabu Search, Karinca Kolonisi) cozen ve sureci canli gosteren bir Streamlit uygulamasidir.

## Proje yapisi

- `streamlit_app.py`: Ince giris noktasi (sadece `optimization_sim.app.main` cagirir)
- `optimization_sim/models.py`: Konfig dataclass'lari ve sabitler
- `optimization_sim/problems.py`: PSO/BO benchmark problem tanimlari
- `optimization_sim/data.py`: Sehir verisi yukleme ve mesafe matrisi
- `optimization_sim/operators.py`: TSP operatorleri (crossover, mutasyon, 2-opt vb.)
- `optimization_sim/visualizations.py`: Tum grafik ve canli panel uretecileri
- `optimization_sim/algorithms.py`: GA/SA/Tabu/ACO/PSO/BO calistiricilari
- `optimization_sim/sidebar.py`: Sidebar konfigurasyonu
- `optimization_sim/registry.py`: Algoritma kayit/yonlendirme katmani
- `optimization_sim/results.py`: Final sonuc render katmani
- `optimization_sim/app.py`: Ana uygulama orkestrasyonu
- `81il.csv`: Sehir koordinat veri seti
- `requirements.txt`: Streamlit Cloud ve lokal kurulum bagimliliklari
- `runtime.txt`: Streamlit Cloud Python surumu (`python-3.11`)

## Yeni algoritma ekleme (moduler akis)

1. `optimization_sim/models.py` icine yeni algoritmanin config dataclass'ini ekle.
2. `optimization_sim/sidebar.py` icinde algoritma secimi ve parametre UI bolumunu ekle.
3. `optimization_sim/algorithms.py` icinde `run_*` fonksiyonunu yaz.
4. `optimization_sim/registry.py` icinde `ALGORITHM_SPECS` tablosuna yeni kayit ekle.
5. Gerekirse `optimization_sim/results.py` icinde final sonuc gosterimini genislet.

## Algoritma sozlesmesi (zorunlu)

- Her algoritma tek bir `run_*` fonksiyonu ve tek bir `*Config` dataclass ile tanimlanir.
- Aktivasyon sadece `optimization_sim/registry.py` icindeki `ALGORITHM_SPECS` kaydi ile yapilir.
- Tum runner ciktilari registry tarafinda ortak anahtarlara normalize edilir:
  - `best`
  - `history`
  - `completed_iterations`
  - `random_seed`
  - `benchmark`
- Karsilastirma tutarliligi icin seed zorunludur: `42`.
- Surekli optimizasyon benchmark seti sabittir: `PSO_PROBLEM_LABELS` (Schwefel, Ackley, Rastrigin, Rosenbrock, Levy).

## Lokal calistirma

1. Sanal ortam olustur:
```bash
python -m venv .venv
```

2. Sanal ortami aktif et:
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

3. Bagimliliklari kur:
```bash
pip install -r requirements.txt
```

4. Uygulamayi calistir:
```bash
python -m streamlit run streamlit_app.py
```

## Streamlit Cloud deploy

1. Bu repoyu GitHub'a push et.
2. Streamlit Cloud'da `New app` sec.
3. Repo/branch secimini yap.
4. `Main file path` olarak `streamlit_app.py` gir.
5. Deploy et.

Notlar:
- `requirements.txt` ve `runtime.txt` kok dizinde oldugu icin Streamlit Cloud otomatik algilar.
- `81il.csv` dosyasi uygulama ile ayni klasorde kalmalidir.

## Kisa teknik not

- Kullanilan ana kutuphaneler: `streamlit`, `numpy`, `pandas`, `plotly`
- Uygulama herhangi bir gizli anahtar/secrets gerektirmez.
