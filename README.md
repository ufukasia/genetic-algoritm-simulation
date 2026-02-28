# Optimization Algorithms Simulation (Streamlit)

Bu proje, Turkiye'deki 81 il icin gezgin satici problemini (TSP) birden fazla yontemle (Genetik Algoritma, Tavlama, Tabu Search, Karinca Kolonisi) cozen ve sureci canli gosteren bir Streamlit uygulamasidir.

## Proje yapisi

- `streamlit_app.py`: Ana Streamlit uygulamasi
- `81il.csv`: Sehir koordinat veri seti
- `requirements.txt`: Streamlit Cloud ve lokal kurulum bagimliliklari
- `runtime.txt`: Streamlit Cloud Python surumu (`python-3.11`)

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
