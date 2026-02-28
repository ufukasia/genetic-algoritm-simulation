from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .models import EARTH_RADIUS_KM, TR_ASCII

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
