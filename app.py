"""
================================================================
JakU - Dashboard Kualitas Udara DKI Jakarta
================================================================
Versi 2: redesign mengikuti spesifikasi mockup Figma.

Perbaikan utama:
- Layout grid presisi (max 1400px)
- Sidebar modern + active state biru
- Typography Plus Jakarta Sans / Inter
- Card border-radius 20px + soft shadow
- Hero ISPU dengan angka raksasa
- Map rounded container + legend rapi
- Navigation seamless via session_state
- Tombol "Lihat Selengkapnya" routing ke Detail Wilayah
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
import joblib

# ================================================================
# KONFIGURASI HALAMAN
# ================================================================
st.set_page_config(
    page_title="JakU - Dashboard Kualitas Udara",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# KONSTANTA
# ================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

PAGES = ["Dashboard", "Detail Wilayah", "Simulasi Prediksi ISPU", "Edukasi & Insight"]
PAGE_ICONS = ["grid-1x2", "geo-alt", "bar-chart-line", "book"]

KATEGORI_INFO = {
    "Baik": {
        "warna": "#16A34A", "warna_bg": "#DCFCE7", "emoji": "😊",
        "rentang": "0 - 50",
        "deskripsi": "Udara bersih, aman untuk beraktivitas sehari-hari.",
        "rekomendasi": "Cocok untuk berolahraga, jalan kaki, dan kegiatan outdoor lainnya."
    },
    "Sedang": {
        "warna": "#2563EB", "warna_bg": "#DBEAFE", "emoji": "😐",
        "rentang": "51 - 100",
        "deskripsi": "udara masih dapat diterima untuk beraktivitas di luar ruangan.",
        "rekomendasi": "Aman untuk beraktivitas di luar ruangan. Cocok untuk berolahraga, jalan kaki, dan kegiatan outdoor lainnya."
    },
    "Tidak Sehat": {
        "warna": "#F59E0B", "warna_bg": "#FEF3C7", "emoji": "😷",
        "rentang": "101 - 200",
        "deskripsi": "Kurangi aktivitas luar ruangan, terutama bagi kelompok sensitif.",
        "rekomendasi": "Kurangi aktivitas di luar ruangan. Gunakan masker jika harus keluar."
    },
    "Sangat Tidak Sehat": {
        "warna": "#EF4444", "warna_bg": "#FEE2E2", "emoji": "🤢",
        "rentang": "201 - 300",
        "deskripsi": "Hindari aktivitas luar ruangan. Gunakan masker jika harus keluar.",
        "rekomendasi": "Hindari semua aktivitas luar ruangan. Pakai masker N95 jika terpaksa keluar."
    },
    "Berbahaya": {
        "warna": "#7C3AED", "warna_bg": "#EDE9FE", "emoji": "☠️",
        "rentang": "≥ 301",
        "deskripsi": "Hindari semua aktivitas luar ruangan. Tetap di dalam ruangan.",
        "rekomendasi": "Tetap di dalam ruangan. Gunakan air purifier jika tersedia."
    },
}

INFO_POLUTAN = {
    "PM2.5": {
        "warna": "#2563EB", "satuan": "µg/m³",
        "deskripsi_pendek": "Partikel sangat halus berukuran ≤ 2.5 mikron",
        "deskripsi": "Partikel sangat halus yang dapat masuk jauh ke dalam paru-paru dan aliran darah."
    },
    "PM10": {
        "warna": "#60A5FA", "satuan": "µg/m³",
        "deskripsi_pendek": "Partikel halus berukuran ≤ 10 mikron",
        "deskripsi": "Partikel halus yang dapat masuk ke saluran pernapasan bagian atas dan menyebabkan iritasi."
    },
    "NO₂": {
        "warna": "#8B5CF6", "satuan": "µg/m³",
        "deskripsi_pendek": "Nitrogen dioksida, gas hasil pembakaran",
        "deskripsi": "Gas hasil pembakaran kendaraan bermotor dan industri, dapat mengiritasi paru-paru."
    },
    "SO₂": {
        "warna": "#F59E0B", "satuan": "µg/m³",
        "deskripsi_pendek": "Sulfur dioksida, gas dari pembakaran bahan bakar fosil",
        "deskripsi": "Gas dari pembakaran bahan bakar fosil, dapat menyebabkan iritasi mata dan saluran pernapasan."
    },
    "CO": {
        "warna": "#10B981", "satuan": "mg/m³",
        "deskripsi_pendek": "Karbon monoksida, gas tidak berwarna dan tidak berbau",
        "deskripsi": "Gas tidak berwarna dan tidak berbau yang dapat mengganggu pasokan oksigen dalam tubuh."
    },
    "O₃": {
        "warna": "#06B6D4", "satuan": "µg/m³",
        "deskripsi_pendek": "Ozon, terbentuk dari reaksi kimia di atmosfer",
        "deskripsi": "Ozon terbentuk dari reaksi kimia polutan dengan sinar matahari, dapat menyebabkan sesak napas."
    },
}


# ================================================================
# CUSTOM CSS - FULL OVERRIDE
# ================================================================
def inject_css():
    st.markdown("""
    <style>
    /* ============================================================
       FONTS
       ============================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, .main, .block-container,
    button, input, textarea, select {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ============================================================
       GLOBAL LAYOUT
       ============================================================ */
    .stApp {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1480px !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    #MainMenu, footer, .stDeployButton {visibility: hidden;}

    /* Scrollbar halus */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

    /* ============================================================
       SIDEBAR
       ============================================================ */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #EEF2F7;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display: none; }

    .sidebar-logo {
        text-align: center;
        padding: 0.25rem 1rem 0.1rem 1rem;
    }
    .sidebar-subtitle {
        text-align: center;
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 1.75rem;
        letter-spacing: 0.01em;
    }
    .sidebar-footer {
        background-color: #F8FAFC;
        border: 1px solid #EEF2F7;
        border-radius: 14px;
        padding: 0.95rem 1.1rem;
        margin: 1rem 0.6rem;
    }
    .sidebar-footer-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.4rem;
    }
    .sidebar-footer-desc {
        font-size: 0.72rem;
        color: #64748B;
        line-height: 1.5;
        margin-bottom: 0.65rem;
    }
    .sidebar-footer-ts-label {
        font-size: 0.7rem;
        color: #94A3B8;
        margin-bottom: 0.15rem;
    }
    .sidebar-footer-ts {
        font-size: 0.78rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* ============================================================
       PAGE HEADER
       ============================================================ */
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.3rem;
        letter-spacing: -0.025em;
        line-height: 1.15;
    }
    .page-subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.75rem;
        font-weight: 400;
    }
    .updated-card {
        background-color: #FFFFFF;
        border: 1px solid #EEF2F7;
        border-radius: 14px;
        padding: 0.85rem 1.25rem;
        display: inline-flex;
        align-items: center;
        gap: 0.7rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .updated-card-icon {
        font-size: 1.15rem;
    }
    .updated-card-label {
        font-size: 0.72rem;
        color: #64748B;
        font-weight: 500;
        line-height: 1.2;
    }
    .updated-card-value {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }

    /* ============================================================
       CARD
       ============================================================ */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #EEF2F7;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: all 0.25s ease;
        height: 100%;
    }
    .card:hover {
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        transform: translateY(-1px);
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 1rem;
        letter-spacing: -0.005em;
    }

    /* ============================================================
       HERO ISPU
       ============================================================ */
    .hero-row {
        display: flex;
        align-items: flex-start;
        gap: 1.5rem;
        margin-top: 0.25rem;
    }
    .hero-number {
        font-size: 5.5rem;
        font-weight: 800;
        line-height: 0.95;
        letter-spacing: -0.05em;
        color: #2563EB;
    }
    .hero-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #64748B;
        margin-top: 0.3rem;
    }
    .hero-emoji {
        font-size: 2.5rem;
        line-height: 1;
        margin-bottom: 0.4rem;
    }
    .hero-status {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        letter-spacing: -0.01em;
    }
    .hero-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.55;
        max-width: 22rem;
    }
    .hero-illustration {
        text-align: center;
        padding: 0.5rem;
        margin-left: auto;
    }

    /* Polutan dominan strip */
    .dom-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 1.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #F1F5F9;
    }
    .dom-strip-left {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.92rem;
        color: #0F172A;
    }
    .dom-strip-icon { color: #16A34A; font-size: 1rem; }

    /* Pollutant pills grid (6 polutan) */
    .pollutant-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.5rem;
        margin-top: 1.2rem;
    }
    .pollutant-cell {
        text-align: center;
        padding: 0.25rem 0;
    }
    .pollutant-name {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 0.3rem;
        letter-spacing: 0.01em;
    }
    .pollutant-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .pollutant-unit {
        font-size: 0.7rem;
        color: #94A3B8;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* ============================================================
       MAP CONTAINER
       ============================================================ */
    .map-wrapper {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #EEF2F7;
    }
    iframe { border-radius: 14px; }

    .legend-block {
        padding: 0.25rem 0 0 0.5rem;
    }
    .legend-title {
        font-weight: 700;
        font-size: 0.9rem;
        color: #0F172A;
        margin-bottom: 0.7rem;
    }
    .legend-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0.4rem 0;
        font-size: 0.83rem;
        color: #334155;
    }
    .legend-dot {
        width: 0.7rem;
        height: 0.7rem;
        border-radius: 999px;
        flex-shrink: 0;
        box-shadow: 0 0 0 3px rgba(255,255,255,1), 0 0 0 4px rgba(15,23,42,0.06);
    }

    /* ============================================================
       PREDIKSI LIST
       ============================================================ */
    .pred-row {
        display: grid;
        grid-template-columns: 1.1fr 0.8fr 1fr 0.9fr;
        align-items: center;
        gap: 0.8rem;
        padding: 0.6rem 0;
        border-bottom: 1px solid #F1F5F9;
    }
    .pred-row:last-child { border-bottom: none; }
    .pred-date {
        font-size: 0.88rem;
        color: #334155;
        font-weight: 500;
    }
    .pred-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        min-width: 3rem;
    }
    .pred-cat {
        font-size: 0.86rem;
        font-weight: 600;
    }
    .pred-pm {
        font-size: 0.82rem;
        color: #64748B;
        text-align: right;
    }

    /* ============================================================
       REKOMENDASI CARD
       ============================================================ */
    .rekom-card {
        background-color: #FFFFFF;
        border: 1px solid #EEF2F7;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        display: flex;
        gap: 0.9rem;
        align-items: flex-start;
        transition: all 0.25s ease;
        height: 100%;
    }
    .rekom-card:hover {
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        transform: translateY(-2px);
        border-color: #DBEAFE;
    }
    .rekom-icon {
        font-size: 2rem;
        flex-shrink: 0;
        line-height: 1;
    }
    .rekom-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .rekom-desc {
        font-size: 0.78rem;
        color: #64748B;
        line-height: 1.5;
    }

    /* ============================================================
       INFO BOX ML
       ============================================================ */
    .info-box {
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 14px;
        padding: 0.95rem 1.25rem;
        display: flex;
        gap: 0.7rem;
        align-items: flex-start;
        margin: 1.25rem 0;
    }
    .info-box-icon {
        color: #2563EB;
        font-size: 1.1rem;
        line-height: 1.4;
        flex-shrink: 0;
    }
    .info-box-text {
        font-size: 0.88rem;
        color: #1E40AF;
        line-height: 1.55;
    }

    /* ============================================================
       BUTTONS — full override
       ============================================================ */
    .stButton > button {
        border-radius: 999px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        font-size: 0.88rem;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
        color: #2563EB;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
        letter-spacing: 0.005em;
    }
    .stButton > button:hover {
        background: #F0F7FF;
        border-color: #2563EB;
        color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12);
    }
    .stButton > button[kind="primary"] {
        background-color: #2563EB;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.28);
    }

    /* ============================================================
       TABS / SLIDER overrides (untuk halaman lain)
       ============================================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 999px;
        padding: 0.5rem 1.1rem;
        font-weight: 600;
        color: #64748B;
        font-size: 0.88rem;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #BFDBFE;
        color: #2563EB;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE !important;
        color: #2563EB !important;
        border-color: #BFDBFE !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #2563EB;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
    }

    /* ============================================================
       SIMULASI PREDIKSI (step bar, hasil)
       ============================================================ */
    .step-bar {
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        display: grid;
        grid-template-columns: auto repeat(3, 1fr);
        gap: 1.5rem;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    .step-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        color: #2563EB;
        font-size: 0.95rem;
    }
    .step-item {
        display: flex;
        gap: 0.65rem;
        align-items: flex-start;
    }
    .step-num {
        background: #FFFFFF;
        border: 1px solid #DBEAFE;
        color: #2563EB;
        width: 1.7rem;
        height: 1.7rem;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .step-text {
        font-size: 0.85rem;
        color: #1E40AF;
        line-height: 1.45;
    }
    .hasil-hero {
        display: flex;
        align-items: flex-start;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .hasil-num {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        color: #2563EB;
        letter-spacing: -0.04em;
    }
    .hasil-label-ispu {
        font-size: 0.95rem;
        color: #64748B;
        font-weight: 600;
        text-align: center;
    }
    .rekom-box {
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
    }
    .rekom-box-title {
        font-size: 1rem;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 0.4rem;
    }
    .rekom-box-text {
        font-size: 0.86rem;
        color: #1E40AF;
        line-height: 1.5;
    }

    /* ============================================================
       EDUKASI - kategori card
       ============================================================ */
    .kat-card {
        border-radius: 16px;
        padding: 1.3rem 1.1rem;
        height: 100%;
        border: 1px solid;
    }
    .kat-range {
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .kat-emoji { font-size: 1.7rem; }
    .kat-name {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.85rem;
        margin-bottom: 0.4rem;
    }
    .kat-desc {
        font-size: 0.78rem;
        color: #334155;
        line-height: 1.45;
    }
    .donut-legend-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        font-size: 0.88rem;
    }
    .donut-legend-left {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #0F172A;
    }
    .donut-legend-dot {
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 999px;
    }
    .donut-legend-pct {
        font-weight: 700;
        color: #0F172A;
    }

    /* ============================================================
       RESPONSIVE
       ============================================================ */
    @media (max-width: 992px) {
        .hero-number { font-size: 4.2rem; }
        .pollutant-grid { grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }
        .step-bar { grid-template-columns: 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# UTILITIES
# ================================================================
@st.cache_data
def load_data():
    return {
        "ispu":     pd.read_csv(DATA_DIR / "ispu_dummy.csv"),
        "wilayah":  pd.read_csv(DATA_DIR / "wilayah_dummy.csv"),
        "prediksi": pd.read_csv(DATA_DIR / "prediksi_dummy.csv"),
        "edukasi":  pd.read_csv(DATA_DIR / "edukasi_dummy.csv"),
    }


@st.cache_resource
def load_model():
    try:
        return {
            "model": joblib.load(MODELS_DIR / "model_xgboost.pkl"),
            "le":    joblib.load(MODELS_DIR / "label_encoder.pkl"),
            "fitur": joblib.load(MODELS_DIR / "fitur_polutan.pkl"),
            "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_logo_b64():
    p = ASSETS_DIR / "logo.svg"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def kategori_dari_ispu(ispu):
    if ispu <= 50:  return "Baik"
    if ispu <= 100: return "Sedang"
    if ispu <= 200: return "Tidak Sehat"
    if ispu <= 300: return "Sangat Tidak Sehat"
    return "Berbahaya"


def prediksi_ispu_xgboost(pm10, pm25, so2, co, o3, no2):
    art = load_model()
    if not art["ok"]:
        nilai = pm25 * 0.30 + pm10 * 0.20 + no2 * 0.15 + so2 * 0.15 + co * 0.10 + o3 * 0.10
        return {"kategori": kategori_dari_ispu(nilai), "nilai_ispu": int(round(nilai)),
                "confidence": None, "fallback": True}

    input_df = pd.DataFrame([{
        "pm_sepuluh": pm10, "pm_duakomalima": pm25, "sulfur_dioksida": so2,
        "karbon_monoksida": co, "ozon": o3, "nitrogen_dioksida": no2,
    }])[art["fitur"]]

    pred_idx = art["model"].predict(input_df)[0]
    kategori_xgb = art["le"].inverse_transform([pred_idx])[0]
    kat_map = {"BAIK": "Baik", "SEDANG": "Sedang", "TIDAK SEHAT": "Tidak Sehat"}
    kategori = kat_map.get(kategori_xgb, "Sedang")

    confidence = None
    try:
        proba = art["model"].predict_proba(input_df)[0]
        confidence = float(np.max(proba))
    except Exception:
        pass

    nilai = pm25 * 0.30 + pm10 * 0.20 + no2 * 0.15 + so2 * 0.15 + co * 0.10 + o3 * 0.10
    if kategori == "Baik":          nilai = min(nilai, 50)
    elif kategori == "Sedang":      nilai = max(51, min(nilai, 100))
    elif kategori == "Tidak Sehat": nilai = max(101, min(nilai, 200))
    return {"kategori": kategori, "nilai_ispu": int(round(nilai)),
            "confidence": confidence, "fallback": False}


def render_popup_polutan():
    @st.dialog("Informasi Polutan", width="large")
    def _popup():
        st.markdown("""
        <p style='color:#64748B; font-size:0.88rem; margin-bottom:1rem; margin-top:-0.5rem;'>
            Penjelasan singkat tiap polutan udara yang dipantau JakU.
        </p>
        """, unsafe_allow_html=True)
        items = list(INFO_POLUTAN.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2, gap="medium")
            for j, col in enumerate(cols):
                if i + j >= len(items): continue
                nama, info = items[i + j]
                with col:
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0;
                                border-radius:14px; padding:1rem 1.1rem;
                                height:100%; min-height:130px;">
                      <div style="font-weight:700; font-size:1rem; color:#0F172A; margin-bottom:0.45rem;">{nama}</div>
                      <div style="font-size:0.82rem; color:#475569; line-height:1.5;">{info["deskripsi"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
    _popup()


def navigate_to(page_name, **extra_state):
    """Pindah halaman + simpan state tambahan, lalu rerun."""
    if page_name not in PAGES:
        return
    st.session_state.current_page = page_name
    st.session_state.menu_key = st.session_state.get("menu_key", 0) + 1
    for k, v in extra_state.items():
        st.session_state[k] = v
    st.rerun()


# ================================================================
# SIDEBAR
# ================================================================
def render_sidebar():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "menu_key" not in st.session_state:
        st.session_state.menu_key = 0

    logo_b64 = get_logo_b64()

    with st.sidebar:
        if logo_b64:
            st.markdown(
                f"""
                <div class='sidebar-logo'>
                    <img src='data:image/svg+xml;base64,{logo_b64}' style='width:140px;' />
                </div>
                <div class='sidebar-subtitle'>Pantau Udara, Jaga Jakarta</div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                "<h2 style='text-align:center; color:#16A34A; margin-bottom:0;'>Jak<span style='color:#2563EB;'>U</span></h2>"
                "<div class='sidebar-subtitle'>Pantau Udara, Jaga Jakarta</div>",
                unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=PAGES,
            icons=PAGE_ICONS,
            default_index=PAGES.index(st.session_state.current_page),
            key=f"main_menu_{st.session_state.menu_key}",
            styles={
                "container": {"padding": "0.25rem 0.5rem", "background-color": "#FFFFFF"},
                "icon": {"font-size": "1.05rem"},
                "nav-link": {
                    "font-size": "0.93rem", "font-weight": "500", "color": "#475569",
                    "padding": "0.72rem 1rem", "margin": "0.2rem 0",
                    "border-radius": "12px", "--hover-color": "#F1F5F9",
                },
                "nav-link-selected": {
                    "background-color": "#DBEAFE", "color": "#2563EB", "font-weight": "600",
                },
            },
        )

        # Update state jika user klik menu manual
        if selected != st.session_state.current_page:
            st.session_state.current_page = selected
            st.rerun()

        st.markdown("<div style='flex:1; min-height:7rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='sidebar-footer'>
                <div class='sidebar-footer-title'>Data tidak realtime</div>
                <div class='sidebar-footer-desc'>
                    Data yang ditampilkan berdasarkan sampel dan diperbarui secara berkala.
                </div>
                <div class='sidebar-footer-ts-label'>Data terakhir diperbarui</div>
                <div class='sidebar-footer-ts'>26 Mei 2025, 10:00 WIB</div>
            </div>
            """, unsafe_allow_html=True)


# ================================================================
# HALAMAN 1: DASHBOARD (REDESIGN)
# ================================================================
def page_dashboard(data):
    # ─────────── HEADER ───────────
    h1, h2 = st.columns([3, 1.1])
    with h1:
        st.markdown(
            "<div class='page-title'>Halo, Selamat Datang di JakU!</div>"
            "<div class='page-subtitle'>Berikut ringkasan kualitas udara di Provinsi DKI Jakarta</div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            """
            <div style='display:flex; justify-content:flex-end; padding-top:0.4rem;'>
                <div class='updated-card'>
                    <div class='updated-card-icon'>📅</div>
                    <div>
                        <div class='updated-card-label'>Data terakhir diperbarui</div>
                        <div class='updated-card-value'>15 Juni 2024, 10:00 WIB</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─────────── ROW 1: HERO + MAP ───────────
    row1_left, row1_right = st.columns([1.2, 1], gap="medium")

    # ▸ Hero card
    with row1_left:
        ispu_val = 78
        kat = kategori_dari_ispu(ispu_val)
        info = KATEGORI_INFO[kat]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>Kualitas Udara di Jakarta Hari ini (Rata-rata)</div>",
            unsafe_allow_html=True,
        )

        # Hero row: angka besar | emoji+status+desc | illustrasi
        st.markdown(
            f"""
            <div class='hero-row'>
                <div>
                    <div class='hero-number' style='color:{info["warna"]};'>{ispu_val}</div>
                    <div class='hero-label'>ISPU</div>
                </div>
                <div style='flex:1; padding-top:0.6rem;'>
                    <div class='hero-emoji'>{info["emoji"]}</div>
                    <div class='hero-status' style='color:{info["warna"]};'>Udara {kat}</div>
                    <div class='hero-desc'>{info["deskripsi"]}</div>
                </div>
                <div class='hero-illustration'>
                    <div style='font-size:4.5rem; line-height:1;'>🏙️</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Polutan dominan strip + button popup
        dom_l, dom_r = st.columns([2, 1])
        with dom_l:
            st.markdown(
                """
                <div class='dom-strip' style='border-top:none; padding-top:1.2rem; margin-top:1.2rem;'>
                    <div class='dom-strip-left'>
                        <span class='dom-strip-icon'>🌿</span>
                        <span><strong>Polutan dominan:</strong>&nbsp; PM2.5 (24 µg/m³)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with dom_r:
            st.markdown("<div style='padding-top:1.4rem;'></div>", unsafe_allow_html=True)
            if st.button("ⓘ  Lihat penjelasan polutan", key="btn_info_dashboard",
                         use_container_width=True):
                render_popup_polutan()

        # Pollutant grid (6 polutan compact)
        st.markdown(
            """
            <div style='border-top:1px solid #F1F5F9; margin-top:0.4rem;'></div>
            <div class='pollutant-grid'>
              <div class='pollutant-cell'><div class='pollutant-name'>PM2.5</div><div class='pollutant-value'>24</div><div class='pollutant-unit'>µg/m³</div></div>
              <div class='pollutant-cell'><div class='pollutant-name'>PM10</div><div class='pollutant-value'>41</div><div class='pollutant-unit'>µg/m³</div></div>
              <div class='pollutant-cell'><div class='pollutant-name'>NO₂</div><div class='pollutant-value'>18</div><div class='pollutant-unit'>µg/m³</div></div>
              <div class='pollutant-cell'><div class='pollutant-name'>SO₂</div><div class='pollutant-value'>7</div><div class='pollutant-unit'>µg/m³</div></div>
              <div class='pollutant-cell'><div class='pollutant-name'>CO</div><div class='pollutant-value'>0.6</div><div class='pollutant-unit'>mg/m³</div></div>
              <div class='pollutant-cell'><div class='pollutant-name'>O₃</div><div class='pollutant-value'>50</div><div class='pollutant-unit'>µg/m³</div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ▸ Map card
    with row1_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>Kualitas Udara per Wilayah di Jakarta</div>",
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns([1.85, 1], gap="small")
        with m1:
            st.markdown("<div class='map-wrapper'>", unsafe_allow_html=True)
            m = folium.Map(
                location=[-6.2088, 106.8456],
                zoom_start=10,
                tiles="CartoDB positron",
                zoom_control=False,
                scrollWheelZoom=False,
                dragging=True,
            )
            for _, row in data["wilayah"].iterrows():
                kat_w = row["kategori"]
                warna = KATEGORI_INFO.get(kat_w, KATEGORI_INFO["Sedang"])["warna"]
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=24, color="white", weight=3,
                    fill=True, fillColor=warna, fillOpacity=0.95,
                    tooltip=f"{row['wilayah']}: {row['ispu']}",
                ).add_to(m)
                folium.map.Marker(
                    [row["lat"], row["lon"]],
                    icon=folium.DivIcon(
                        icon_size=(40, 40), icon_anchor=(20, 20),
                        html=f"<div style='font-size:12px; font-weight:800; color:white; text-align:center; line-height:40px;'>{row['ispu']}</div>",
                    ),
                ).add_to(m)
            st_folium(m, height=300, use_container_width=True, returned_objects=[])
            st.markdown("</div>", unsafe_allow_html=True)

        with m2:
            legend_html = "<div class='legend-block'><div class='legend-title'>Keterangan:</div>"
            for nama, info_kat in KATEGORI_INFO.items():
                legend_html += f"""
                <div class='legend-row'>
                    <div class='legend-dot' style='background:{info_kat["warna"]};'></div>
                    <span>{nama} ({info_kat["rentang"]})</span>
                </div>
                """
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            if st.button("Lihat Selengkapnya  →", key="btn_lihat_selengkapnya",
                         type="primary", use_container_width=True):
                navigate_to("Detail Wilayah")

        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────── ROW 2: PREDIKSI + TREN ───────────
    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
    row2_left, row2_right = st.columns([1, 1.5], gap="medium")

    # ▸ Prediksi 7 hari
    with row2_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>Prediksi ISPU di Jakarta (7 Hari Mendatang)</div>",
            unsafe_allow_html=True,
        )
        pred = data["prediksi"][data["prediksi"]["wilayah"] == "DKI Jakarta"]
        rows_html = ""
        for _, r in pred.iterrows():
            kat2 = r["kategori"]
            warna = KATEGORI_INFO.get(kat2, KATEGORI_INFO["Sedang"])["warna"]
            tgl = pd.to_datetime(r["tanggal"]).strftime("%d %b %Y")
            rows_html += f"""
            <div class='pred-row'>
                <div class='pred-date'>{tgl}</div>
                <div><span class='pred-pill' style='background:{warna};'>{r["ispu"]}</span></div>
                <div class='pred-cat' style='color:{warna};'>{kat2}</div>
                <div class='pred-pm'>{r["pm25"]} µg/m³</div>
            </div>
            """
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ▸ Tren chart 7 hari terakhir
    with row2_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>Tren ISPU di Jakarta (7 Hari Terakhir)</div>",
            unsafe_allow_html=True,
        )

        df = data["ispu"].copy()
        df["tanggal"] = pd.to_datetime(df["tanggal"])
        df["label_x"] = df["tanggal"].dt.strftime("%d %b")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["label_x"], y=df["ispu"],
            mode="lines+markers+text",
            text=df["ispu"], textposition="top center",
            textfont=dict(size=11, color="#0F172A", weight=600),
            line=dict(color="#2563EB", width=3, shape="spline", smoothing=1.0),
            marker=dict(size=9, color="#2563EB", line=dict(color="white", width=2)),
            fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
            hovertemplate="<b>%{x}</b><br>ISPU: %{y}<extra></extra>",
            showlegend=False,
        ))
        for nilai, label, warna in [
            (50, "Baik", "#16A34A"),
            (100, "Sedang", "#2563EB"),
            (200, "Tidak Sehat", "#F59E0B"),
            (300, "Sangat Tidak Sehat", "#EF4444"),
        ]:
            fig.add_hline(y=nilai, line_dash="dot", line_color="#E2E8F0", line_width=1)
            fig.add_annotation(
                x=1.0, xref="paper", y=nilai, text=label, showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(size=10, color=warna, weight=600), xshift=8,
            )

        fig.update_layout(
            height=320,
            margin=dict(l=20, r=130, t=20, b=20),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color="#64748B")),
            yaxis=dict(
                range=[0, 310], gridcolor="#F1F5F9", showline=False,
                tickfont=dict(size=11, color="#94A3B8"),
                tickvals=[0, 50, 100, 150, 200, 300],
            ),
            hoverlabel=dict(bgcolor="white", bordercolor="#E2E8F0",
                            font=dict(size=12, color="#0F172A", family="Plus Jakarta Sans")),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────── INFO BOX ML ───────────
    st.markdown(
        """
        <div class='info-box'>
            <div class='info-box-icon'>ⓘ</div>
            <div class='info-box-text'>
                Prediksi ini dibuat menggunakan model machine learning <strong>XGBoost</strong>
                berdasarkan data historis ISPU pada tahun 2024.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────── ROW 3: REKOMENDASI ───────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Rekomendasi Aktivitas</div>", unsafe_allow_html=True)

    rekomendasi = [
        ("🏃‍♀️", "Olahraga Luar Ruangan", "Aktivitas luar ruangan aman dilakukan."),
        ("😷",   "Gunakan Masker",         "Gunakan masker jika Anda sensitif terhadap polusi."),
        ("👵",   "Kelompok Sensitif",      "Jaga kesehatan dan hindari area dengan polusi tinggi."),
        ("🌳",   "Buka Jendela",           "Sirkulasi udara di dalam ruangan masih aman."),
    ]
    rc = st.columns(4, gap="medium")
    for col, (icon, judul, desc) in zip(rc, rekomendasi):
        with col:
            st.markdown(
                f"""
                <div class='rekom-card'>
                    <div class='rekom-icon'>{icon}</div>
                    <div>
                        <div class='rekom-title'>{judul}</div>
                        <div class='rekom-desc'>{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# HALAMAN 2: DETAIL WILAYAH
# ================================================================
def page_detail_wilayah(data):
    st.markdown(
        "<div class='page-title'>Detail Wilayah</div>"
        "<div class='page-subtitle'>Pilih wilayah untuk melihat informasi kualitas udara lebih detail.</div>",
        unsafe_allow_html=True,
    )

    wilayah_list = data["wilayah"]["wilayah"].tolist()

    # Default tab dari session_state (jika datang dari "Lihat Selengkapnya")
    default_idx = 0
    if "selected_region" in st.session_state and st.session_state.selected_region in wilayah_list:
        default_idx = wilayah_list.index(st.session_state.selected_region)
        # Reset agar tidak sticky
        del st.session_state["selected_region"]

    tabs = st.tabs(wilayah_list)
    for tab, wilayah in zip(tabs, wilayah_list):
        with tab:
            row = data["wilayah"][data["wilayah"]["wilayah"] == wilayah].iloc[0]
            kat = row["kategori"]
            info = KATEGORI_INFO[kat]

            c1, c2 = st.columns([1.1, 1], gap="medium")

            with c1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='card-title'>Kualitas Udara {wilayah}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class='hero-row'>
                        <div>
                            <div class='hero-number' style='color:{info["warna"]}; font-size:4.5rem;'>{row["ispu"]}</div>
                            <div class='hero-label'>ISPU</div>
                        </div>
                        <div style='flex:1; padding-top:0.6rem;'>
                            <div class='hero-emoji'>{info["emoji"]}</div>
                            <div class='hero-status' style='color:{info["warna"]};'>Udara {kat}</div>
                            <div class='hero-desc'>{info["deskripsi"]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True,
                )

                dl, dr = st.columns([2, 1])
                with dl:
                    st.markdown(
                        f"""
                        <div class='dom-strip' style='margin-top:1.2rem;'>
                            <div class='dom-strip-left'>
                                <span class='dom-strip-icon'>🌿</span>
                                <span><strong>Polutan dominan:</strong>&nbsp; PM2.5 ({row["pm25"]} µg/m³)</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True,
                    )
                with dr:
                    st.markdown("<div style='padding-top:1.4rem;'></div>", unsafe_allow_html=True)
                    if st.button("ⓘ  Lihat penjelasan polutan", key=f"btn_info_{wilayah}",
                                 use_container_width=True):
                        render_popup_polutan()

                st.markdown(
                    f"""
                    <div class='pollutant-grid'>
                      <div class='pollutant-cell'><div class='pollutant-name'>PM2.5</div><div class='pollutant-value'>{row["pm25"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>PM10</div><div class='pollutant-value'>{row["pm10"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>NO₂</div><div class='pollutant-value'>{row["no2"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>SO₂</div><div class='pollutant-value'>{row["so2"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>CO</div><div class='pollutant-value'>{row["co"]}</div><div class='pollutant-unit'>mg/m³</div></div>
                      <div class='pollutant-cell'><div class='pollutant-name'>O₃</div><div class='pollutant-value'>{row["o3"]}</div><div class='pollutant-unit'>µg/m³</div></div>
                    </div>
                    """, unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Rekomendasi Aktivitas</div>", unsafe_allow_html=True)
                rekom = [
                    ("🏃‍♀️", "Olahraga Luar Ruangan", "Aktivitas luar ruangan aman dilakukan."),
                    ("😷",   "Gunakan Masker",         "Gunakan masker jika Anda sensitif terhadap polusi."),
                    ("👵",   "Kelompok Sensitif",      "Jaga kesehatan dan hindari area dengan polusi tinggi."),
                    ("🌳",   "Buka Jendela",           "Sirkulasi udara di dalam ruangan masih aman."),
                ]
                gc1, gc2 = st.columns(2, gap="small")
                for idx, (icon, judul, desc) in enumerate(rekom):
                    with (gc1 if idx % 2 == 0 else gc2):
                        st.markdown(
                            f"""
                            <div class='rekom-card' style='margin-bottom:0.6rem;'>
                                <div class='rekom-icon'>{icon}</div>
                                <div>
                                    <div class='rekom-title'>{judul}</div>
                                    <div class='rekom-desc'>{desc}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
            pc1, pc2 = st.columns([1, 1.4], gap="medium")

            with pc1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='card-title'>Prediksi ISPU di {wilayah} (7 Hari Mendatang)</div>",
                    unsafe_allow_html=True,
                )
                pred_w = data["prediksi"][data["prediksi"]["wilayah"] == wilayah]
                rows_html = ""
                for _, r in pred_w.iterrows():
                    kat2 = r["kategori"]
                    warna = KATEGORI_INFO.get(kat2, KATEGORI_INFO["Sedang"])["warna"]
                    tgl = pd.to_datetime(r["tanggal"]).strftime("%d %b %Y")
                    rows_html += f"""
                    <div class='pred-row'>
                        <div class='pred-date'>{tgl}</div>
                        <div><span class='pred-pill' style='background:{warna};'>{r["ispu"]}</span></div>
                        <div class='pred-cat' style='color:{warna};'>{kat2}</div>
                        <div class='pred-pm'>{r["pm25"]} µg/m³</div>
                    </div>
                    """
                st.markdown(rows_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with pc2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='card-title'>Tren ISPU di {wilayah} (7 Hari Terakhir)</div>",
                    unsafe_allow_html=True,
                )
                df = data["ispu"].copy()
                df["tanggal"] = pd.to_datetime(df["tanggal"])
                np.random.seed(hash(wilayah) % 1000)
                df["ispu_w"] = (df["ispu"] + np.random.uniform(-15, 15, len(df))).clip(20, 250).round().astype(int)
                df["label_x"] = df["tanggal"].dt.strftime("%d %b")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["label_x"], y=df["ispu_w"],
                    mode="lines+markers+text",
                    text=df["ispu_w"], textposition="top center",
                    textfont=dict(size=11, color="#0F172A", weight=600),
                    line=dict(color="#2563EB", width=3, shape="spline", smoothing=1.0),
                    marker=dict(size=9, color="#2563EB", line=dict(color="white", width=2)),
                    fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
                    hovertemplate="<b>%{x}</b><br>ISPU: %{y}<extra></extra>",
                    showlegend=False,
                ))
                for nilai, label, warna in [
                    (50, "Baik", "#16A34A"), (100, "Sedang", "#2563EB"),
                    (200, "Tidak Sehat", "#F59E0B"), (300, "Sangat Tidak Sehat", "#EF4444"),
                ]:
                    fig.add_hline(y=nilai, line_dash="dot", line_color="#E2E8F0", line_width=1)
                    fig.add_annotation(
                        x=1.0, xref="paper", y=nilai, text=label, showarrow=False,
                        xanchor="left", yanchor="middle",
                        font=dict(size=10, color=warna, weight=600), xshift=8,
                    )
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=130, t=20, b=20),
                    paper_bgcolor="white", plot_bgcolor="white",
                    xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color="#64748B")),
                    yaxis=dict(range=[0, 310], gridcolor="#F1F5F9", showline=False,
                               tickfont=dict(size=11, color="#94A3B8"),
                               tickvals=[0, 50, 100, 150, 200, 300]),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
                <div class='info-box'>
                    <div class='info-box-icon'>ⓘ</div>
                    <div class='info-box-text'>
                        Prediksi ini dibuat menggunakan model machine learning <strong>XGBoost</strong>
                        berdasarkan data historis ISPU pada tahun 2024.
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ================================================================
# HALAMAN 3: SIMULASI
# ================================================================
def page_simulasi(data):
    st.markdown(
        "<div class='page-title'>Simulasi Prediksi ISPU</div>"
        "<div class='page-subtitle'>Simulasikan kualitas udara berdasarkan konsentrasi polutan.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='step-bar'>
            <div class='step-title'>ⓘ Cara Menggunakan Simulasi</div>
            <div class='step-item'><div class='step-num'>1</div><div class='step-text'>Masukkan nilai konsentrasi 6 polutan sesuai satuan yang tertera.</div></div>
            <div class='step-item'><div class='step-num'>2</div><div class='step-text'>Klik tombol "Submit Simulasi" untuk melihat hasil prediksi.</div></div>
            <div class='step-item'><div class='step-num'>3</div><div class='step-text'>Hasil prediksi menunjukkan kategori ISPU dan rekomendasi kesehatan.</div></div>
        </div>
        """, unsafe_allow_html=True)

    if "sim_values" not in st.session_state:
        st.session_state["sim_values"] = {"pm25": 50.0, "pm10": 70.0, "no2": 25.0,
                                          "so2": 35.0, "co": 1.5, "o3": 50.0}
    if "sim_hasil" not in st.session_state:
        st.session_state["sim_hasil"] = None

    def apply_preset(name):
        presets = {
            "Udara Bersih": {"pm25": 8.0, "pm10": 25.0, "no2": 15.0, "so2": 10.0, "co": 0.3, "o3": 30.0},
            "Udara Sedang": {"pm25": 30.0, "pm10": 70.0, "no2": 25.0, "so2": 35.0, "co": 1.5, "o3": 60.0},
            "Udara Kurang Baik": {"pm25": 80.0, "pm10": 180.0, "no2": 180.0, "so2": 220.0, "co": 10.0, "o3": 250.0},
        }
        st.session_state["sim_values"] = presets[name]
        st.session_state["sim_hasil"] = None

    col_left, col_right = st.columns([1.05, 1], gap="medium")

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        head = st.columns([5, 1])
        with head[0]:
            st.markdown(
                "<div class='card-title' style='margin-bottom:0.3rem;'>Komposisi Polutan</div>"
                "<div style='font-size:0.85rem; color:#64748B; line-height:1.5; margin-bottom:1rem;'>"
                "Sesuaikan slider di bawah untuk mensimulasikan kondisi polutan dan memprediksi "
                "Indeks Standar Pencemar Udara (ISPU).</div>",
                unsafe_allow_html=True)
        with head[1]:
            st.markdown("<div style='padding-top:0.3rem;'></div>", unsafe_allow_html=True)
            if st.button("ⓘ Info", key="btn_info_simulasi", use_container_width=True):
                render_popup_polutan()

        st.markdown("<div style='margin-bottom:0.5rem; font-size:0.85rem; color:#475569; font-weight:600;'>Preset</div>",
                    unsafe_allow_html=True)
        pc = st.columns(3, gap="small")
        with pc[0]:
            if st.button("Udara Bersih", key="preset_bersih", use_container_width=True):
                apply_preset("Udara Bersih"); st.rerun()
        with pc[1]:
            if st.button("Udara Sedang", key="preset_sedang", use_container_width=True):
                apply_preset("Udara Sedang"); st.rerun()
        with pc[2]:
            if st.button("Udara Kurang Baik", key="preset_buruk", use_container_width=True):
                apply_preset("Udara Kurang Baik"); st.rerun()

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        sc1, sc2 = st.columns(2, gap="medium")
        vals = st.session_state["sim_values"]

        def slider_block(col, key_state, label, info_key, vmin, vmax, step, unit):
            with col:
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:0.4rem; font-weight:600; color:#0F172A; margin-bottom:0.1rem;'>"
                    f"<span style='width:0.7rem; height:0.7rem; border-radius:999px; background:{INFO_POLUTAN[info_key]['warna']};'></span>"
                    f"{label}</div>"
                    f"<div style='font-size:0.78rem; color:#64748B; margin-bottom:0.3rem;'>"
                    f"{INFO_POLUTAN[info_key]['deskripsi_pendek']}</div>",
                    unsafe_allow_html=True)
                vals[key_state] = st.slider(label, vmin, vmax, vals[key_state], step,
                                             key=f"sl_{key_state}", label_visibility="collapsed")
                st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#64748B;'>{vals[key_state]:.2f} ({unit})</div>",
                            unsafe_allow_html=True)

        slider_block(sc1, "pm25", "PM2.5", "PM2.5", 0.0, 200.0, 0.5, "µg/m³")
        slider_block(sc2, "pm10", "PM10",  "PM10",  0.0, 300.0, 0.5, "µg/m³")
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        sc1, sc2 = st.columns(2, gap="medium")
        slider_block(sc1, "no2", "NO₂", "NO₂", 0.0, 200.0, 0.5, "µg/m³")
        slider_block(sc2, "so2", "SO₂", "SO₂", 0.0, 200.0, 0.5, "µg/m³")
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        sc1, sc2 = st.columns(2, gap="medium")
        slider_block(sc1, "co", "CO", "CO", 0.0, 50.0, 0.1, "mg/m³")
        slider_block(sc2, "o3", "O₃", "O₃", 0.0, 300.0, 0.5, "µg/m³")

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        bc1, bc2, _ = st.columns([1, 1, 2])
        with bc1:
            if st.button("Submit Simulasi", key="btn_submit", type="primary", use_container_width=True):
                st.session_state["sim_hasil"] = prediksi_ispu_xgboost(
                    pm10=vals["pm10"], pm25=vals["pm25"], so2=vals["so2"],
                    co=vals["co"], o3=vals["o3"], no2=vals["no2"]
                )
                st.rerun()
        with bc2:
            if st.button("Reset", key="btn_reset", type="secondary", use_container_width=True):
                st.session_state["sim_values"] = {"pm25": 50.0, "pm10": 70.0, "no2": 25.0,
                                                  "so2": 35.0, "co": 1.5, "o3": 50.0}
                st.session_state["sim_hasil"] = None
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Hasil Prediksi ISPU</div>", unsafe_allow_html=True)
        hasil = st.session_state["sim_hasil"]

        if hasil is None:
            st.markdown(
                """
                <div style='text-align:center; padding:3rem 1rem; color:#94A3B8;'>
                    <div style='font-size:3rem; margin-bottom:0.5rem;'>📊</div>
                    <div style='font-size:0.95rem; font-weight:600; color:#475569;'>
                        Atur slider polutan, lalu klik <strong>Submit Simulasi</strong>
                    </div>
                    <div style='font-size:0.82rem; margin-top:0.4rem;'>Prediksi akan ditampilkan di sini.</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            kat = hasil["kategori"]
            info = KATEGORI_INFO[kat]
            st.markdown(
                f"""
                <div class='hasil-hero'>
                    <div>
                        <div class='hasil-num' style='color:{info["warna"]};'>{hasil["nilai_ispu"]}</div>
                        <div class='hasil-label-ispu'>ISPU</div>
                    </div>
                    <div>
                        <div style='font-size:2.5rem; line-height:1;'>{info["emoji"]}</div>
                        <div class='hero-status' style='color:{info["warna"]}; margin-top:0.4rem;'>Udara {kat}</div>
                        <div class='hero-desc'>{info["deskripsi"]}</div>
                    </div>
                </div>
                <div class='rekom-box' style='border:1px solid {info["warna"]}40; background:{info["warna_bg"]};'>
                    <div class='rekom-box-title' style='color:{info["warna"]};'>Rekomendasi Aktivitas</div>
                    <div class='rekom-box-text' style='color:#334155;'>{info["rekomendasi"]}</div>
                </div>
                """, unsafe_allow_html=True)
            if hasil.get("confidence") is not None:
                st.markdown(
                    f"""
                    <div class='info-box' style='margin-top:1rem;'>
                        <div class='info-box-icon'>ⓘ</div>
                        <div class='info-box-text'>
                            Prediksi dibuat dengan model <strong>XGBoost</strong> 
                            (tingkat keyakinan: <strong>{hasil["confidence"]*100:.1f}%</strong>).
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            if hasil.get("fallback"):
                st.warning("⚠ Model XGBoost belum tersedia; hasil menggunakan formula bobot polutan.")

        st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# HALAMAN 4: EDUKASI & INSIGHT
# ================================================================
def page_edukasi(data):
    st.markdown(
        "<div class='page-title'>Edukasi & Insight</div>"
        "<div class='page-subtitle'>Pelajari kategori ISPU, dampak kesehatan, dan tips menjaga kualitas hidup saat polusi udara meningkat.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'>Mengenal ISPU (Indeks Standar Pencemar Udara)</div>"
        "<div style='font-size:0.88rem; color:#475569; margin-bottom:1.2rem; line-height:1.5;'>"
        "ISPU digunakan untuk menggambarkan kualitas udara ambien di sekitar kita.</div>",
        unsafe_allow_html=True)
    kc = st.columns(5, gap="small")
    for col, (nama, info) in zip(kc, KATEGORI_INFO.items()):
        with col:
            st.markdown(
                f"""
                <div class='kat-card' style='background:{info["warna_bg"]}; border-color:{info["warna"]}40;'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                        <div class='kat-range' style='color:{info["warna"]};'>{info["rentang"]}</div>
                        <div class='kat-emoji'>{info["emoji"]}</div>
                    </div>
                    <div class='kat-name' style='color:{info["warna"]};'>{nama}</div>
                    <div class='kat-desc'>{info["deskripsi"]}</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    dc1, dc2 = st.columns([1.4, 1], gap="medium")

    with dc1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Dampak Kualitas Udara terhadap Kesehatan</div>",
                    unsafe_allow_html=True)
        dampak = [
            ("🫁", "Sistem Pernapasan", "Polusi udara dapat menyebabkan iritasi, batuk, sesak napas, dan memperparah asma."),
            ("❤️", "Sistem Kardiovaskular", "Paparan jangka panjang meningkatkan risiko penyakit jantung dan tekanan darah tinggi."),
            ("👶", "Anak-anak", "Anak lebih rentan terhadap infeksi pernapasan dan gangguan perkembangan paru-paru."),
            ("👴", "Lansia", "Risiko penyakit kronis meningkat, terutama jika memiliki riwayat penyakit."),
        ]
        dr1, dr2 = st.columns(2, gap="medium")
        for idx, (icon, judul, desc) in enumerate(dampak):
            with (dr1 if idx % 2 == 0 else dr2):
                st.markdown(
                    f"""
                    <div style='display:flex; gap:0.85rem; align-items:flex-start; margin-bottom:1.2rem;'>
                        <div style='font-size:1.8rem; flex-shrink:0; line-height:1;'>{icon}</div>
                        <div>
                            <div style='font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:0.25rem;'>{judul}</div>
                            <div style='font-size:0.82rem; color:#475569; line-height:1.5;'>{desc}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with dc2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Sumber Polusi Udara di Jakarta</div>", unsafe_allow_html=True)
        sumber = {
            "Transportasi": (45, "#2563EB"), "Industri": (20, "#16A34A"),
            "Aktivitas Rumah Tangga": (15, "#F59E0B"), "Konstruksi": (10, "#EF4444"),
            "Lainnya": (10, "#7C3AED"),
        }
        chart_col, leg_col = st.columns([1, 1.1], gap="small")
        with chart_col:
            fig = go.Figure(go.Pie(
                labels=list(sumber.keys()),
                values=[v[0] for v in sumber.values()],
                hole=0.6,
                marker=dict(colors=[v[1] for v in sumber.values()],
                            line=dict(color="white", width=3)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>",
            ))
            fig.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=10),
                              showlegend=False, paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with leg_col:
            st.markdown("<div style='padding-top:1rem;'>", unsafe_allow_html=True)
            for nama, (pct, warna) in sumber.items():
                st.markdown(
                    f"""
                    <div class='donut-legend-row'>
                        <div class='donut-legend-left'>
                            <div class='donut-legend-dot' style='background:{warna};'></div>
                            <span>{nama}</span>
                        </div>
                        <div class='donut-legend-pct'>{pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>💡 Tips Menjaga Kesehatan Saat Kualitas Udara Tidak Sehat</div>",
                unsafe_allow_html=True)
    tips = [
        ("😷",  "Gunakan Masker",       "Gunakan masker berstandar untuk mengurangi paparan polusi udara."),
        ("❌",  "Batasi Aktivitas Luar","Kurangi aktivitas fisik berat di luar ruangan, terutama saat sore hingga malam hari."),
        ("💨",  "Ventilasi yang Baik",  "Tutup jendela saat polusi tinggi dan pastikan ventilasi rumah tetap berfungsi baik."),
        ("💧",  "Perbanyak Minum Air",  "Cairan tubuh yang cukup membantu mengurangi efek polutan pada tubuh."),
        ("🌬️", "Gunakan Air Purifier", "Jika memungkinkan, gunakan alat penyaring udara di dalam ruangan untuk udara lebih bersih."),
    ]
    tc = st.columns(5, gap="medium")
    for col, (icon, judul, desc) in zip(tc, tips):
        with col:
            st.markdown(
                f"""
                <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px;
                            padding:1.1rem; height:100%; transition:all 0.25s ease;'>
                    <div style='font-size:2rem; color:#2563EB; margin-bottom:0.6rem; line-height:1;'>{icon}</div>
                    <div style='font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:0.4rem;'>{judul}</div>
                    <div style='font-size:0.78rem; color:#64748B; line-height:1.5;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# MAIN ROUTER
# ================================================================
def main():
    inject_css()
    data = load_data()
    render_sidebar()

    page = st.session_state.get("current_page", "Dashboard")
    if page == "Dashboard":
        page_dashboard(data)
    elif page == "Detail Wilayah":
        page_detail_wilayah(data)
    elif page == "Simulasi Prediksi ISPU":
        page_simulasi(data)
    elif page == "Edukasi & Insight":
        page_edukasi(data)


if __name__ == "__main__":
    main()
